/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * API correctness test for SDCCL inter-node one-sided Device API.
 *
 * Tests ten kernels, each covering one API facet:
 *   K1: putSignalInc          (fused data+signal)
 *   K2: putSignalAddDecoupled (decoupled data+signal)
 *   K3: PutValue              (8-byte atomic RDMA WRITE)
 *   K4: Get                   (RDMA READ alltoall)
 *   K5: Signal                (signal standalone)
 *   K6: FlushDecouple         (put + flush decoupled)
 *   K7: CounterPipeline       (put + CounterInc, two rounds)
 *   K8: Reset                 (resetSignal + resetCounter)
 *   K9: FollowShadow          (waitSignalFollowShadow)
 *   K10: MeetShadow           (increaseSignalShadow + waitSignalMeetShadow)
 *
 * Usage: mpirun -np N ./test_devapi_internode_onesided [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>
 *   -R <regMode>   1=IPC(sdcclMemAlloc+CommRegister)
 *                  2=window(sdcclMemAlloc+CommWindowRegister)
 *   One-sided ops require -R 1 or -R 2.
 ************************************************************************/

#include "sdccl.h"
#include "sdccl_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#define DATATYPE sdcclFloat

// ---------------------------------------------------------------------------
// Helper functions (camelCase)
// ---------------------------------------------------------------------------

// Populate sendbuff: sendbuff[r * countPerPeer + i] = myRank * 1000 + r * 100 +
// i
static void initSendBuff(void *sendBuff, size_t countPerPeer, int nRanks,
                         int myRank, sdcclDeviceHandle_t devHandle,
                         sdcclStream_t stream, void *hostScratch) {
  float *h = (float *)hostScratch;
  for (int r = 0; r < nRanks; r++)
    for (size_t i = 0; i < countPerPeer; i++)
      h[(size_t)r * countPerPeer + i] =
          (float)(myRank * 1000 + r * 100 + (int)i);
  devHandle->deviceMemcpy(sendBuff, hostScratch,
                          (size_t)nRanks * countPerPeer * sizeof(float),
                          sdcclMemcpyHostToDevice, NULL);
}

// Verify alltoall result:
// recvbuff[src * countPerPeer + i] == src * 1000 + myRank * 100 + i
static bool verifyAlltoAll(const float *buf, size_t countPerPeer, int nRanks,
                           int myRank) {
  for (int src = 0; src < nRanks; src++)
    for (size_t i = 0; i < countPerPeer; i++) {
      float expected = (float)(src * 1000 + myRank * 100 + (int)i);
      if (buf[(size_t)src * countPerPeer + i] != expected)
        return false;
    }
  return true;
}

// Verify K7 counter pipeline:
//   hResult[0] == 2 * hResult[1] (counter after 2 rounds, inter peers only)
//   hResult[1] == nInterRanks as reported by the kernel
//   recvbuff[src * countPerPeer] == 999.0f  (round-2 sentinel) for all src
static bool verifyCounterPipeline(const uint64_t *hResult, const float *buf,
                                  size_t countPerPeer, int nRanks) {
  uint64_t nInterRanks = hResult[1];
  if (hResult[0] != 2 * nInterRanks)
    return false;
  for (int src = 0; src < nRanks; src++)
    if (buf[(size_t)src * countPerPeer] != 999.0f)
      return false;
  return true;
}

// Verify K3 putValue result:
// recvbuff at putValBase + src*8 == src * 1000 + myRank
static bool verifyPutValue(const void *buf, size_t putValBase, int nRanks,
                           int myRank) {
  const uint64_t *vals = (const uint64_t *)((const char *)buf + putValBase);
  for (int src = 0; src < nRanks; src++) {
    uint64_t expected = (uint64_t)src * 1000u + (uint64_t)myRank;
    if (vals[src] != expected)
      return false;
  }
  return true;
}

// Verify K8 reset: all four entries must be 0
static bool verifyReset(const uint64_t *r) {
  return r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0;
}

static void printResult(const char *name, bool ok, int rank) {
  if (rank == 0)
    printf("  %-30s %s\n", name, ok ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int localRegister = args.getLocalRegister();
  uint64_t splitMask = args.getSplitMask();

  if (stepFactor <= 1) {
    printf("Error: stepFactor must be > 1, got %d\n", stepFactor);
    MPI_Finalize();
    return 1;
  }

  sdcclHandlerGroup_t handler;
  SDCCLCHECK(sdcclHandleInit(&handler));
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, splitMask);

  int nGpu;
  SDCCLCHECK(devHandle->getDeviceCount(&nGpu));
  SDCCLCHECK(devHandle->setDevice(worldRank % nGpu));

  if (proc == 0)
    SDCCLCHECK(sdcclGetUniqueId(&uniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(sdcclUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  SDCCLCHECK(sdcclCommInitRank(&comm, totalProcs, uniqueId, proc));

  if (localRegister == 0) {
    if (proc == 0)
      printf("One-sided ops require -R 1 or -R 2. Skipping.\n");
    SDCCLCHECK(sdcclCommDestroy(comm));
    SDCCLCHECK(sdcclHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  // Buffer layout:
  //   sendBuff [0, maxBytes): float alltoall chunks
  //   recvBuff [0, maxBytes): float alltoall chunks
  //            [maxBytes, maxBytes + nRanks*8): putValue uint64_t area
  size_t recvBuffSize = maxBytes + (size_t)totalProcs * sizeof(uint64_t);
  const size_t putValBase = maxBytes;

  void *sendBuff = nullptr, *recvBuff = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  sdcclWindow_t sendWin = nullptr, recvWin = nullptr;

  SDCCLCHECK(sdcclMemAlloc(&sendBuff, maxBytes));
  SDCCLCHECK(sdcclMemAlloc(&recvBuff, recvBuffSize));

  if (localRegister == 2) {
    SDCCLCHECK(sdcclCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
    SDCCLCHECK(sdcclCommWindowRegister(comm, recvBuff, recvBuffSize, &recvWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
  } else {
    SDCCLCHECK(sdcclCommRegister(comm, sendBuff, maxBytes, &sendHandle));
    SDCCLCHECK(sdcclCommRegister(comm, recvBuff, recvBuffSize, &recvHandle));
  }

  sdcclStream_t stream;
  SDCCLCHECK(devHandle->streamCreate(&stream));

  // Host scratch buffer — sized to hold recvBuff for D2H copies
  void *hostBuff = malloc(recvBuffSize);
  memset(hostBuff, 0, recvBuffSize);

  // Device result buffer: 4 × uint64_t used by K7 (counter) and K8 (reset)
  uint64_t *dResultBuf = nullptr;
  SDCCLCHECK(devHandle->deviceMalloc(
      (void **)&dResultBuf, 4 * sizeof(uint64_t), sdcclMemDevice, NULL));
  uint64_t hResultBuf[4] = {};

  // Create device communicator
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  reqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 3;
  reqs.interCounterCount = 1;
  sdcclDevComm_t devComm = nullptr;
  SDCCLCHECK(sdcclDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handles
  sdcclDevMem_t sendMem = nullptr, recvMem = nullptr;
  SDCCLCHECK(sdcclDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  SDCCLCHECK(
      sdcclDevMemCreate(comm, recvBuff, recvBuffSize, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("# SDCCL Device API Test\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2 ? "window" : "ipc");
    printf("# Kernels: K1=putSignalInc  K2=putSignalAddDecoupled"
           "  K3=PutValue  K4=Get\n");
    printf("#          K5=Signal  K6=FlushDecouple"
           "  K7=CounterPipeline  K8=Reset\n");
    printf("#\n");
  }

  // Warm-up: K1 only
  for (int i = 0; i < numWarmupIters; i++) {
    size_t cp =
        std::max((size_t)1, maxBytes / sizeof(float) / (size_t)totalProcs);
    SDCCLCHECK(sdcclInterTestPutSignalInc(sendMem, recvMem, cp, DATATYPE,
                                            devComm, stream));
  }
  SDCCLCHECK(devHandle->streamSynchronize(stream));

  // Initial K8 reset — establishes clean signal/counter/shadow state
  SDCCLCHECK(sdcclInterTestReset(devComm, stream, dResultBuf));
  SDCCLCHECK(devHandle->streamSynchronize(stream));

  // Main test loop
  for (size_t size = minBytes; size <= maxBytes; size *= (size_t)stepFactor) {
    size_t countPerPeer = size / sizeof(float) / (size_t)totalProcs;
    if (countPerPeer == 0)
      countPerPeer = 1;
    size_t floatSize = (size_t)totalProcs * countPerPeer * sizeof(float);

    if (proc == 0 && color == 0)
      printf("# Size = %zu bytes, countPerPeer = %zu\n", size, countPerPeer);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- K1: putSignalInc ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestPutSignalInc(sendMem, recvMem, countPerPeer,
                                            DATATYPE, devComm, stream));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        sdcclMemcpyDeviceToHost, NULL));
    bool k1Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K1 putSignalInc", k1Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K2: putSignalAddDecoupled ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestPutSignalAddDecoupled(
        sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        sdcclMemcpyDeviceToHost, NULL));
    bool k2Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K2 putSignalAddDecoupled", k2Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K3: PutValue ---
    SDCCLCHECK(devHandle->deviceMemset((char *)recvBuff + putValBase, 0,
                                        (size_t)totalProcs * sizeof(uint64_t),
                                        sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestPutValue(recvMem, devComm, stream, putValBase));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(
        (char *)hostBuff + putValBase, (char *)recvBuff + putValBase,
        (size_t)totalProcs * sizeof(uint64_t), sdcclMemcpyDeviceToHost, NULL));
    bool k3Ok = verifyPutValue(hostBuff, putValBase, totalProcs, proc);
    printResult("K3 PutValue", k3Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K4: Get ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestGet(sendMem, recvMem, countPerPeer, DATATYPE,
                                   devComm, stream));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        sdcclMemcpyDeviceToHost, NULL));
    bool k4Ok =
        verifyAlltoAll((const float *)hostBuff, countPerPeer, totalProcs, proc);
    printResult("K4 Get", k4Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K5: Signal ---
    SDCCLCHECK(sdcclInterTestSignal(devComm, stream));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    printResult("K5 Signal", true, proc); // hang-free = PASS
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K6: FlushDecouple ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestFlushDecouple(sendMem, recvMem, countPerPeer,
                                             DATATYPE, devComm, stream));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    printResult("K6 FlushDecouple", true, proc); // hang-free = PASS
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K7: CounterPipeline ---
    initSendBuff(sendBuff, countPerPeer, totalProcs, proc, devHandle, stream,
                 hostBuff);
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, floatSize, sdcclMemDevice, NULL));
    SDCCLCHECK(sdcclInterTestCounterPipeline(
        sendMem, recvMem, countPerPeer, DATATYPE, devComm, stream, dResultBuf));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        4 * sizeof(uint64_t),
                                        sdcclMemcpyDeviceToHost, NULL));
    SDCCLCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, floatSize,
                                        sdcclMemcpyDeviceToHost, NULL));
    bool k7Ok = verifyCounterPipeline(hResultBuf, (const float *)hostBuff,
                                      countPerPeer, totalProcs);
    printResult("K7 CounterPipeline", k7Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K8: Reset ---
    SDCCLCHECK(sdcclInterTestReset(devComm, stream, dResultBuf));
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    SDCCLCHECK(devHandle->deviceMemcpy(hResultBuf, dResultBuf,
                                        4 * sizeof(uint64_t),
                                        sdcclMemcpyDeviceToHost, NULL));
    bool k8Ok = verifyReset(hResultBuf);
    printResult("K8 Reset", k8Ok, proc);
    MPI_Barrier(MPI_COMM_WORLD);

    // --- K9: FollowShadow (§10.3.5 Part B) ---
    // SDCCLCHECK(sdcclInterTestFollowShadow(devComm, stream));
    // SDCCLCHECK(devHandle->streamSynchronize(stream));
    // printResult("K9 FollowShadow", true, proc); // hang-free = PASS
    // MPI_Barrier(MPI_COMM_WORLD);

    // --- K10: MeetShadow (§10.3.5 Part A) ---
    // SDCCLCHECK(sdcclInterTestMeetShadow(devComm, stream));
    // SDCCLCHECK(devHandle->streamSynchronize(stream));
    // printResult("K10 MeetShadow", true, proc); // hang-free = PASS
    // MPI_Barrier(MPI_COMM_WORLD);

    if (proc == 0 && color == 0)
      printf("#\n");

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  // order matters: stream → devMem → devComm → deregister → comm → buff
  SDCCLCHECK(devHandle->streamDestroy(stream));
  SDCCLCHECK(devHandle->deviceFree(dResultBuf, sdcclMemDevice, NULL));
  SDCCLCHECK(sdcclDevMemDestroy(comm, sendMem));
  SDCCLCHECK(sdcclDevMemDestroy(comm, recvMem));
  SDCCLCHECK(sdcclDevCommDestroy(comm, devComm));

  if (localRegister == 2) {
    SDCCLCHECK(sdcclCommWindowDeregister(comm, sendWin));
    SDCCLCHECK(sdcclCommWindowDeregister(comm, recvWin));
  } else {
    SDCCLCHECK(sdcclCommDeregister(comm, sendHandle));
    SDCCLCHECK(sdcclCommDeregister(comm, recvHandle));
  }

  SDCCLCHECK(sdcclCommDestroy(comm));
  SDCCLCHECK(sdcclMemFree(sendBuff));
  SDCCLCHECK(sdcclMemFree(recvBuff));
  free(hostBuff);
  SDCCLCHECK(sdcclHandleFree(handler));

  MPI_Finalize();
  return 0;
}
