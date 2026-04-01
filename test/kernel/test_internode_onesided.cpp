/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Benchmark for SDCCL inter-node one-sided AlltoAll using Device API.
 *
 * Tests one-sided put + waitSignal + flush (NCCL GIN AlltoAll pattern):
 *   All ranks: put data to all peers, signal each peer, waitSignal, flush.
 *
 * Works with N MPI ranks (N >= 2).
 *
 * Usage: mpirun -np N ./test_kernel_internode_onesided [options]
 *   -b <minbytes>  -e <maxbytes>  -f <stepfactor>
 *   -w <warmup>    -n <iters>     -p <printbuffer 0/1>
 *   -R <regMode>   0=raw(deviceMalloc), 1=IPC(sdcclMemAlloc+CommRegister),
 *                  2=window(sdcclMemAlloc+CommWindowRegister)
 *   One-sided ops require -R 1 or -R 2.
 ************************************************************************/

#include "sdccl.h"
#include "sdccl_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <unistd.h>

#define DATATYPE sdcclFloat

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t minBytes = args.getMinBytes();
  size_t maxBytes = args.getMaxBytes();
  int stepFactor = args.getStepFactor();
  int numWarmupIters = args.getWarmupIters();
  int numIters = args.getTestIters();
  int printBuffer = args.isPrintBuffer();
  uint64_t splitMask = args.getSplitMask();
  int localRegister = args.getLocalRegister();

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

  if (totalProcs < 2) {
    if (proc == 0)
      printf("test_kernel_internode_onesided requires at least 2 ranks.\n");
    SDCCLCHECK(sdcclCommDestroy(comm));
    SDCCLCHECK(sdcclHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  if (localRegister == 0) {
    if (proc == 0)
      printf("One-sided ops require -R 1 or -R 2. Skipping.\n");
    SDCCLCHECK(sdcclCommDestroy(comm));
    SDCCLCHECK(sdcclHandleFree(handler));
    MPI_Finalize();
    return 0;
  }

  // AlltoAll buffer layout: [rank0_data][rank1_data]...[rankN_data]
  // Each chunk has countPerPeer elements (= maxBytes / nRanks / sizeof(float))
  // Total buffer size = maxBytes (contains data for all peers)

  void *sendBuff = nullptr, *recvBuff = nullptr;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  sdcclWindow_t sendWin = nullptr, recvWin = nullptr;

  if (localRegister == 0) {
    SDCCLCHECK(
        devHandle->deviceMalloc(&sendBuff, maxBytes, sdcclMemDevice, NULL));
    SDCCLCHECK(
        devHandle->deviceMalloc(&recvBuff, maxBytes, sdcclMemDevice, NULL));
  } else {
    SDCCLCHECK(sdcclMemAlloc(&sendBuff, maxBytes));
    SDCCLCHECK(sdcclMemAlloc(&recvBuff, maxBytes));
  }

  if (localRegister == 2) {
    SDCCLCHECK(sdcclCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
    SDCCLCHECK(sdcclCommWindowRegister(comm, recvBuff, maxBytes, &recvWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
  } else if (localRegister == 1) {
    SDCCLCHECK(sdcclCommRegister(comm, sendBuff, maxBytes, &sendHandle));
    SDCCLCHECK(sdcclCommRegister(comm, recvBuff, maxBytes, &recvHandle));
  }

  sdcclStream_t stream;
  SDCCLCHECK(devHandle->streamCreate(&stream));

  // Host buffer for data preparation and verification
  void *hostBuff = malloc(maxBytes);
  memset(hostBuff, 0, maxBytes);

  // Create device communicator with inter-node barrier + signal
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 1;
  sdcclDevComm_t devComm = nullptr;
  SDCCLCHECK(sdcclDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handles
  sdcclDevMem_t sendMem = nullptr, recvMem = nullptr;
  SDCCLCHECK(sdcclDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  SDCCLCHECK(sdcclDevMemCreate(comm, recvBuff, maxBytes, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("# SDCCL Device API Inter-node One-sided AlltoAll Benchmark\n");
    printf("# nRanks: %d, regMode: %s\n", totalProcs,
           localRegister == 2   ? "window"
           : localRegister == 1 ? "ipc"
                                : "raw (no registration)");
    printf("# %-12s %-14s %-14s %-8s\n", "Size(B)", "Time(us)", "BW(GB/s)",
           "Result");
  }

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    size_t countPerPeer =
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs);
    SDCCLCHECK(sdcclInterOneSidedAlltoAll(sendMem, recvMem, countPerPeer,
                                            DATATYPE, devComm, stream));
  }
  SDCCLCHECK(devHandle->streamSynchronize(stream));

  // Benchmark loop
  timer tim;
  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    size_t countPerPeer = size / sizeof(float) / totalProcs;
    if (countPerPeer == 0)
      countPerPeer = 1;

    // Initialize sendbuff: sendbuff[r * countPerPeer + i] =
    //   proc * 1000 + r * 100 + i
    // After alltoall: recvbuff[src * countPerPeer + i] =
    //   src * 1000 + proc * 100 + i
    float *hostFloat = (float *)hostBuff;
    for (int r = 0; r < totalProcs; r++) {
      for (size_t i = 0; i < countPerPeer; i++) {
        hostFloat[r * countPerPeer + i] =
            (float)(proc * 1000 + r * 100 + (int)i);
      }
    }
    SDCCLCHECK(devHandle->deviceMemcpy(sendBuff, hostBuff, size,
                                        sdcclMemcpyHostToDevice, NULL));
    // Clear recvbuff
    SDCCLCHECK(
        devHandle->deviceMemset(recvBuff, 0, size, sdcclMemDevice, NULL));

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      SDCCLCHECK(sdcclInterOneSidedAlltoAll(sendMem, recvMem, countPerPeer,
                                              DATATYPE, devComm, stream));
    }
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    double elapsedTime = tim.elapsed() / numIters;

    // Verify correctness
    memset(hostBuff, 0, size);
    SDCCLCHECK(devHandle->deviceMemcpy(hostBuff, recvBuff, size,
                                        sdcclMemcpyDeviceToHost, NULL));
    hostFloat = (float *)hostBuff;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++) {
      for (size_t i = 0; i < countPerPeer && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (hostFloat[src * countPerPeer + i] != expected) {
          correct = false;
          if (proc == 0) {
            printf("  MISMATCH rank%d recvbuff[%d*%zu+%zu]: got %.0f expected "
                   "%.0f\n",
                   proc, src, countPerPeer, i,
                   hostFloat[src * countPerPeer + i], expected);
          }
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;
    double bandwidth = (double)size / 1.0e9 / elapsedTime;

    if (proc == 0 && color == 0) {
      printf("  %-12zu %-14.3lf %-14.3lf %-8s\n", size, elapsedTime * 1e6,
             bandwidth, correct ? "PASS" : "FAIL");
    }

    if (printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        float *sendFloat = (float *)hostBuff;
        // Re-read sendbuff for display
        devHandle->deviceMemcpy(hostBuff, sendBuff, size,
                                sdcclMemcpyDeviceToHost, NULL);
        sendFloat = (float *)hostBuff;
        printf(" %.0f", sendFloat[p * countPerPeer]);
      }
      printf("\n");
      // Re-read recvbuff for display
      devHandle->deviceMemcpy(hostBuff, recvBuff, size,
                              sdcclMemcpyDeviceToHost, NULL);
      hostFloat = (float *)hostBuff;
      printf("rank%d recvbuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", hostFloat[p * countPerPeer]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Destroy stream first (sync any pending work)
  SDCCLCHECK(devHandle->streamDestroy(stream));

  // Destroy device memory handles
  SDCCLCHECK(sdcclDevMemDestroy(comm, sendMem));
  SDCCLCHECK(sdcclDevMemDestroy(comm, recvMem));

  // Destroy device communicator (before comm destroy)
  SDCCLCHECK(sdcclDevCommDestroy(comm, devComm));

  // Deregister buffer (before comm destroy)
  if (localRegister == 2) {
    SDCCLCHECK(sdcclCommWindowDeregister(comm, sendWin));
    SDCCLCHECK(sdcclCommWindowDeregister(comm, recvWin));
  } else if (localRegister == 1) {
    SDCCLCHECK(sdcclCommDeregister(comm, sendHandle));
    SDCCLCHECK(sdcclCommDeregister(comm, recvHandle));
  }

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  SDCCLCHECK(sdcclCommDestroy(comm));

  // Free buffer
  if (localRegister >= 1) {
    SDCCLCHECK(sdcclMemFree(sendBuff));
    SDCCLCHECK(sdcclMemFree(recvBuff));
  } else if (localRegister == 0) {
    SDCCLCHECK(devHandle->deviceFree(sendBuff, sdcclMemDevice, NULL));
    SDCCLCHECK(devHandle->deviceFree(recvBuff, sdcclMemDevice, NULL));
  }
  free(hostBuff);
  SDCCLCHECK(sdcclHandleFree(handler));

  MPI_Finalize();
  return 0;
}
