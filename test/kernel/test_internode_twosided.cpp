/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "sdccl.h"
#include "sdccl_kernel.h"
#include "tools.h"
#include <algorithm>
#include <cstring>
#include <iostream>

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

  sdcclStream_t stream;
  SDCCLCHECK(devHandle->streamCreate(&stream));

  void *sendBuff = nullptr, *recvBuff = nullptr, *hello;
  void *sendHandle = nullptr, *recvHandle = nullptr;
  sdcclWindow_t sendWin = nullptr, recvWin = nullptr;
  size_t count;
  timer tim;

  if (localRegister == 2) {
    // Window mode: VMM alloc with comm (for sdcclCommWindowRegister later)
    SDCCLCHECK(sdcclMemAlloc(&sendBuff, maxBytes));
    SDCCLCHECK(sdcclMemAlloc(&recvBuff, maxBytes));
    SDCCLCHECK(sdcclCommWindowRegister(comm, sendBuff, maxBytes, &sendWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
    SDCCLCHECK(sdcclCommWindowRegister(comm, recvBuff, maxBytes, &recvWin,
                                         SDCCL_WIN_COLL_SYMMETRIC));
  } else if (localRegister == 1) {
    // Zero-copy: alloc + register for NIC RDMA access
    SDCCLCHECK(sdcclMemAlloc(&sendBuff, maxBytes));
    SDCCLCHECK(sdcclMemAlloc(&recvBuff, maxBytes));
    SDCCLCHECK(sdcclCommRegister(comm, sendBuff, maxBytes, &sendHandle));
    SDCCLCHECK(sdcclCommRegister(comm, recvBuff, maxBytes, &recvHandle));
  } else {
    // Unregistered
    SDCCLCHECK(
        devHandle->deviceMalloc(&sendBuff, maxBytes, sdcclMemDevice, NULL));
    SDCCLCHECK(
        devHandle->deviceMalloc(&recvBuff, maxBytes, sdcclMemDevice, NULL));
  }
  hello = malloc(maxBytes);
  memset(hello, 0, maxBytes);

  // Create device communicator for AlltoAll demo
  // Inter-only barrier needs inter barrier resources (GIN/FIFO Signal)
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  sdcclDevComm_t devComm = nullptr;
  SDCCLCHECK(sdcclDevCommCreate(comm, &reqs, &devComm));

  // Create device memory handles for send/recv buffers
  sdcclDevMem_t sendMem = nullptr, recvMem = nullptr;
  SDCCLCHECK(sdcclDevMemCreate(comm, sendBuff, maxBytes, sendWin, &sendMem));
  SDCCLCHECK(sdcclDevMemCreate(comm, recvBuff, maxBytes, recvWin, &recvMem));

  if (proc == 0 && color == 0) {
    printf("\n# FIFO AlltoAll test (two-sided send/recv, -R %d)\n",
           localRegister);
  }

  // Warm-up
  for (int i = 0; i < numWarmupIters; i++) {
    SDCCLCHECK(sdcclInterTwoSidedAlltoAll(
        sendMem, recvMem,
        std::max((size_t)1, maxBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  SDCCLCHECK(devHandle->streamSynchronize(stream));
  for (int i = 0; i < numWarmupIters; i++) {
    SDCCLCHECK(sdcclInterTwoSidedAlltoAll(
        sendMem, recvMem,
        std::max((size_t)1, minBytes / sizeof(float) / totalProcs), DATATYPE,
        devComm, stream));
  }
  SDCCLCHECK(devHandle->streamSynchronize(stream));

  for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
    count = size / sizeof(float) / totalProcs;
    if (count == 0)
      count = 1;

    // Initialize sendBuff: sendBuff[r * count + i] = proc * 1000 + r * 100 + i
    // After alltoall: recvBuff[src * count + i] = src * 1000 + proc * 100 + i
    float *helloFloat = (float *)hello;
    for (int r = 0; r < totalProcs; r++) {
      for (size_t i = 0; i < count; i++) {
        helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
      }
    }
    SDCCLCHECK(devHandle->deviceMemcpy(sendBuff, hello, size,
                                        sdcclMemcpyHostToDevice, NULL));
    memset(hello, 0, size);
    SDCCLCHECK(devHandle->deviceMemcpy(recvBuff, hello, size,
                                        sdcclMemcpyHostToDevice, NULL));

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d sendBuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < numIters; i++) {
      SDCCLCHECK(sdcclInterTwoSidedAlltoAll(sendMem, recvMem, count, DATATYPE,
                                              devComm, stream));
    }
    SDCCLCHECK(devHandle->streamSynchronize(stream));
    double elapsedTime = tim.elapsed() / numIters;

    // Verify correctness
    memset(hello, 0, size);
    SDCCLCHECK(devHandle->deviceMemcpy(hello, recvBuff, size,
                                        sdcclMemcpyDeviceToHost, NULL));
    helloFloat = (float *)hello;
    bool correct = true;
    for (int src = 0; src < totalProcs && correct; src++) {
      for (size_t i = 0; i < count && correct; i++) {
        float expected = (float)(src * 1000 + proc * 100 + (int)i);
        if (helloFloat[src * count + i] != expected) {
          correct = false;
          if (proc == 0) {
            printf("  MISMATCH at recvBuff[%d*%zu+%zu]: got %.0f expected "
                   "%.0f\n",
                   src, count, i, helloFloat[src * count + i], expected);
          }
        }
      }
    }

    if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
      printf("rank%d recvBuff:", proc);
      for (int p = 0; p < totalProcs; p++) {
        printf(" %.0f", helloFloat[p * count]);
      }
      printf("\n");
    }

    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsedTime /= worldSize;
    double bw = (double)(size) / 1.0E9 / elapsedTime;

    if (proc == 0 && color == 0) {
      printf("FIFO AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n", size,
             elapsedTime * 1e6, bw, correct ? "PASS" : "FAIL");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ==========================================================================
  // Window AlltoAll test (requires -R 2 for window registration)
  // Falls back to FIFO AlltoAll on Fallback when window not available
  // ==========================================================================
  if (localRegister == 2) {
    sdcclDevComm_t a2aDevComm = nullptr;
    sdcclDevMem_t a2aSendMem = nullptr, a2aRecvMem = nullptr;

    if (proc == 0 && color == 0) {
      printf("\n# Window AlltoAll test (two-sided send/recv, -R 2)\n");
    }

    sdcclDevCommRequirements a2aReqs =
        SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    a2aReqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
    SDCCLCHECK(sdcclDevCommCreate(comm, &a2aReqs, &a2aDevComm));

    SDCCLCHECK(
        sdcclDevMemCreate(comm, sendBuff, maxBytes, sendWin, &a2aSendMem));
    SDCCLCHECK(
        sdcclDevMemCreate(comm, recvBuff, maxBytes, recvWin, &a2aRecvMem));

    for (size_t size = minBytes; size <= maxBytes; size *= stepFactor) {
      count = size / sizeof(float) / totalProcs;
      if (count == 0)
        count = 1;

      // Initialize sendBuff: sendBuff[r * count + i] = proc * 1000 + r * 100 +
      // i After alltoall: recvBuff[src * count + i] = src * 1000 + proc * 100 +
      // i
      float *helloFloat = (float *)hello;
      for (int r = 0; r < totalProcs; r++) {
        for (size_t i = 0; i < count; i++) {
          helloFloat[r * count + i] = (float)(proc * 1000 + r * 100 + (int)i);
        }
      }
      SDCCLCHECK(devHandle->deviceMemcpy(sendBuff, hello, size,
                                          sdcclMemcpyHostToDevice, NULL));
      memset(hello, 0, size);
      SDCCLCHECK(devHandle->deviceMemcpy(recvBuff, hello, size,
                                          sdcclMemcpyHostToDevice, NULL));

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d sendBuff:", proc);
        for (int p = 0; p < totalProcs; p++) {
          printf(" %.0f", helloFloat[p * count]);
        }
        printf("\n");
      }

      MPI_Barrier(MPI_COMM_WORLD);

      tim.reset();
      for (int i = 0; i < numIters; i++) {
        SDCCLCHECK(sdcclInterTwoSidedAlltoAll(a2aSendMem, a2aRecvMem, count,
                                                DATATYPE, a2aDevComm, stream));
      }
      SDCCLCHECK(devHandle->streamSynchronize(stream));
      double elapsedTime = tim.elapsed() / numIters;

      // Verify correctness
      memset(hello, 0, size);
      SDCCLCHECK(devHandle->deviceMemcpy(hello, recvBuff, size,
                                          sdcclMemcpyDeviceToHost, NULL));
      helloFloat = (float *)hello;
      bool correct = true;
      for (int src = 0; src < totalProcs && correct; src++) {
        for (size_t i = 0; i < count && correct; i++) {
          float expected = (float)(src * 1000 + proc * 100 + (int)i);
          if (helloFloat[src * count + i] != expected) {
            correct = false;
            if (proc == 0) {
              printf("  MISMATCH at recvBuff[%d*%zu+%zu]: got %.0f expected "
                     "%.0f\n",
                     src, count, i, helloFloat[src * count + i], expected);
            }
          }
        }
      }

      if (color == 0 && printBuffer && (proc == 0 || proc == totalProcs - 1)) {
        printf("rank%d recvBuff:", proc);
        for (int p = 0; p < totalProcs; p++) {
          printf(" %.0f", helloFloat[p * count]);
        }
        printf("\n");
      }

      MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsedTime, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsedTime /= worldSize;
      double bw = (double)(size) / 1.0E9 / elapsedTime;

      if (proc == 0 && color == 0) {
        printf("Window AlltoAll %zu bytes; %.3lf us; %.3lf GB/s; %s\n", size,
               elapsedTime * 1e6, bw, correct ? "PASS" : "FAIL");
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Cleanup
    SDCCLCHECK(sdcclDevMemDestroy(comm, a2aSendMem));
    SDCCLCHECK(sdcclDevMemDestroy(comm, a2aRecvMem));
    SDCCLCHECK(sdcclDevCommDestroy(comm, a2aDevComm));
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
  free(hello);
  SDCCLCHECK(sdcclHandleFree(handler));

  MPI_Finalize();
  return 0;
}
