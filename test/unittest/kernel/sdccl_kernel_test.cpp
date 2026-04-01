#include "sdccl_kernel_test.hpp"
#include <cstring>
#include <iostream>

void SDCCLKernelTest::SetUp() {
  SDCCLTest::SetUp();

  // initialize sdccl handles
  sdcclHandleInit(&handler);
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;
  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  size = 1ULL * 1024 * 1024; // 1MB for kernel test
  count = size / sizeof(float);

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  // Create and broadcast uniqueId
  if (rank == 0)
    sdcclGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(sdcclUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  // Create comm and stream
  sdcclCommInitRank(&comm, nranks, uniqueId, rank);
  devHandle->streamCreate(&stream);

  // allocate device buffers
  devHandle->deviceMalloc(&sendbuff, size, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, size, sdcclMemDevice, NULL);

  // allocate host buffers
  hostsendbuff = malloc(size);
  memset(hostsendbuff, 0, size);
  hostrecvbuff = malloc(size);
  memset(hostrecvbuff, 0, size);
}

void SDCCLKernelTest::TearDown() {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  // Destroy stream first (sync any pending work)
  devHandle->streamDestroy(stream);

  // Destroy comm to stop kernel proxy thread BEFORE freeing device memory
  // The kernel proxy thread holds a CUDA stream that can interfere with
  // deviceFree
  sdcclCommDestroy(comm);

  // free data
  devHandle->deviceFree(sendbuff, sdcclMemDevice, NULL);
  devHandle->deviceFree(recvbuff, sdcclMemDevice, NULL);
  free(hostsendbuff);
  free(hostrecvbuff);

  // free handles
  sdcclHandleFree(handler);

  SDCCLTest::TearDown();
}

void SDCCLKernelTest::Run() {}
