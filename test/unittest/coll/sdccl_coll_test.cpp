#include "sdccl_coll_test.hpp"
#include <iostream>

void SDCCLCollTest::SetUp() {
  SDCCLTest::SetUp();
  std::cout << "rank = " << rank << "; nranks = " << nranks << std::endl;

  // initialize sdccl handles
  sdcclHandleInit(&handler);
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;
  sendbuff = nullptr;
  recvbuff = nullptr;
  hostsendbuff = nullptr;
  hostrecvbuff = nullptr;
  size = 1ULL * 1024 * 1024 * 1024; // 1GB
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

  // allocate data and set inital value
  devHandle->deviceMalloc(&sendbuff, size, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, size, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&hostsendbuff, size, sdcclMemHost, NULL);
  devHandle->deviceMemset(hostsendbuff, 0, size, sdcclMemHost, NULL);
  devHandle->deviceMalloc(&hostrecvbuff, size, sdcclMemHost, NULL);
  devHandle->deviceMemset(hostrecvbuff, 0, size, sdcclMemHost, NULL);
}

void SDCCLCollTest::TearDown() {
  // destroy comm
  sdcclComm_t &comm = handler->comm;
  sdcclCommDestroy(comm);

  // destroy stream
  sdcclDeviceHandle_t &devHandle = handler->devHandle;
  devHandle->streamDestroy(stream);

  // free data
  devHandle->deviceFree(sendbuff, sdcclMemDevice, NULL);
  devHandle->deviceFree(recvbuff, sdcclMemDevice, NULL);
  devHandle->deviceFree(hostsendbuff, sdcclMemHost, NULL);
  devHandle->deviceFree(hostrecvbuff, sdcclMemHost, NULL);

  // free handles
  sdcclHandleFree(handler);

  SDCCLTest::TearDown();
}

void SDCCLCollTest::Run() {}
