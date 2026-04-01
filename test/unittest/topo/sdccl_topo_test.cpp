#include "sdccl_topo_test.hpp"
#include <iostream>

void SDCCLTopoTest::SetUp() {
  SDCCLTest::SetUp();
  std::cout << "rank = " << rank << "; nranks = " << nranks << std::endl;

  // initialize sdccl handles
  sdcclHandleInit(&handler);
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  if (rank == 0)
    sdcclGetUniqueId(&uniqueId);
  std::cout << "finished getting uniqueId" << std::endl;
  MPI_Bcast((void *)uniqueId, sizeof(sdcclUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  /*
  we don't initialize communicator here
  because topology detection is part of communicator initialization
  */
}

void SDCCLTopoTest::TearDown() {
  sdcclComm_t &comm = handler->comm;
  sdcclCommDestroy(comm);

  sdcclHandleFree(handler);
}
