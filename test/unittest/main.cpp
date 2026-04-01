#include "sdccl_coll_test.hpp"
#include "sdccl_kernel_test.hpp"
#include "sdccl_topo_test.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>

#define BASELINE_FILE "baseline_result.txt"
#define NUM_BASELINE_ENTRIES 1000

TEST_F(SDCCLCollTest, AllReduce) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclAllReduce(sendbuff, recvbuff, count, sdcclFloat, sdcclSum, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  for (size_t i = 0; i < count; i++) {
    ((float *)hostrecvbuff)[i] /= nranks;
  }

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, AllGather) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclAllGather(sendbuff, recvbuff, count / nranks, sdcclFloat, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, ReduceScatter) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclReduceScatter(sendbuff, recvbuff, count / nranks, sdcclFloat,
                      sdcclSum, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, Reduce) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclReduce(sendbuff, recvbuff, count, sdcclFloat, sdcclSum, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, Gather) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff  = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << static_cast<float *>(hostsendbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclGather(sendbuff, recvbuff, count / nranks, sdcclFloat, 0, comm,
               stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "recvbuff  = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << static_cast<float *>(hostrecvbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, Scatter) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  if (rank == 0) {
    for (size_t i = 0; i < count; i++) {
      ((float *)hostsendbuff)[i] = static_cast<float>(i);
    }

    devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                            sdcclMemcpyHostToDevice, stream);

    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclScatter(sendbuff, recvbuff, count / nranks, sdcclFloat, 0, comm,
                stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size / nranks,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLCollTest, Broadcast) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = i % 10;
  }
  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          sdcclMemcpyHostToDevice, stream);

  if (rank == 0) {
    std::cout << "sendbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostsendbuff)[i] << " ";
    }
    std::cout << ((float *)hostsendbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sdcclBroadcast(sendbuff, recvbuff, count, sdcclFloat, 0, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, stream);

  devHandle->streamSynchronize(stream);

  if (rank == 0) {
    std::cout << "recvbuff = ";
    for (size_t i = 0; i < 10; i++) {
      std::cout << ((float *)hostrecvbuff)[i] << " ";
    }
    std::cout << ((float *)hostrecvbuff)[10] << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(strcmp(static_cast<char *>(hostsendbuff),
                   static_cast<char *>(hostrecvbuff)),
            0);
}

TEST_F(SDCCLTopoTest, TopoDetection) {
  sdcclComm_t &comm = handler->comm;
  sdcclUniqueId_t &uniqueId = handler->uniqueId;

  std::cout << "executing sdcclCommInitRank" << std::endl;
  auto result = sdcclCommInitRank(&comm, nranks, uniqueId, rank);
  EXPECT_EQ(result, sdcclSuccess);
}

// ---------------------------------------------------------------------------
// Intra-node AllReduce: each rank fills with (rank+1), verify sum
// ---------------------------------------------------------------------------
TEST_F(SDCCLKernelTest, IntraAllReduce) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  // Allocate a separate buffer for the kernel (aligned with
  // test_kernel_intranode -R 0)
  void *regBuff = nullptr;
  devHandle->deviceMalloc(&regBuff, size, sdcclMemDevice, NULL);

  // Initialize: each rank fills with (rank + 1)
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)(rank + 1);
  }
  devHandle->deviceMemcpy(regBuff, hostsendbuff, size, sdcclMemcpyHostToDevice,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator with intra barriers
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  sdcclDevComm_t devComm = nullptr;
  ASSERT_EQ(sdcclDevCommCreate(comm, &reqs, &devComm), sdcclSuccess);

  // Create device memory handle (implicit IPC via sdcclDevMemCreate)
  sdcclDevMem_t devMem = nullptr;
  ASSERT_EQ(sdcclDevMemCreate(comm, regBuff, size, NULL, &devMem),
            sdcclSuccess);

  // Run AllReduce
  sdcclResult_t result =
      sdcclIntraAllReduce(devMem, count, sdcclFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Copy results back from regBuff
  devHandle->deviceMemcpy(hostrecvbuff, regBuff, size, sdcclMemcpyDeviceToHost,
                          NULL);

  // Verify: expected = nranks*(nranks+1)/2
  float expected = (float)(nranks * (nranks + 1)) / 2.0f;
  bool success = true;
  for (size_t i = 0; i < count && success; i++) {
    if (fabsf(((float *)hostrecvbuff)[i] - expected) > 1e-3f) {
      success = false;
      if (rank == 0) {
        std::cout << "IntraAllReduce MISMATCH at [" << i << "]: got "
                  << ((float *)hostrecvbuff)[i] << ", expected " << expected
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);

  // Cleanup
  sdcclDevMemDestroy(comm, devMem);
  sdcclDevCommDestroy(comm, devComm);
  devHandle->deviceFree(regBuff, sdcclMemDevice, NULL);
}

// ---------------------------------------------------------------------------
// Inter-node AlltoAll: two-sided send/recv via FIFO
// ---------------------------------------------------------------------------
TEST_F(SDCCLKernelTest, InterTwoSidedAlltoAll) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  // count per peer
  size_t countPerPeer = count / nranks;

  // Initialize sendbuff: all elements = rank (my rank)
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)rank;
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          sdcclMemcpyHostToDevice, NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator
  // Request inter barriers — needed by sdcclInterBarrierSession in the kernel
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  sdcclDevComm_t devComm = nullptr;
  ASSERT_EQ(sdcclDevCommCreate(comm, &reqs, &devComm), sdcclSuccess);

  // Create raw device memory handles for send/recv buffers
  sdcclDevMem_t sendMem = nullptr, recvMem = nullptr;
  ASSERT_EQ(sdcclDevMemCreate(comm, sendbuff, size, NULL, &sendMem),
            sdcclSuccess);
  ASSERT_EQ(sdcclDevMemCreate(comm, recvbuff, size, NULL, &recvMem),
            sdcclSuccess);

  // Launch AlltoAll kernel
  sdcclResult_t result = sdcclInterTwoSidedAlltoAll(
      sendMem, recvMem, countPerPeer, sdcclFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Destroy raw device memory handles
  sdcclDevMemDestroy(comm, sendMem);
  sdcclDevMemDestroy(comm, recvMem);

  // Destroy device communicator
  sdcclDevCommDestroy(comm, devComm);

  // Copy results back
  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          sdcclMemcpyDeviceToHost, NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: recvbuff[p*countPerPeer] should equal p for all p
  bool success = true;
  for (int p = 0; p < nranks; p++) {
    float expected = (float)p;
    float actual = ((float *)hostrecvbuff)[p * countPerPeer];
    if (actual != expected) {
      success = false;
      if (rank == 0) {
        std::cout << "InterTwoSidedAlltoAll mismatch at peer " << p
                  << ": expected " << expected << ", got " << actual
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);
}

// ---------------------------------------------------------------------------
// Inter-node one-sided AlltoAll: put + waitSignal + flush
// ---------------------------------------------------------------------------
TEST_F(SDCCLKernelTest, InterOneSidedAlltoAll) {
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  size_t countPerPeer = count / nranks;

  // One-sided needs VMM memory + RDMA registration.
  // Allocate separate buffers (fixture's sendbuff/recvbuff stay untouched).
  void *osSend = nullptr, *osRecv = nullptr;
  ASSERT_EQ(sdcclMemAlloc(&osSend, size), sdcclSuccess);
  ASSERT_EQ(sdcclMemAlloc(&osRecv, size), sdcclSuccess);

  void *sendRegHandle = nullptr, *recvRegHandle = nullptr;
  ASSERT_EQ(sdcclCommRegister(comm, osSend, size, &sendRegHandle),
            sdcclSuccess);
  ASSERT_EQ(sdcclCommRegister(comm, osRecv, size, &recvRegHandle),
            sdcclSuccess);

  // Initialize sendbuff: all elements = rank
  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = (float)rank;
  }
  devHandle->deviceMemcpy(osSend, hostsendbuff, size, sdcclMemcpyHostToDevice,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create device communicator with inter-node barrier + signal
  sdcclDevCommRequirements reqs = SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interBarrierCount = SDCCL_DEVICE_CTA_COUNT;
  reqs.interSignalCount = 1;
  sdcclDevComm_t devComm = nullptr;
  ASSERT_EQ(sdcclDevCommCreate(comm, &reqs, &devComm), sdcclSuccess);

  // Create device memory handles
  sdcclDevMem_t sendMem = nullptr, recvMem = nullptr;
  ASSERT_EQ(sdcclDevMemCreate(comm, osSend, size, NULL, &sendMem),
            sdcclSuccess);
  ASSERT_EQ(sdcclDevMemCreate(comm, osRecv, size, NULL, &recvMem),
            sdcclSuccess);

  // Launch one-sided AlltoAll
  sdcclResult_t result = sdcclInterOneSidedAlltoAll(
      sendMem, recvMem, countPerPeer, sdcclFloat, devComm, stream);
  devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Destroy device memory handles
  sdcclDevMemDestroy(comm, sendMem);
  sdcclDevMemDestroy(comm, recvMem);

  // Destroy device communicator
  sdcclDevCommDestroy(comm, devComm);

  // Copy results back from osRecv
  devHandle->deviceMemcpy(hostrecvbuff, osRecv, size, sdcclMemcpyDeviceToHost,
                          NULL);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: recvbuff[p*countPerPeer] should equal p for all p
  bool success = true;
  for (int p = 0; p < nranks; p++) {
    float expected = (float)p;
    float actual = ((float *)hostrecvbuff)[p * countPerPeer];
    if (actual != expected) {
      success = false;
      if (rank == 0) {
        std::cout << "InterOneSidedAlltoAll mismatch at peer " << p
                  << ": expected " << expected << ", got " << actual
                  << std::endl;
      }
    }
  }
  EXPECT_TRUE(success);

  // Cleanup one-sided buffers
  sdcclCommDeregister(comm, sendRegHandle);
  sdcclCommDeregister(comm, recvRegHandle);
  sdcclMemFree(osSend);
  sdcclMemFree(osRecv);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
