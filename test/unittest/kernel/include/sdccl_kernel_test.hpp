#pragma once

#include "sdccl.h"
#include "sdccl_kernel.h"
#include "sdccl_test.hpp"

class SDCCLKernelTest : public SDCCLTest {
protected:
  SDCCLKernelTest() {}

  void SetUp();

  void TearDown();

  void Run();

  sdcclHandlerGroup_t handler;
  sdcclStream_t stream;
  void *sendbuff;
  void *recvbuff;
  void *hostsendbuff;
  void *hostrecvbuff;
  size_t size;
  size_t count;
};
