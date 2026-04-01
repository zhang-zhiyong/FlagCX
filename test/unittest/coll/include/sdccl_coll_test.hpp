#pragma once

#include "sdccl.h"
#include "sdccl_test.hpp"

class SDCCLCollTest : public SDCCLTest {
protected:
  SDCCLCollTest() {}

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
