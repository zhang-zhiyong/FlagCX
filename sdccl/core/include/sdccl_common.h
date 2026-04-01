/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_COMMON_H_
#define SDCCL_COMMON_H_

#include "debug.h"

#define SDCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum {
  sdcclFuncBroadcast = 0,
  sdcclFuncReduce = 1,
  sdcclFuncAllGather = 2,
  sdcclFuncReduceScatter = 3,
  sdcclFuncAllReduce = 4,
  sdcclFuncSendRecv = 5,
  sdcclFuncSend = 6,
  sdcclFuncRecv = 7,
  sdcclNumFuncs = 8
} sdcclFunc_t;

#define SDCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define SDCCL_ALGO_UNDEF -1
#define SDCCL_ALGO_TREE 0
#define SDCCL_ALGO_RING 1
#define SDCCL_ALGO_COLLNET_DIRECT 2
#define SDCCL_ALGO_COLLNET_CHAIN 3
#define SDCCL_ALGO_NVLS 4
#define SDCCL_ALGO_NVLS_TREE 5

#define SDCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define SDCCL_PROTO_UNDEF -1
#define SDCCL_PROTO_LL 0
#define SDCCL_PROTO_LL128 1
#define SDCCL_PROTO_SIMPLE 2

#define SDCCL_DEVICE_PCI_BUSID_BUFFER_SIZE 16

#endif
