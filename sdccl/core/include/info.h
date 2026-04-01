/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_INFO_H_
#define SDCCL_INFO_H_

#include "device.h"
#include "utils.h"

typedef struct sdcclHeteroComm *sdcclHeteroComm_t;

#define SDCCL_MAX_LOCAL_RANKS 64

typedef enum : uint8_t {
  sdcclPatternRing,
  sdcclPatternRingTwice,
  sdcclPatternPipelineFrom,
  sdcclPatternPipelineTo,
  sdcclPatternTreeUp,
  sdcclPatternTreeDown,
  sdcclPatternTreeUpDown,
  sdcclPatternCollnetChain,
  sdcclPatternCollnetDirect,
  sdcclPatternNvls,
  sdcclPatternNvlsTree,
  sdcclPatternSend,
  sdcclPatternRecv
} sdcclPattern_t;

// Used to pass SDCCL call information between functions
struct sdcclInfo {
  sdcclFunc_t coll;
  const char *opName;
  // SDCCL Coll Args
  const void *sendbuff;
  void *recvbuff;
  size_t count;
  sdcclDataType_t datatype;
  sdcclRedOp_t op;
  int root; // peer for p2p operations
  sdcclHeteroComm_t comm;
  sdcclStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  sdcclDevRedOpFull opFull;
  sdcclPattern_t pattern;
  size_t nBytes;
  size_t aggnBytes;
  size_t workBytes;
  size_t sendbuffSize;
  size_t recvbuffSize;
  int stepSize;
  int chunkCount;
  int chunkSize;
  int channelId;
  int workFuncIndex;
  sdcclRegBufferType regBufType;
  void *regBufSend[SDCCL_MAX_LOCAL_RANKS];
  void *regBufRecv[SDCCL_MAX_LOCAL_RANKS];
  // collnet buffer reg handles
  void *sendMhandle;
  void *recvMhandle;
  // Need to initialize
  int nThreads;
  int nChannels;
  int algorithm;
  int protocol;
  bool userTuned;
  struct sdcclInfo *next;
  unsigned long long sdcclFuncTimes;
  uint64_t groupHash;
};

inline sdcclResult_t sdcclInfoSetDerived(struct sdcclInfo *info,
                                           int nRanks) {
  info->nBytes = info->workBytes =
      info->count * getSdcclDataTypeSize(info->datatype);
  if (info->coll == sdcclFuncAllGather || info->coll == sdcclFuncBroadcast) {
    info->count = info->workBytes;
    info->datatype = sdcclInt8;
  }
  if (info->coll == sdcclFuncAllGather ||
      info->coll == sdcclFuncReduceScatter)
    info->nBytes *= nRanks; // count is per rank

  /* compute buffer size for NVLS buffer registration */
  if (info->coll == sdcclFuncAllGather) {
    info->sendbuffSize = info->workBytes;
    info->recvbuffSize = info->sendbuffSize * nRanks;
  } else if (info->coll == sdcclFuncReduceScatter) {
    info->recvbuffSize = info->workBytes;
    info->sendbuffSize = info->recvbuffSize * nRanks;
  } else {
    info->sendbuffSize = info->recvbuffSize = info->workBytes;
  }
  return sdcclSuccess;
}

struct sdcclTaskColl {
  struct sdcclTaskColl *next;
  sdcclFunc_t func;
  void const *sendbuff;
  void *recvbuff;
  size_t count;
  int root;
  sdcclDataType_t datatype;
  sdcclDevRedOpFull op;
  int chunkSteps, sliceSteps;
  struct sdcclInfo info;
};
struct sdcclTaskP2p {
  sdcclTaskP2p *next;
  void *buff;
  size_t bytes;
  // Stateful chunk index. If a p2p gets "cut" over two plans this keeps track
  // of where it left off.
  int chunk;
  int opId;
  int step;
  sdcclDataType_t dtype;
  sdcclStream_t stream;
};

struct sdcclCudaStreamList {
  struct sdcclCudaStreamList *next;
  sdcclStream_t stream;
};
struct sdcclTasks {
  struct Peer {
    bool sendSeen, recvSeen;
    struct sdcclIntruQueue<struct sdcclTaskP2p, &sdcclTaskP2p::next>
        sendQueue;
    struct sdcclIntruQueue<struct sdcclTaskP2p, &sdcclTaskP2p::next>
        recvQueue;
  };
  struct sdcclIntruQueue<struct sdcclInfo, &sdcclInfo::next> collQueue;
  // Queue for user-tuned executed collectives
  struct sdcclIntruQueue<struct sdcclInfo, &sdcclInfo::next> collTunedQueue;
  // Queue for continuous bytes distribution (CBD) collectives
  struct sdcclIntruQueue<struct sdcclInfo, &sdcclInfo::next> collCBDQueue;
  // Queue for collnet
  struct sdcclIntruQueue<struct sdcclInfo, &sdcclInfo::next> collnetQueue;
  size_t workBytesTotal;
  int usableChannels;
  bool sorted;
  struct Peer *peers /*[nRanks]*/;
  int *p2pOrder;
  int p2pOrderSteps;
  int nTasksColl, nTasksP2p;

  // The list of user streams aggregated over all tasks present.
  struct sdcclCudaStreamList *streams;
  // The most recent user stream. Ignored if streams==nullptr
  sdcclStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict
  // the user that all streams must be captured in the same graph or not
  // captured at all. Technically we could probably relax this, but that would
  // mean collecting a different `sdcclTasks` per graph and one for non-graph.
};

#endif
