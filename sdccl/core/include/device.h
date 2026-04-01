/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_DEVICE_H_
#define SDCCL_DEVICE_H_

#include "align.h"
#include "sdccl_common.h"
#include "net_device.h"
#include "type.h"
#include <stdint.h>

extern const char *sdcclFuncStr[SDCCL_NUM_FUNCTIONS];

extern const char *sdcclAlgoStr[SDCCL_NUM_ALGORITHMS];

extern const char *sdcclProtoStr[SDCCL_NUM_PROTOCOLS];

#define SDCCL_MAX_OPS 2048
#define SDCCL_STEPS 8

enum sdcclDevRedOp_t {
  sdcclDevSum,
  sdcclDevProd,
  sdcclDevMinMax,
  sdcclDevPreMulSum,
  sdcclDevSumPostDiv,
  sdcclNumDevRedOps
};
struct sdcclDevRedOpFull {
  sdcclDevRedOp_t op;
  sdcclRedOp_t proxyOp;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

union sdcclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
};

#define WARP_SIZE 32
#define MAXCHANNELS 32
#define SDCCL_MAX_NTHREADS 640
#define SDCCL_SIMPLE_MAX_NTHREADS 512
#define SDCCL_LL_MAX_NTHREADS 512
#define SDCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define SDCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define SDCCL_LL_FLAG_MAX 0x100
#define SDCCL_LL_FLAG(a) ((uint32_t)((a) % SDCCL_LL_FLAG_MAX))
#else
#define SDCCL_LL_CLEAN_MASK 0x7ffffff8
#define SDCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least SDCCL_NSTEPS
static_assert(SDCCL_LL_CLEAN_MASK % SDCCL_STEPS == 0,
              "Invalid SDCCL_LL_CLEAN_MASK value");

#define SDCCL_LL128_LINESIZE 128
#define SDCCL_LL128_LINEELEMS (SDCCL_LL128_LINESIZE / sizeof(uint64_t))
#define SDCCL_LL128_DATAELEMS (SDCCL_LL128_LINEELEMS - 1)

#define SDCCL_LL128_MAX_NTHREADS 640
#define SDCCL_LL128_ELEMS_PER_THREAD 120

#define SDCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define SDCCL_LL128_SHMEM_SIZE                                                \
  (SDCCL_LL128_SHMEM_ELEMS_PER_THREAD * SDCCL_LL128_MAX_NTHREADS)

#define SDCCL_DIRECT_WRITE 0x01
#define SDCCL_DIRECT_READ 0x02
#define SDCCL_DIRECT_NIC 0x04
#define SDCCL_IPC_WRITE 0x08
#define SDCCL_IPC_READ 0x10
#define SDCCL_NVLS_MIN_POLL 0x20

#define SDCCL_MAX_COLLNET_SIZE (1L << 29)

enum sdcclRegBufferType {
  SDCCL_REGULAR_BUFFER = 0,
  SDCCL_IPC_REG_BUFFER = 1,
  SDCCL_NVLS_REG_BUFFER = 2,
  SDCCL_COLLNET_REG_BUFFER = 3
};

struct sdcclConnInfo {
  // Regular comm mechanism
  char *buffs[SDCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void *mhandles[SDCCL_NUM_PROTOCOLS];
  uint64_t *tail; // Local for recv, remote for send
  uint64_t *head; // Local for send, remote for recv

  int flags;                  // Direct communication / other flags
  int shared;                 // Buffers are shared
  int stepSize;               // Step size for the SIMPLE buffer
  void **ptrExchange;         // Pointer exchange for direct communication
  uint64_t *redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct sdcclConnFifo *connFifo; // Used for GPU - Proxy communication

  uint64_t step; // Keep where we are
  uint64_t llLastCleaning;
  sdcclNetDeviceHandle_t netDeviceHandle;
};

struct sdcclProxyConnector {
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct sdcclProxyConnection *connection;
  sdcclResult_t (*proxyProgress)(
      struct sdcclProxyState *proxyState,
      struct sdcclProxyArgs *); // Copied from transport if necessary
};

struct sdcclConnector {
  int connected;
  int registered;
  struct sdcclProxyConnector proxyConn;
  struct sdcclTransportComm *transportComm;
  void *transportResources;
  struct sdcclConnInfo conn;
};

struct sdcclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal sdccl index to user-specified rank order. This is
  // necessary since we need to know how the user expects data to be ordered
  // across devices. Ordered from current device.
  int *userRanks;

  int index; // This rank's index in the ring
};

// The root of each tree only has one node down (+1 intra-node).
#define SDCCL_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define SDCCL_MAX_TREE_ARITY 3
struct sdcclTree {
  int depth;
  int up;
  int down[SDCCL_MAX_TREE_ARITY];
};

#define SDCCL_MAX_DIRECT_ARITY 7
struct sdcclDirect {
  int depth;
  int out;
  int nHeads; // Number of parallel N<->1<->net operations we'll do in parallel;
              // size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a
                // head rank (no local NIC)
  int shift; // Shuffling of send/recv for scatter/gather operations, basically
             // localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[SDCCL_MAX_DIRECT_ARITY + 1];
  int up[SDCCL_MAX_DIRECT_ARITY];
  int down[SDCCL_MAX_DIRECT_ARITY];
};

#define SDCCL_MAX_NVLS_ARITY 32
#define SDCCL_MAX_NVLS_TREE_ARITY 3
struct sdcclNvls {
  int out;
  int nHeads; // Number of parallel N<->1<->net operations we'll do in parallel;
              // size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a
                // head rank (no local NIC)
  int up[SDCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[SDCCL_MAX_NVLS_TREE_ARITY];
  int node;
  int nNodes;
};

#if __CUDA_ARCH__ >= 900
#define SDCCL_MAX_ARITY SDCCL_MAX_NVLS_ARITY
#else
#define SDCCL_MAX_ARITY SDCCL_MAX_DIRECT_ARITY
#endif

#define SDCCL_MAX_CONNS 4
struct sdcclChannelPeer {
  struct sdcclConnector send[SDCCL_MAX_CONNS];
  struct sdcclConnector recv[SDCCL_MAX_CONNS];
  int refCount;
};

struct sdcclKernelComm;

/* sdcclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of sdcclWorkElem. */
#define SDCCL_WORK_SIZE 512

enum sdcclWorkType : uint8_t {
  sdcclWorkTypeUnused = 0,
  sdcclWorkTypeColl = 1,
  sdcclWorkTypeP2p = 2,
  sdcclWorkTypeRegColl = 3
};
enum sdcclWorkP2PType : uint8_t {
  sdcclWorkP2pTypeUnused = 0,
  sdcclWorkP2pTypeSend,
  sdcclWorkP2pTypeRecv
};

struct sdcclWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send
                       // back.
  };
  uint16_t funcIndex;
  uint8_t isLast : 1; // last work for this kernel
  uint8_t inFifo : 1; // is this work in the fifo
  enum sdcclWorkType type;
};

struct sdcclWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed : 1, redOpArgIsPtr : 1, oneNode : 1;
    };
  };
  uint8_t regUsed;
  uint8_t nWarps;
  uint8_t direct;
  uint32_t root;
  const void *sendbuff;
  void *recvbuff;

  size_t count;
  uint64_t redOpArg;
  uint64_t chunkCount : 25, workCount : 39;
  union {
    struct {
      uint64_t lastChunkCount : 25;
      uint64_t workOffset : 39;
    };
    struct {
      uint64_t bid : 32;
      uint64_t nChannels : 32;
    };
  };
};

#define SDCCL_MAX_WORK_ELEMENTS                                               \
  ((SDCCL_WORK_SIZE -                                                         \
    alignUp(sizeof(sdcclWorkHeader), alignof(sdcclWorkElem))) /              \
   sizeof(sdcclWorkElem))
static_assert(SDCCL_MAX_WORK_ELEMENTS == 9,
              "Sanity check: SDCCL_MAX_WORK_ELEMENTS == 9");

struct sdcclWorkElemP2p {
  int peer : 30;
  int proto : 2;

  enum sdcclWorkP2PType p2pType;
  uint8_t reg : 1;
  uint8_t nWarps : 5;
  uint8_t warpStart;
  uint8_t ngroups;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(sdcclWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  // void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  // size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((SDCCL_WORK_SIZE -
                alignUp(sizeof(sdcclWorkHeader), alignof(sdcclWorkElemP2p))) /
               sizeof(sdcclWorkElemP2p)) >= 16,
              "Sanity check: SDCCL_MAX_WORK_ELEMENTS_P2P == 16");
#define SDCCL_MAX_WORK_ELEMENTS_P2P 16

struct sdcclWorkElemReg {
  struct sdcclWorkElem elem;
  void *dnInputs[SDCCL_MAX_DIRECT_ARITY + 1];
  void *dnOutputs[SDCCL_MAX_DIRECT_ARITY + 1];
  void *upOutputs[SDCCL_MAX_DIRECT_ARITY + 1];
};

#define SDCCL_MAX_WORK_ELEMENTS_REG                                           \
  ((SDCCL_WORK_SIZE -                                                         \
    alignUp(sizeof(sdcclWorkHeader), alignof(sdcclWorkElemReg))) /           \
   sizeof(sdcclWorkElemReg))
static_assert(SDCCL_MAX_WORK_ELEMENTS_REG == 2,
              "Sanity check: SDCCL_MAX_WORK_ELEMENTS_REG == 2");

// Number of named barriers supported by CUDA
#define SDCCL_MAX_GROUPS 16

struct sdcclWork {
  struct sdcclWorkHeader header;
  union {
    char pad[SDCCL_WORK_SIZE - sizeof(struct sdcclWorkHeader)];
    struct sdcclWorkElem elems[SDCCL_MAX_WORK_ELEMENTS];
    struct sdcclWorkElemP2p p2pElems[SDCCL_MAX_WORK_ELEMENTS_P2P];
    struct sdcclWorkElemReg regElems[SDCCL_MAX_WORK_ELEMENTS_REG];
  };
};
static_assert(sizeof(struct sdcclWork) == SDCCL_WORK_SIZE,
              "Sanity check: sizeof(struct sdcclWork) == SDCCL_WORK_SIZE");
static_assert(sizeof(struct sdcclWork) % 16 == 0,
              "Sanity check: sizeof(struct sdcclWork)%16 == 0");

struct sdcclDevChannelPeer {
  // Stripped version of sdcclChannelPeer where we only keep the sdcclConnInfo
  // instead of the full sdcclConnector.
  struct sdcclConnInfo send[SDCCL_MAX_CONNS];
  struct sdcclConnInfo recv[SDCCL_MAX_CONNS];
};

struct alignas(16) sdcclDevChannel {
  struct sdcclDevChannelPeer **peers;
  struct sdcclRing ring;
  struct sdcclTree tree;
  struct sdcclTree collnetChain;
  struct sdcclDirect collnetDirect;
  struct sdcclNvls nvls;
  uint32_t *workFifoDone; // Location of done counter, device writes index+1 of
                          // last work processed
};

struct sdcclKernelComm {
  int rank;
  int nRanks;
  int node;
  int nNodes;
  int buffSizes[SDCCL_NUM_PROTOCOLS];
  int p2pChunkSize;

  // Operation list for aggregation
  int workFifoDepth;
  struct sdcclWork *workFifoHeap; // may be cudaHost or GDR memory

  int *collNetDenseToUserRank;

  // Flag to ask SDCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct sdcclDevChannel *channels /*[MAXCHANNELS]*/;
};

struct alignas(16) sdcclKernelCommAndChannels {
  struct sdcclKernelComm comm;
  struct sdcclDevChannel channels[MAXCHANNELS];
};

#ifdef __CUDA_ARCH__
#define SDCCL_CUDA_ARCH __CUDA_ARCH__
#else
#define SDCCL_CUDA_ARCH 0
#endif

#endif
