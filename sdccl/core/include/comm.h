/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_COMM_H_
#define SDCCL_COMM_H_

#include "bootstrap.h"
#include "device.h"
#include "sdccl_kernel.h"
#include "sdccl_net.h"
#include "sdccl_tuner.h"
#include "info.h"
#include "register.h"
#include "type.h"
#include <stdint.h>

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define SDCCL_LL_THREAD_THRESHOLD 8
#define SDCCL_LL128_THREAD_THRESHOLD 8
#define SDCCL_SIMPLE_THREAD_THRESHOLD 64

struct sdcclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      void *ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE - sizeof(void *) - 2 * sizeof(uint64_t)];
      int offsFifo[SDCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct sdcclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      struct sdcclConnFifo connFifo[SDCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState { ThreadStart, ThreadStop };

#define SDCCL_IPC_POOL_SIZE (2 * SDCCL_MAX_LOCAL_RANKS * SDCCL_MAX_OPS)

struct sdcclGraphHelperResources {
  sdcclHeteroComm *comm;
  pthread_mutex_t threadLock;
  pthread_cond_t threadCond;
  enum helperThreadState threadState;
  void *ipcBases[SDCCL_IPC_POOL_SIZE];
  int ipcTail;
  int ipcHead;
};

struct sdcclUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  sdcclDataType_t datatype;
  sdcclDevRedOpFull opFull;
};

struct sdcclNodeRanks {
  int localRanks;
  int *localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int *ranks;
};

struct sdcclDestructor {
  struct sdcclDestructor *next;
  void *obj;
  sdcclResult_t (*fn)(struct sdcclDestructor *me);
};

struct sdcclCommCallback {
  struct sdcclCommCallback *next;
  sdcclResult_t (*fn)(struct sdcclHeteroComm *comm,
                       struct sdcclCommCallback *cb);
};

struct sdcclSharedResources {
  int refCount;
  struct sdcclHeteroComm *owner; /* comm which creates this shared res. */
  struct sdcclChannelPeer *peers[MAXCHANNELS];
  struct sdcclDevChannelPeer *devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int *tpRankToLocalRank;

  /* proxy related shared res */
  struct sdcclProxyState *proxyState;
};

struct sdcclChannel {
  struct sdcclChannelPeer **peers;
  struct sdcclDevChannelPeer **devPeers;
  /* devPeer pointer array used for host side access */
  struct sdcclDevChannelPeer **devPeersHostPtr;
  struct sdcclRing ring;
  int *devRingUserRanks;
  struct sdcclTree tree;

  struct sdcclTree collnetChain;
  struct sdcclDirect collnetDirect;

  struct sdcclNvls nvls;

  int id;                // index of this channel
  uint32_t workFifoSent; // last used work index+1

  /* comm split sharable resources */
  struct sdcclChannelPeer *collnetPeers;
  struct sdcclDevChannelPeer *collnetDevPeers;
  struct sdcclChannelPeer *nvlsPeers;
  struct sdcclDevChannelPeer *nvlsDevPeers;
};

struct sdcclWorkList {
  struct sdcclWorkList *next;
  struct sdcclWork work;
};

struct sdcclPointerList {
  struct sdcclPointerList *next;
  void *ptr;
};

struct sdcclCollnetHandleList {
  struct sdcclCollnetHandleList *next;
  void *collnetHandle;
  size_t size;
  const void *buffer;
  struct sdcclProxyConnector *proxyconn;
};

#define SDCCL_MAGIC 0x0280028002800280 // Nickel atomic number is 28.

struct sdcclHeteroComm {
  uint64_t startMagic;
  struct sdcclMemoryStack memPermanent, memScoped;
  // List of destructors to run when comm is destructed
  struct sdcclDestructor *destructorHead;

  struct sdcclSharedResources *sharedRes;
  /* map to top parent ranks. */
  int *topParentRanks;
  int *topParentLocalRanks;
  struct sdcclChannel channels[MAXCHANNELS];
  struct sdcclPeerInfo *peerInfo;
  struct sdcclTopoServer *topoServer;
  struct sdcclInterServerTopo *interServerTopo;

  struct sdcclNetAdaptor *netAdaptor;
  struct bootstrapState *bootstrap;
  // Bitmasks for sdcclTransportP2pSetup
  uint64_t *connectSend;
  uint64_t *connectRecv;

  uint64_t magic; // Magic number for all network communication. Not a security
                  // key -- only goal is to detect mismatches.

  uint64_t commHash;
  int rank;                   // my rank in the communicator
  int nRanks;                 // number of GPUs in communicator
  int cudaDev;                // my cuda device index
  int netDev;                 // my net  device index
  int nvmlDev;                // my nvml device index
  int compCap;                // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;              // my PCI bus ID in int format
  cpu_set_t cpuAffinity;      // CPU affinity of the GPU
  int cudaArch;               // matches __CUDA_ARCH__ of device

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int *rankToNode;
  int *rankToLocalRank;
  int *localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct sdcclNodeRanks *nodeRanks;
  // MNNVL: Multi-Node NVLink
  int MNNVL;                // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank;           // Our rank within the MNNVL clique

  bool checkPointers;
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;

  // Channels for collectives
  int nChannels;    // connection nChannels
  int collChannels; // enqueue nChannels
  int nvlsChannels; // enqueue nChannels
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pChannels[MAXCHANNELS];

  // P2P schedule for pairing send/recv operations
  struct P2pSchedulePair {
    int sendRank;
    int recvRank;
  } * p2pSchedule;

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[SDCCL_NUM_PROTOCOLS];
  int p2pChunkSize;
  int nvlsChunkSize;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[SDCCL_NUM_ALGORITHMS][SDCCL_NUM_PROTOCOLS];
  float latencies[SDCCL_NUM_FUNCTIONS][SDCCL_NUM_ALGORITHMS]
                 [SDCCL_NUM_PROTOCOLS];
  float bandwidths[SDCCL_NUM_FUNCTIONS][SDCCL_NUM_ALGORITHMS]
                  [SDCCL_NUM_PROTOCOLS];
  float ringbdw[SDCCL_NUM_FUNCTIONS][SDCCL_NUM_PROTOCOLS];
  int maxThreads[SDCCL_NUM_ALGORITHMS][SDCCL_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous SDCCL operations. */
  sdcclResult_t asyncResult;

  // Flag to ask SDCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile uint32_t *childAbortFlag;
  uint32_t *abortFlagRefCount;

  // Device side of the communicator (for cudaFree's)
  struct sdcclKernelComm
      *devComm; // actually = &sdcclKernelCommAndChannels::comm

  // Operation pool.
  int workFifoDepth; // size of workFifoHeap[], power of 2
  struct sdcclWork *workFifoHeap;
  struct sdcclWork *devWorkFifoHeap;
  void *workFifoHeapGdrHandle;

  // Work completion notificaion
  uint32_t *workFifoDone /*[MAXCHANNELS]*/; // in cudaHost memory
  uint32_t
      workFifoSent; // Monotonic (mod 1<<32) index of next unused fifo slot.
  uint32_t workFifoAckdMin; // Monotonic index of least unprocessed fifo slot
                            // over all channels.

  // Intra-process sync
  struct sdcclHeteroComm
      *intraComm0; // leader of intra-process comms (self possible)
  struct sdcclHeteroComm
      *intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct sdcclProxyState *proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  int collNetSupport;
  bool collNetRegSupport;
  uint8_t collNetSupportMatrix[4 /*sum,prod,min,max*/][sdcclNumTypes];
  int intraHighestTransportType;
  int *collNetHeads;
  int collNetHeadsNum;
  int *collNetDenseToUserRank;
  int *collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  struct sdcclCollNetSharedRes *collNetSharedRes;

  // NVLink SHARP (NVLS) support
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct sdcclNvlsSharedRes *nvlsResources;

  // pools backed by comm->memPermanent
  struct sdcclMemoryPool memPool_sdcclProxyOp;
  struct sdcclMemoryPool memPool_sdcclKernelPlan;
  struct sdcclMemoryPool memPool_sdcclPointerList;
  struct sdcclMemoryPool memPool_sdcclNvlsHandleList;
  struct sdcclMemoryPool memPool_sdcclCollnetHandleList;
  // Next comm in this thread's active sdcclGroup[Start|End](). Holds "0x1"
  // when this comm is not yet in a group.
  struct sdcclHeteroComm *groupNext;
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct sdcclHeteroComm *preconnectNext;
  int persistentRefs; // number of persistent plan-lists capturing this comm
  struct sdcclTasks tasks;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  sdcclUserRedOp *userRedOps;

  // Queue of things for the main thread to do
  struct sdcclIntruQueueMpsc<struct sdcclCommCallback,
                              &sdcclCommCallback::next>
      callbackQueue;

  sdcclConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  sdcclResult_t initState;
  // flag to indicate if sdcclCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct sdcclGroupJob *groupJob;

  // Tuning plugin
  sdcclTuner_t *tuner;
  void *tunerContext;
  // buffer registration cache
  struct sdcclRegCache regCache;
  uint64_t groupHash;
  uint64_t endMagic;
  // Kernel FIFO buffer for device side communication
  void *fifoBuffer;
  // uniRunner FIFO buffer
  void *uniRunnerFifoBuffer;
  // Device communicator (set by sdcclDevCommCreate).
  // Used by proxy for BarrierSignal, WaitSignal, PutValue handlers.
  sdcclDevComm_t devCommHandle;
};

typedef struct sdcclHeteroComm *sdcclHeteroComm_t;

enum sdcclLaunchMode {
  sdcclLaunchModeInvalid = 0,
  sdcclLaunchModeParallel,
  sdcclLaunchModeGroup
};
extern enum sdcclLaunchMode sdcclParamLaunchMode;

void sdcclCommPushFree(struct sdcclHeteroComm *comm, void *buf);
void sdcclCommPushCudaFree(struct sdcclHeteroComm *comm, void *buf);
void sdcclCommPushCudaHostFree(struct sdcclHeteroComm *comm, void *buf);
void sdcclCommPushCudaGdrFree(struct sdcclHeteroComm *comm, void *handle);

inline sdcclResult_t sdcclCommPollCallbacks(struct sdcclHeteroComm *comm,
                                              bool waitSome) {
  sdcclResult_t result = sdcclSuccess;
  struct sdcclCommCallback *cb =
      sdcclIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct sdcclCommCallback *next = cb->next;
    sdcclResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != sdcclSuccess)
      result = res1;
    cb = next;
  }
  SDCCLCHECK(result);
  return sdcclSuccess;
}

inline void sdcclCommIntraBarrierIn(struct sdcclHeteroComm *comm,
                                     uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x) << 32) | (phase ^ 1);
  } else {
    struct sdcclHeteroComm *comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(
        &comm0->intraBarrierCounter, (uint64_t(x) << 32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate,
                       (count >> 32 << 32) | (phase ^ 1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to sdcclCommIntraBarrierIn(comm, x)
inline uint32_t sdcclCommIntraBarrierOut(struct sdcclHeteroComm *comm) {
  struct sdcclHeteroComm *comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano() - t0 >= 5 * 1000)
        sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1)
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate >> 32;
}

// Scrambles the bits of non-builtin values of sdcclRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline sdcclRedOp_t sdcclUserRedOpMangle(sdcclHeteroComm *comm,
                                                  sdcclRedOp_t op) {
  // Preserve the built-in values.
  if (int(op) < int(sdcclNumRedOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(sdcclMaxRedOp); // sdcclMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their
  // preimage.
  return op1 < int(sdcclNumRedOps) ? op : sdcclRedOp_t(op1);
}

sdcclResult_t sdcclCommEnsureReady(sdcclHeteroComm_t comm);
sdcclResult_t sdcclCommSetAsyncError(sdcclHeteroComm_t comm,
                                       sdcclResult_t nextState);

#endif
