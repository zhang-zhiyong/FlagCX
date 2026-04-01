/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_PROXY_H_
#define SDCCL_PROXY_H_

#include "device.h"
#include "sdccl_kernel.h"
#include "sdccl_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include "uni_runner_impl.h"
#include <memory>
#include <pthread.h>

enum sdcclProxyOpState {
  sdcclProxyOpNone,
  sdcclProxyOpReady,
  sdcclProxyOpProgress
};

struct sdcclProxyKernelState {
  pthread_t thread;
  sdcclFifo_t fifo;
  sdcclStream_t stream;
  int stop = 0;
  // Synchronization for initialization
  pthread_mutex_t initMutex;
  pthread_cond_t initCond;
  int ready = 0;
};

struct sdcclProxyArgs;
typedef sdcclResult_t (*proxyProgressFunc_t)(struct sdcclProxyState *,
                                              struct sdcclProxyArgs *);

#define SDCCL_PROXY_MAX_SUBS MAXCHANNELS
static_assert(SDCCL_MAX_WORK_ELEMENTS <= MAXCHANNELS,
              "Not enough sub space for max work elements");

union sdcclProxyOpSpecifics {
  struct {
    size_t sizePerRank;
    int nNodes, node;
  } collnetDirect;
};

struct sdcclProxySubArgs {
  struct sdcclProxyConnection *connection;
  int reg;
  // p2p mhandle
  void *mhandle;
  int stepSize;
  void *stepBuff;
  void *stream;
  // kernel copy
  void *copyArgs;

  // collnet handles
  void *sendMhandle;
  void *recvMhandle;
  uint8_t *sendbuff;
  uint8_t *recvbuff;
  size_t offset;
  int channelId;
  int nsteps;
  ssize_t nbytes;
  int peer;

  int groupSize; // Number of consecutive sub operations sharing the same
                 // recvComm
  uint64_t base;
  uint64_t posted;
  uint64_t received;
  uint64_t flushed;
  uint64_t transmitted;
  uint64_t done;
  uint64_t end;
  void *requests[SDCCL_STEPS];
  void *profilingEvents[SDCCL_STEPS];
  void *recvRequestsCache[SDCCL_STEPS];
  int recvRequestsSubCount;
};

struct sdcclProxyArgs {
  struct sdcclProxySubArgs subs[SDCCL_NET_MAX_STEPS];
  proxyProgressFunc_t progress;
  int nsubs;
  int done;
  uint64_t opCount;
  int sliceSteps;
  int chunkSteps;
  size_t chunkSize;
  size_t stepSize;
  void *stepBuff;
  int waitCopy = 0;
  int posted = 0;
  int copied = 0;
  int postFlush = 0;
  int transmitted = 0;
  int sendStepMask;
  size_t totalCopySize;
  size_t totalPostSize;
  size_t totalSendSize;
  size_t totalRecvSize;
  size_t sendSizePerRound;
  size_t recvSizePerRound;
  uint8_t /*sdcclDataType_t*/ dtype;
  uint8_t /*sdcclDevRedOp_t*/ redOp;
  uint8_t /*sdcclPattern_t*/ pattern;
  uint8_t /*sdcclFunc_t*/ coll;
  uint8_t protocol;
  int state;
  char *sharedBuff[SDCCL_STEPS];
  int sharedSize[SDCCL_STEPS];

  int idle;

  // Element linking
  struct sdcclProxyArgs *next;
  struct sdcclProxyArgs *nextPeer;
  struct sdcclProxyArgs **proxyAppendPtr;

  /*for launch*/
  std::shared_ptr<sdcclSemaphore> semaphore;
  int opId;
  int step;

  // user buffer registration
  void *regHandle = nullptr;
  int regBufFlag = 0;

  // P2P operation slot management
  uint64_t p2pOpHash = -1;
  uint64_t p2pPeerOpHash = -1;
  size_t p2pSlotIdx = 0;
  size_t p2pPeerSlotIdx = 0;
  void *p2pRmtAddr = nullptr; // Remote address for P2P zero-copy

  union sdcclProxyOpSpecifics specifics;
};

struct sdcclProxyOp {
  struct sdcclProxyConnection *connection;
  ssize_t nbytes;
  uint64_t opCount;
  int root;
  struct sdcclProxyOp *next;
  int nsteps;
  int chunkSize;
  uint8_t sliceSteps;
  uint8_t chunkSteps;
  uint8_t channelId;
  uint8_t /*sdcclDataType_t*/ dtype;
  uint8_t /*sdcclDevRedOp_t*/ redOp;
  uint8_t /*sdcclFunc_t*/ coll;
  uint8_t /*sdcclPattern_t*/ pattern;
  void *kernelSyncPtr;
  uint8_t protocol;
  uint8_t reg;
  // collnet buffer reg handles
  void *sendMhandle;
  void *recvMhandle;
  uint8_t *sendbuff;
  uint8_t *recvbuff;

  union sdcclProxyOpSpecifics specifics;

  struct sdcclProxyOp *enqNext;
  unsigned long long sdcclFuncTimes;
  int peerRank;
  int rank;
  uint64_t groupHash;
  /**
   * TODO: just for test, we will delete the sdcclHeteroComm_t comm;
   **/
  sdcclHeteroComm_t comm;
  sdcclProxyArgs args;
  sdcclStream_t stream;
  sdcclEvent_t event; // used to record host/device func
  int selfCopy = 0;
};

#define SDCCL_MAX_NETDEVS 128

// ProxyOps are used to communicate between main thread and service thread
// Make sure we have enough to store two full rounds of operations on all
// channels. Otherwise we'd be unable to post half of them to free new elements.
#define MAX_OPS_PER_PEER (2 * MAXCHANNELS * SDCCL_MAX_WORK_ELEMENTS_P2P)

struct sdcclProxyOpsPool {
  struct sdcclProxyOp ops[MAX_OPS_PER_PEER * SDCCL_MAX_LOCAL_RANKS];
  volatile int nextOps;
  volatile int nextOpsEnd;
  volatile int freeOps[SDCCL_MAX_LOCAL_RANKS];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

struct sdcclProxyOps {
  pthread_mutex_t mutex;
  struct consPeer {
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next>
        sendQueue;
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next>
        recvQueue;
    struct consPeer *nextPeer;
    struct consPeer *prevPeer;
  };
  struct prodPeer {
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next>
        sendQueue;
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next>
        recvQueue;
  };

  struct consPeer *consPeers;
  struct prodPeer prodPeers;
  struct consPeer *consProgPeerHead;
  struct sdcclProxyOps *prodNextChannel;
  struct sdcclProxyOps *prodPrevChannel;
  struct sdcclProxyOps *consNextChannel;
  struct sdcclProxyOps *consPrevChannel;
};

struct sdcclProxySharedP2p {
  int refcount;
  int size;
  char *cudaBuff;
  char *hostBuff;
  struct sdcclProxyArgs *proxyAppend[MAXCHANNELS]; // Separate send and recv
};

struct sdcclProxyPeer {
  struct sdcclProxySharedP2p send;
  struct sdcclProxySharedP2p recv;
};

struct sdcclSharedNetComms {
  void *sendComm[MAXCHANNELS];
  void *recvComm[MAXCHANNELS];
  int sendRefCount[MAXCHANNELS];
  int recvRefCount[MAXCHANNELS];
};

struct sdcclProxyPool;
struct sdcclProxyProgressState {
  // Used by main threads to send work to progress thread
  struct sdcclProxyOpsPool *opsPool;
  char opsPoolShmSuffix[6];

  pthread_t thread;
  volatile int stop;
  struct sdcclProxyPeer **localPeers;
  struct sdcclSharedNetComms *netComms[SDCCL_MAX_NETDEVS];
  struct sdcclProxyArgs *active;
  struct sdcclProxyArgs *pool;
  struct sdcclProxyPool *pools;
  int nextOps;
};

// Expected proxy response fifo
struct sdcclExpectedProxyResponse {
  void *opId;
  int respSize;
  bool done;
  void *respBuff;
  sdcclResult_t res;
  struct sdcclExpectedProxyResponse *next;
};

struct sdcclProxyAsyncOp {
  int type;
  bool done;
  sdcclProxyArgs args;
  struct sdcclProxyConnection *connection;
  int reqSize, respSize;
  char *reqBuff, *respBuff;
  void *opId;
  sdcclProxyAsyncOp *prev;
  sdcclProxyAsyncOp *next;
};

// Common response header for all proxyOps
// We pack this into a struct to reduce the number of blocking send and recv
// calls
struct sdcclProxyRpcResponseHeader {
  void *opId;
  sdcclResult_t res;
  int respSize;
};

// UDS support
struct sdcclIpcHdr {
  int type;
  int rank;
  int reqSize;
  int respSize;
  void *opId;
  uint64_t data[16]; // 128-bytes
};

struct sdcclProxyState {
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[SDCCL_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  struct sdcclNetAdaptor *netAdaptor;
  volatile uint32_t *abortFlag;
  // Service threads
  pthread_t thread;
  pthread_t threadUDS;
  struct sdcclSocket listenSock;
  struct sdcclSocket ipcSock;
  int stop;
  sdcclResult_t asyncResult;
  int nRanks;

  // Used by main thread
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  union sdcclSocketAddress *peerAddresses;
  struct sdcclSocket peerSock;
  struct sdcclProxyOps proxyOps[MAXCHANNELS];

  struct sdcclProxyOps *prodProgChannelHead; /*producer*/
  struct sdcclProxyOps *consProgChannelHead; /*consumer*/

  void **sharedDevMems;
  struct sdcclIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t *peerAddressesUDS;         // cuMem API support (UDS)

  // Progress thread
  struct sdcclProxyProgressState progressState;

  // Kernel thread
  struct sdcclProxyKernelState kernelState;
  sdcclUniRunnerState uniRunnerState;

  // Queue of expected responses from the proxy
  struct sdcclExpectedProxyResponse *expectedResponses;

  // flag indicating if the proxy is initialized.
  // This flag is used for lazy initialization of the proxy.
  // Cooperate with SDCCL_RUNTIME_PROXY environment variable.
  int initialized;
};

enum proxyConnectState {
  connUninitialized = 0,
  connInitialized = 1,
  connSharedInitialized = 2,
  connSetupDone = 3,
  connConnected = 4,
  numConnStates = 5
};

struct sdcclProxyConnection {
  int send, transport, shared;
  int tpLocalRank, sameProcess;
  struct sdcclSocket *sock;
  struct sdcclTransportComm *tcomm;
  struct sdcclProxyArgs *proxyAppend;
  struct sdcclProxyArgs **proxyAppendPtr;
  void *transportResources;
  sdcclNetDeviceHandle_t *netDeviceHandle;
  void *mhandles[SDCCL_NUM_PROTOCOLS];
  proxyConnectState state;
  struct sdcclCollNetSharedRes *collNet;
  int needsProxyProgress;
};

typedef sdcclResult_t (*threadFunc_t)(struct sdcclProxyArgs *);

enum proxyMode { proxyRing = 0, proxyFrom = 1, proxyTo = 2 };

void *sdcclProxyService(void *args);
sdcclResult_t sdcclProxySaveOp(struct sdcclHeteroComm *comm,
                                 struct sdcclProxyOp *proxyOp,
                                 bool *justInquire = NULL);
sdcclResult_t sdcclProxyComputeP2p(struct sdcclInfo *info,
                                     struct sdcclProxyOp *proxyOp, int reg);
sdcclResult_t sdcclProxyStart(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclProxyInit(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclProxyCreate(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclProxyConnect(struct sdcclHeteroComm *comm, int transport,
                                  int send, int proxyRank,
                                  struct sdcclProxyConnector *proxyConn);

void *sdcclProxyKernelService(void *args);

// Only sdcclProxyMsgConnect & sdcclProxyMsgStop types are used for now.
enum sdcclProxyMsgType {
  sdcclProxyMsgInit = 1,
  sdcclProxyMsgSharedInit = 2,
  sdcclProxyMsgSetup = 3,
  sdcclProxyMsgConnect = 4,
  sdcclProxyMsgStart = 5,
  sdcclProxyMsgClose = 6,
  sdcclProxyMsgAbort = 7,
  sdcclProxyMsgStop = 8,
  sdcclProxyMsgGetFd = 9, // cuMem API support (UDS)
  sdcclProxyMsgRegister = 10,
  sdcclProxyMsgDeregister = 11,
  sdcclProxyMsgRegMr = 12,
  sdcclProxyMsgDeregMr = 13,
  sdcclProxyMsgSendRecv = 14
};

// This function is called by a client of the proxy that needs to invoke any of
// the non-progress proxyOp types Call this function on the client, supplying a
// locally unique opId. Then, poll on the return value of
// sdcclPollProxyResponse(), supplying the same opId to confirm the operation
// has completed
sdcclResult_t sdcclProxyCallAsync(struct sdcclHeteroComm *comm,
                                    struct sdcclProxyConnector *proxyConn,
                                    int type, void *reqBuff, int reqSize,
                                    int respSize, void *opId);

// This function will internally call sdcclProxyCallAsync() and spin until
// sdcclPollProxyResponse() confirms the result is received
sdcclResult_t sdcclProxyCallBlocking(struct sdcclHeteroComm *comm,
                                       struct sdcclProxyConnector *proxyConn,
                                       int type, void *reqBuff, int reqSize,
                                       void *respBuff, int respSize);
sdcclResult_t sdcclPollProxyResponse(struct sdcclHeteroComm *comm,
                                       struct sdcclProxyConnector *proxyConn,
                                       void *respBuff, void *opId);

// UDS support
sdcclResult_t sdcclProxyClientGetFdBlocking(struct sdcclHeteroComm *comm,
                                              int rank, void *handle,
                                              int *convertedFd);

sdcclResult_t sdcclProxyStop(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclProxyShmUnlink(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclProxyDestroy(struct sdcclHeteroComm *comm);

#endif
