/*************************************************************************
 * Copyright (c) 2023 BAAI. All rights reserved.
 *
 * This file contains common InfiniBand structures and constants
 * shared between IBRC and UCX adaptors.
 ************************************************************************/

#ifndef SDCCL_IB_COMMON_H_
#define SDCCL_IB_COMMON_H_

#include "sdccl_net.h"
#include "ibvcore.h"
#include "ibvwrap.h"
#include "net.h"
#include "onesided.h"
#include <pthread.h>
#include <stdint.h>

// Backward-compat alias so adaptor code can keep using the old name.
typedef struct sdcclOneSideHandleInfo sdcclIbGlobalHandleInfo;

#define MAXNAMESIZE 64
#define MAX_IB_DEVS 32
#define SDCCL_IB_MAX_DEVS_PER_NIC 2
#define SDCCL_NET_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME                                                    \
  (MAXNAMESIZE * SDCCL_IB_MAX_DEVS_PER_NIC) + SDCCL_IB_MAX_DEVS_PER_NIC
#define MAX_IB_VDEVS MAX_IB_DEVS * 8

#define ENABLE_TIMER 0
#define SDCCL_IB_MAX_QPS 128
#define SDCCL_NET_IB_MAX_RECVS 8
#define MAX_REQUESTS (SDCCL_NET_MAX_REQUESTS * SDCCL_NET_IB_MAX_RECVS)

enum sdcclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MLX4 = 2
};

static const char *ibProviderName[]
    __attribute__((unused)) = {"NONE", "MLX5", "MLX4"};

extern int64_t sdcclParamIbMergeVfs(void);
extern int64_t sdcclParamIbAdaptiveRouting(void);
extern int64_t sdcclParamIbMergeNics(void);

struct sdcclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  struct ibv_mr *mr;
};

struct sdcclIbMrCache {
  struct sdcclIbMr *slots;
  int capacity, population;
};

struct sdcclIbStats {
  int fatalErrorCount;
};

struct sdcclIbDev {
  pthread_mutex_t lock;
  int device;
  int ibProvider;
  uint64_t guid;
  struct ibv_port_attr portAttr;
  int portNum;
  int link;
  int speed;
  struct ibv_context *context;
  int pdRefs;
  struct ibv_pd *pd;
  char devName[MAXNAMESIZE];
  char *pciPath;
  int realPort;
  int maxQp;
  struct sdcclIbMrCache mrCache;
  struct sdcclIbStats stats;
  int ar; // ADAPTIVE_ROUTING
  int isSharpDev;
  struct {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
  int dmaBufSupported;
} __attribute__((aligned(64)));

struct sdcclIbMergedDev {
  int ndevs;
  int devs[SDCCL_IB_MAX_DEVS_PER_NIC];
  sdcclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME];
} __attribute__((aligned(64)));

struct sdcclIbQpInfo {
  uint32_t qpn;
  struct ibv_ece ece;
  int eceSupported;
  int devIndex;
};

struct sdcclIbDevInfo {
  uint32_t lid;
  uint8_t ibPort;
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint64_t spn;
  uint64_t iid;
  uint32_t fifoRkey;
  union ibv_gid remoteGid;
};

struct sdcclIbGidInfo {
  uint8_t linkLayer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct sdcclIbMrHandle {
  ibv_mr *mrs[SDCCL_IB_MAX_DEVS_PER_NIC];
};

#define SDCCL_NET_IB_REQ_UNUSED 0
#define SDCCL_NET_IB_REQ_SEND 1
#define SDCCL_NET_IB_REQ_RECV 2
#define SDCCL_NET_IB_REQ_FLUSH 3
#define SDCCL_NET_IB_REQ_IPUT 4
#define SDCCL_NET_IB_REQ_IGET 5

extern const char *reqTypeStr[];

#define SDCCL_IB_RETRANS_MAX_INFLIGHT 2048
#define SDCCL_IB_RETRANS_BUFFER_SIZE 1024
#define SDCCL_IB_RETRANS_MAX_CHUNK_SIZE (8 * 1024 * 1024)
#define SDCCL_IB_SRQ_SIZE 1024

#define SDCCL_IB_ACK_BUF_PADDING 40
#define SDCCL_IB_ACK_BUF_COUNT 64

struct sdcclIbRetransHdr {
  uint32_t magic;
  uint32_t seq;
  uint32_t size;
  uint32_t rkey;
  uint64_t remoteAddr;
  uint32_t immData;
  uint32_t padding;
} __attribute__((packed));

struct sdcclIbAckMsg {
  uint16_t peerId;
  uint16_t flowId;
  uint16_t path;
  uint16_t ackSeq;
  uint16_t sackBitmapCount;
  uint16_t padding;
  uint64_t timestampUs;
  uint64_t sackBitmap;
} __attribute__((packed));

struct sdcclIbCtrlQp {
  struct ibv_qp *qp;
  struct ibv_cq *cq;
  struct ibv_ah *ah;
  uint32_t remoteQpn;
  uint32_t remoteQkey;
};

struct sdcclIbRetransRecvBuf {
  void *buffer;
  struct ibv_mr *mr;
  size_t size;
  int inUse;
};

struct sdcclIbSrqMgr {
  void *srq;
  struct ibv_cq *cq;
  struct sdcclIbRetransRecvBuf bufs[SDCCL_IB_SRQ_SIZE];
  int bufCount;
  // Buffer management for SRQ (similar to UCCL)
  int freeBufIndices[SDCCL_IB_SRQ_SIZE]; // Stack of free buffer indices
  int freeBufCount;                       // Number of free buffers available
  int postSrqCount; // Number of recv WRs that need to be posted to SRQ
};

struct sdcclIbRetransEntry {
  uint32_t seq;
  uint32_t size;
  uint64_t sendTimeUs;
  uint64_t remoteAddr;
  void *data;
  uint32_t lkeys[SDCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t rkeys[SDCCL_IB_MAX_DEVS_PER_NIC];
  int retryCount;
  int valid;
};

struct sdcclIbRetransState {
  uint32_t sendSeq;
  uint32_t sendUna;
  uint32_t recvSeq;

  struct sdcclIbRetransEntry buffer[SDCCL_IB_RETRANS_MAX_INFLIGHT];
  int bufferHead;
  int bufferTail;
  int bufferCount;

  uint64_t lastAckTimeUs;
  uint64_t rtoUs;
  uint64_t srttUs;
  uint64_t rttvarUs;

  uint64_t totalSent;
  uint64_t totalRetrans;
  uint64_t totalAcked;
  uint64_t totalTimeout;

  int enabled;
  int maxRetry;
  int ackInterval;
  uint32_t minRtoUs;
  uint32_t maxRtoUs;
  int retransQPIndex;
  uint32_t lastAckSeq;
  uint64_t lastAckSendTimeUs;
};

struct sdcclIbQp {
  struct ibv_qp *qp;
  int devIndex;
  int remDevIdx;
};

struct sdcclIbSendFifo {
  uint64_t addr;
  size_t size;
  uint32_t rkeys[SDCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[24];
};

struct sdcclIbRequest {
  struct sdcclIbNetCommBase *base;
  int type;
  struct sdcclSocket *sock;
  int events[SDCCL_IB_MAX_DEVS_PER_NIC];
  struct sdcclIbNetCommDevBase *devBases[SDCCL_IB_MAX_DEVS_PER_NIC];
  int nreqs;
  union {
    struct {
      int size;
      void *data;
      uint32_t lkeys[SDCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int *sizes;
    } recv;
  };
};

struct sdcclIbListenComm {
  int dev;
  struct sdcclSocket sock;
  struct sdcclIbCommStage stage;
};

struct sdcclIbConnectionMetadata {
  struct sdcclIbQpInfo qpInfo[SDCCL_IB_MAX_QPS];
  struct sdcclIbDevInfo devs[SDCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;

  uint32_t ctrlQpn[SDCCL_IB_MAX_DEVS_PER_NIC];
  union ibv_gid ctrlGid[SDCCL_IB_MAX_DEVS_PER_NIC];
  uint16_t ctrlLid[SDCCL_IB_MAX_DEVS_PER_NIC];
  int retransEnabled;
};

struct sdcclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  uint64_t pad[2];
  struct sdcclIbGidInfo gidInfo;
};

struct sdcclIbRemSizesFifo {
  int elems[MAX_REQUESTS][SDCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkeys[SDCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
  struct ibv_mr *mrs[SDCCL_IB_MAX_DEVS_PER_NIC];
  struct ibv_sge sge;
};

struct sdcclIbSendCommDev {
  struct sdcclIbNetCommDevBase base;
  struct ibv_mr *fifoMr;
  struct ibv_mr *putSignalScratchpadMr;

  struct sdcclIbCtrlQp ctrlQp;
  struct ibv_mr *ackMr;
  void *ackBuffer;
};

struct alignas(32) sdcclIbNetCommBase {
  int ndevs;
  bool isSend;
  struct sdcclIbRequest reqs[MAX_REQUESTS];
  struct sdcclIbQp qps[SDCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
  struct sdcclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  struct sdcclIbDevInfo remDevs[SDCCL_IB_MAX_DEVS_PER_NIC];
};

struct sdcclIbSendComm {
  struct sdcclIbNetCommBase base;
  struct sdcclIbSendFifo fifo[MAX_REQUESTS][SDCCL_NET_IB_MAX_RECVS];
  // Each dev correlates to a mergedIbDev
  struct sdcclIbSendCommDev devs[SDCCL_IB_MAX_DEVS_PER_NIC];
  struct sdcclIbRequest *fifoReqs[MAX_REQUESTS][SDCCL_NET_IB_MAX_RECVS];
  alignas(32) struct ibv_sge sges[SDCCL_NET_IB_MAX_RECVS];
  alignas(32) struct ibv_send_wr wrs[SDCCL_NET_IB_MAX_RECVS + 1];
  struct sdcclIbRemSizesFifo remSizesFifo;
  uint64_t fifoHead;
  uint64_t putSignalScratchpad;
  int ar;

  struct sdcclIbRetransState retrans;
  uint64_t lastTimeoutCheckUs;

  int outstandingSends;
  int outstandingRetrans;
  int maxOutstanding;

  struct sdcclIbRetransHdr retransHdrPool[32];
  struct ibv_mr *retransHdrMr;
};

struct sdcclIbGpuFlush {
  struct ibv_mr *hostMr;
  struct ibv_sge sge;
  struct sdcclIbQp qp;
};

struct alignas(32) sdcclIbRemFifo {
  struct sdcclIbSendFifo elems[MAX_REQUESTS][SDCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t flags;
};

struct alignas(16) sdcclIbRecvCommDev {
  struct sdcclIbNetCommDevBase base;
  struct sdcclIbGpuFlush gpuFlush;
  uint32_t fifoRkey;
  struct ibv_mr *fifoMr;
  struct ibv_sge fifoSge;
  struct ibv_mr *sizesFifoMr;
  struct sdcclIbCtrlQp ctrlQp;
  struct ibv_mr *ackMr;
  void *ackBuffer;

  void *retransRecvBufs[32];
  struct ibv_mr *retransRecvMr;
  int retransRecvBufCount;
};

struct alignas(32) sdcclIbRecvComm {
  struct sdcclIbNetCommBase base;
  struct sdcclIbRecvCommDev devs[SDCCL_IB_MAX_DEVS_PER_NIC];
  struct sdcclIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][SDCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;

  struct sdcclIbRetransState retrans;
  struct sdcclIbSrqMgr srqMgr;
};

// Global arrays (declared as extern, defined in adaptor files)
extern struct sdcclIbDev sdcclIbDevs[MAX_IB_DEVS];
extern struct sdcclIbMergedDev sdcclIbMergedDevs[MAX_IB_VDEVS];

// Global variables (declared as extern, defined in adaptor files)
extern char sdcclIbIfName[MAX_IF_NAME_SIZE + 1];
extern union sdcclSocketAddress sdcclIbIfAddr;
extern int sdcclNMergedIbDevs;
extern int sdcclNIbDevs;
extern pthread_mutex_t sdcclIbLock;
extern int sdcclIbRelaxedOrderingEnabled;
extern pthread_t sdcclIbAsyncThread;

// Parameter functions
extern int64_t sdcclParamIbGidIndex(void);
extern int64_t sdcclParamIbRoceVersionNum(void);
extern int64_t sdcclParamIbTimeout(void);
extern int64_t sdcclParamIbRetryCnt(void);
extern int64_t sdcclParamIbPkey(void);
extern int64_t sdcclParamIbUseInline(void);
extern int64_t sdcclParamIbSl(void);
extern int64_t sdcclParamIbTc(void);
extern int64_t sdcclParamIbArThreshold(void);
extern int64_t sdcclParamIbPciRelaxedOrdering(void);
extern int64_t sdcclParamIbAdaptiveRouting(void);
extern int64_t sdcclParamIbDisable(void);
extern int64_t sdcclParamIbMergeVfs(void);
extern int64_t sdcclParamIbMergeNics(void);
extern int64_t sdcclParamIbQpsPerConn(void);

extern sa_family_t envIbAddrFamily(void);
extern void *envIbAddrRange(sa_family_t af, int *mask);
extern sa_family_t getGidAddrFamily(union ibv_gid *gid);
extern bool matchGidAddrPrefix(sa_family_t af, void *prefix, int prefixlen,
                               union ibv_gid *gid);
extern bool configuredGid(union ibv_gid *gid);
extern bool linkLocalGid(union ibv_gid *gid);
extern bool validGid(union ibv_gid *gid);
extern sdcclResult_t sdcclIbRoceGetVersionNum(const char *deviceName,
                                                int portNum, int gidIndex,
                                                int *version);
extern sdcclResult_t sdcclUpdateGidIndex(struct ibv_context *context,
                                           uint8_t portNum, sa_family_t af,
                                           void *prefix, int prefixlen,
                                           int roceVer, int gidIndexCandidate,
                                           int *gidIndex);
extern sdcclResult_t sdcclIbGetGidIndex(struct ibv_context *context,
                                          uint8_t portNum, int gidTblLen,
                                          int *gidIndex);
extern sdcclResult_t sdcclIbGetPciPath(char *devName, char **path,
                                         int *realPort);
extern int sdcclIbWidth(int width);
extern int sdcclIbSpeed(int speed);
extern int sdcclIbRelaxedOrderingCapable(void);
extern int sdcclIbFindMatchingDev(int dev);
extern void *sdcclIbAsyncThreadMain(void *args);

extern int ibvWidths[];
extern int ibvSpeeds[];

extern int firstBitSet(int val, int max);

extern sdcclResult_t sdcclIbDevices(int *ndev);
extern sdcclResult_t sdcclIbGdrSupport(void);
extern sdcclResult_t sdcclIbDmaBufSupport(int dev);
extern sdcclResult_t sdcclIbFreeRequest(struct sdcclIbRequest *r);

struct sdcclIbCommonTestOps {
  const char *component;
  sdcclResult_t (*pre_check)(struct sdcclIbRequest *req);
  sdcclResult_t (*process_wc)(struct sdcclIbRequest *req, struct ibv_wc *wc,
                               int devIndex, bool *handled);
};

sdcclResult_t
sdcclIbCommonPostFifo(struct sdcclIbRecvComm *comm, int n, void **data,
                       size_t *sizes, int *tags, void **mhandles,
                       struct sdcclIbRequest *req,
                       void (*addEventFunc)(struct sdcclIbRequest *, int,
                                            struct sdcclIbNetCommDevBase *));

sdcclResult_t
sdcclIbCommonTestDataQp(struct sdcclIbRequest *r, int *done, int *sizes,
                         const struct sdcclIbCommonTestOps *ops);

static_assert((sizeof(struct sdcclIbNetCommBase) % 32) == 0,
              "sdcclIbNetCommBase size must be 32-byte multiple to ensure "
              "fifo is at proper offset");
static_assert((offsetof(struct sdcclIbSendComm, fifo) % 32) == 0,
              "sdcclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct sdcclIbSendFifo) % 32) == 0,
              "sdcclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct sdcclIbSendComm, sges) % 32) == 0,
              "sges must be 32-byte aligned");
static_assert((offsetof(struct sdcclIbSendComm, wrs) % 32) == 0,
              "wrs must be 32-byte aligned");
static_assert((offsetof(struct sdcclIbRecvComm, remFifo) % 32) == 0,
              "sdcclIbRecvComm fifo must be 32-byte aligned");
static_assert(
    sizeof(struct sdcclIbHandle) < SDCCL_NET_HANDLE_MAXSIZE,
    "sdcclIbHandle size must be smaller than SDCCL_NET_HANDLE_MAXSIZE");

static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we "
                                   "need up to 8 requests ids per completion");

#endif // SDCCL_IB_COMMON_H_
