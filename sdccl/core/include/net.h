/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_INT_NET_H_
#define SDCCL_INT_NET_H_

#include "check.h"
#include "comm.h"
#include "device.h"
#include "sdccl_net.h"
#include "register.h"
#include <socket.h>

typedef char sdcclNetHandle_t[SDCCL_NET_HANDLE_MAXSIZE];

extern int64_t sdcclNetBufferSize;
extern int64_t sdcclNetChunkSize;
extern int64_t sdcclNetChunks;

enum sdcclNetState {
  sdcclNetStateInit = 0,
  sdcclNetStateEnabled = 1,
  sdcclNetStateDisabled = 2
};
extern enum sdcclNetState sdcclNetStates[3];
#define SDCCL_NET_MAX_STEPS 16
#define SDCCL_MAX_NET_SIZE_BYTES (1 * 1024 * 1024 * 1024 * 1024L)

sdcclResult_t sdcclNetInit(struct sdcclHeteroComm *comm);
int sdcclNetVersion(struct sdcclHeteroComm *comm);

// Test whether the current GPU support GPU Direct RDMA.
sdcclResult_t sdcclGpuGdrSupport(struct sdcclHeteroComm *comm,
                                   int *gdrSupport);

// Network adaptor declarations
extern struct sdcclNetAdaptor sdcclNetSocket;
extern struct sdcclNetAdaptor sdcclNetIb;

struct sendNetResources {
  void *netSendComm;
  struct sdcclSendMem *sendMem;
  struct sdcclRecvMem *recvMem;

  struct sdcclHeteroComm *commPtr;
  struct sdcclNetAdaptor *netAdaptor;
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int ptrSupport;
  int maxRecvs;
  uint64_t *gdcSync;
  void *gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char *buffers[SDCCL_NUM_PROTOCOLS];
  int buffSizes[SDCCL_NUM_PROTOCOLS];
  void *mhandles[1]; /*just one for memory copy from device to gdr buffer*/
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  sdcclNetDeviceType netDeviceType;
  sdcclNetDeviceHandle_t *netDeviceHandle;
  sdcclStream_t cpStream;
  sdcclEvent_t cpEvents[SDCCL_NET_MAX_STEPS];
};

struct recvNetResources {
  void *netListenComm;
  void *netRecvComm;
  struct sdcclSendMem *sendMem;
  struct sdcclRecvMem *recvMem;

  struct sdcclHeteroComm *commPtr;
  struct sdcclNetAdaptor *netAdaptor;
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int ptrSupport;
  int needFlush;
  int maxRecvs;
  uint64_t *gdcSync;
  uint64_t *gdcFlush;
  void *gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char *buffers[SDCCL_NUM_PROTOCOLS];
  int buffSizes[SDCCL_NUM_PROTOCOLS];
  void *mhandles[SDCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  sdcclNetDeviceType netDeviceType;
  sdcclNetDeviceHandle_t *netDeviceHandle;
  sdcclStream_t cpStream;
  sdcclEvent_t cpEvents[SDCCL_NET_MAX_STEPS];
};

enum sdcclIbCommState {
  sdcclIbCommStateStart = 0,
  sdcclIbCommStateConnect = 1,
  sdcclIbCommStateAccept = 3,
  sdcclIbCommStateSend = 4,
  sdcclIbCommStateRecv = 5,
  sdcclIbCommStateConnecting = 6,
  sdcclIbCommStateConnected = 7,
  sdcclIbCommStatePendingReady = 8,
};

struct sdcclIbCommStage {
  enum sdcclIbCommState state;
  int offset;
  void *buffer;
  void *comm;
};

struct sendRecvDataInfo {
  void *data;
  size_t size;
};

struct sdcclIbHandle {
  union sdcclSocketAddress connectAddr; // Filled by the target
  uint64_t magic;                        // random number to help debugging
  struct sdcclIbCommStage stage; // Used by the other side when connecting
};

sdcclResult_t sdcclSendRegMr(sdcclHeteroComm_t comm, void *data, size_t size,
                               int peer, int channel);
sdcclResult_t sdcclRecvRegMr(sdcclHeteroComm_t comm, void *data, size_t size,
                               int peer, int channel);
sdcclResult_t sdcclProxySend(sendNetResources *resources, void *data,
                               size_t size, sdcclProxyArgs *args);
sdcclResult_t sdcclProxyRecv(recvNetResources *resources, void *data,
                               size_t size, sdcclProxyArgs *args);
sdcclResult_t sdcclSend(sdcclHeteroComm_t comm, void *data, size_t size,
                          int peer, int channel);
sdcclResult_t sdcclRecv(sdcclHeteroComm_t comm, void *data, size_t size,
                          int peer, int channel);
sdcclResult_t sdcclSendProxyFree(sendNetResources *resources);
sdcclResult_t sdcclRecvProxyFree(recvNetResources *resources);

sdcclResult_t sdcclNetRegisterBuffer(sdcclHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct sdcclConnector **peerConns,
                                       int nPeers, int *outRegBufFlag,
                                       void **outHandle);
sdcclResult_t sdcclNetDeregisterBuffer(void *comm,
                                         struct sdcclProxyConnector *proxyConn,
                                         void *handle);

#endif
