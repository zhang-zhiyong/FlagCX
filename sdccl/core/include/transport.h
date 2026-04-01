/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_TRANSPORT_H_
#define SDCCL_TRANSPORT_H_

#include "core.h"
#include "device.h"

#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"

extern struct sdcclTransport p2pTransport;
extern struct sdcclTransport shmTransport;
extern struct sdcclTransport netTransport;
extern struct sdcclTransport collNetTransport;

extern struct sdcclTransport *sdcclTransports[];

// Forward declarations
struct sdcclRing;
struct sdcclConnector;
struct sdcclHeteroComm;

struct sdcclPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  struct sdcclHeteroComm *comm;
  int cudaCompCap;
};

#define CONNECT_SIZE 128
struct sdcclConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010
/*
#define NVLS_HANDLE_SIZE 64
struct sdcclNvlsSharedRes {
  int refCount;
  CUmulticastObjectProp properties;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t size;
  size_t granularity;
  CUmemGenericAllocationHandle mcHandle; // Multicast handle for NVLS buffer
  char* mcBuff; // Multicast NVLS buffer address
  CUmemGenericAllocationHandle ucHandle; // Unicast Handle for NVLS buffer
  char* ucBuff; // Unicast NVLS buffer address
  char shareableHandle[NVLS_HANDLE_SIZE];
  size_t ucGran;
  int nChannels;
  struct sdcclShmemCollBuff nvlsShmem;
  void *nvlsShmemHandle;
};
*/
#endif /* CUDART_VERSION >= 12010 */

struct sdcclCollNetSharedRes {
  int refCount;
  int size;
  char *cudaBuff;
  char *hostBuff;
  struct sdcclProxyArgs *proxyAppend[2 * SDCCL_MAX_NETDEVS];
  void *resources;
  int nChannels;
  size_t buffSize;
};

struct sdcclTransportComm {
  sdcclResult_t (*setup)(struct sdcclHeteroComm *comm,
                          struct sdcclTopoGraph *graph,
                          struct sdcclPeerInfo *, struct sdcclPeerInfo *,
                          struct sdcclConnect *, struct sdcclConnector *,
                          int channelId, int connIndex);
  sdcclResult_t (*connect)(struct sdcclHeteroComm *comm,
                            struct sdcclConnect *, int nranks, int rank,
                            struct sdcclConnector *);
  sdcclResult_t (*free)(struct sdcclConnector *);
  sdcclResult_t (*proxySharedInit)(struct sdcclProxyConnection *connection,
                                    struct sdcclProxyState *proxyState,
                                    int nChannels);
  sdcclResult_t (*proxySetup)(struct sdcclProxyConnection *connection,
                               struct sdcclProxyState *proxyState,
                               void *reqBuff, int reqSize, void *respBuff,
                               int respSize, int *done);
  sdcclResult_t (*proxyConnect)(struct sdcclProxyConnection *connection,
                                 struct sdcclProxyState *proxyState,
                                 void *reqBuff, int reqSize, void *respBuff,
                                 int respSize, int *done);
  sdcclResult_t (*proxyFree)(struct sdcclProxyConnection *connection,
                              struct sdcclProxyState *proxyState);
  sdcclResult_t (*proxyProgress)(struct sdcclProxyState *proxyState,
                                  struct sdcclProxyArgs *);
  sdcclResult_t (*proxyRegister)(struct sdcclProxyConnection *connection,
                                  struct sdcclProxyState *proxyState,
                                  void *reqBuff, int reqSize, void *respBuff,
                                  int respSize, int *done);
  sdcclResult_t (*proxyDeregister)(struct sdcclProxyConnection *connection,
                                    struct sdcclProxyState *proxyState,
                                    void *reqBuff, int reqSize, int *done);
};

struct sdcclTransport {
  const char name[8];
  sdcclResult_t (*canConnect)(int *, struct sdcclTopoServer *topoServer,
                               struct sdcclTopoGraph *graph,
                               struct sdcclPeerInfo *,
                               struct sdcclPeerInfo *);
  struct sdcclTransportComm send;
  struct sdcclTransportComm recv;
};

sdcclResult_t sdcclTransportP2pConnect(struct sdcclHeteroComm *comm,
                                         int channelId, int nrecv,
                                         int *peerRecv, int nsend,
                                         int *peerSend, int connIndex);
sdcclResult_t sdcclTransportP2pSetup(struct sdcclHeteroComm *comm,
                                       struct sdcclTopoGraph *graph,
                                       int connIndex,
                                       int *highestTransportType = NULL);

sdcclResult_t sdcclNvlsInit(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclNvlsSetup(struct sdcclHeteroComm *comm,
                               struct sdcclHeteroComm *parent);
sdcclResult_t sdcclNvlsGraphRegisterBuffer(
    struct sdcclHeteroComm *comm, struct sdcclKernelPlan *plan,
    const void *sendbuff, void *recvbuff, size_t sendbuffSize,
    size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend,
    void **outRegBufRecv);
sdcclResult_t sdcclNvlsLocalRegisterBuffer(
    struct sdcclHeteroComm *comm, const void *sendbuff, void *recvbuff,
    size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed,
    void **outRegBufSend, void **outRegBufRecv);
sdcclResult_t sdcclNvlsFree(struct sdcclHeteroComm *comm);

enum { collNetRecv = 0, collNetSend = 1 };

int sdcclTransportCollNetSetup(struct sdcclHeteroComm *comm,
                                struct sdcclTopoGraph *collNetGraph,
                                struct sdcclChannel *channel, int masterRank,
                                int masterPeer, int collNetGraphChannelId,
                                int type, sdcclConnect *connect);
sdcclResult_t sdcclTransportCollNetCheck(struct sdcclHeteroComm *comm,
                                           int collNetSetupFail);
sdcclResult_t sdcclTransportCollNetFree(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclCollnetLocalRegisterBuffer(struct sdcclHeteroComm *comm,
                                                const void *userbuff,
                                                size_t buffSize, int type,
                                                int *outRegBufUsed,
                                                void **outHandle);
sdcclResult_t sdcclCollnetGraphRegisterBuffer(struct sdcclHeteroComm *comm,
                                                struct sdcclKernelPlan *plan,
                                                const void *userbuff,
                                                size_t buffSize, int type,
                                                int *outRegBufFlag,
                                                void **outHandle);
sdcclResult_t sdcclCollnetDeregBuffer(struct sdcclHeteroComm *comm,
                                        struct sdcclProxyConnector *proxyconn,
                                        void *handle);

#endif
