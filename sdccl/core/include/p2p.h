#ifndef SDCCL_INT_P2P_H_
#define SDCCL_INT_P2P_H_

#include "adaptor.h"
#include "check.h"
#include "comm.h"
#include "device.h"
#include "register.h"
#include "shmutils.h"
#include "transport.h"
#include <stddef.h>

extern int64_t sdcclP2pBufferSize;
extern int64_t sdcclP2pChunkSize;
extern int64_t sdcclP2pChunks;
size_t computeP2pChunkSize(size_t nbytes);
#define SDCCL_P2P_MAX_STEPS 16
#define SDCCL_P2P_MAX_OPS                                                     \
  (SDCCL_P2P_MAX_STEPS * 2) // Maximum number of concurrent P2P operation pairs
#define SDCCL_P2P_IPC_HANDLE_SIZE SDCCL_IPC_HANDLE_SIZE

#ifdef __cplusplus
extern "C" {
#endif

struct sdcclP2pRequest {
  size_t size;
  int refcount;
};

struct sdcclP2pIpcDesc {
  sdcclIpcHandleData handleData; // Actual IPC handle data
  size_t size;
};

struct sdcclP2pBuff {
  void *directPtr;
  size_t size;
  sdcclP2pIpcDesc ipcDesc;
};

struct sdcclP2pConnectInfo {
  int rank;
  int read;
  sdcclP2pBuff p2pBuff;
  sdcclShmIpcDesc_t desc;
};

// Synchronization structure for a single P2P operation pair
struct sdcclP2pSyncSlot {
  uint64_t sendHead;
  uint64_t recvTail;
  uint64_t opHash; // Hash identifying which operation owns this slot
  int done;        // 1 = slot is free, 0 = slot is in use
  int peerDone;    // 1 = slot is free, 0 = slot is in use
};

struct p2pRegInfo {
  int copyDone;    // Indicates if the copy operation is complete
  int copyStarted; // Indicates if the copy operation has started
  uintptr_t
      ipcUserOffset; // Per-slot IPC offset (recv-side writes, send-side reads)
  int ipcRegReady;   // 1 = ipcUserOffset is valid for current op
};

struct sdcclP2pShm {
  // Array of synchronization slots for multiple concurrent operations
  struct sdcclP2pSyncSlot slots[SDCCL_P2P_MAX_OPS];
  // Array of registration info for multiple concurrent operations
  struct p2pRegInfo regInfos[SDCCL_P2P_MAX_OPS];
};

// need to make sure this matches sdcclP2pShmProxyInfo in p2p.cc
struct sdcclP2pShmProxyInfo {
  // CPU side
  struct sdcclP2pShm *shm;
  sdcclShmIpcDesc_t desc;

  // Device side
  char *recvFifo;
  sdcclStream_t stream;
  sdcclEvent_t events[SDCCL_P2P_MAX_STEPS];
};

struct sdcclP2pResources {
  // Shared memory for synchronization
  struct sdcclP2pShm *shm;
  sdcclShmIpcDesc_t desc;

  // Proxy info for async operations
  struct sdcclP2pShmProxyInfo proxyInfo;
};

// Bootstrap tag for one-sided IPC registration (first call only).
// Recv-side sends to peer with tag = P2P_IPC_TAG_BASE + recvRank.
// Send-side receives from peer with tag = P2P_IPC_TAG_BASE + peerRank.
// Subsequent calls use SHM for offset refresh (zero bootstrap overhead).
#define P2P_IPC_TAG_BASE 4000

sdcclResult_t sdcclP2pProxySend(struct sdcclP2pResources *resources,
                                  void *data, size_t size,
                                  struct sdcclProxyArgs *args);

sdcclResult_t sdcclP2pProxyRecv(struct sdcclP2pResources *resources,
                                  void *data, size_t size,
                                  struct sdcclProxyArgs *args);

sdcclResult_t sdcclP2pProxySelfCopy(struct sdcclP2pResources *resources,
                                      void *sendData, void *recvData,
                                      size_t size,
                                      struct sdcclProxyArgs *args);

sdcclResult_t sdcclP2pSendProxySetup(struct sdcclProxyConnection *connection,
                                       struct sdcclProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize, int *done);

sdcclResult_t sdcclP2pRecvProxySetup(struct sdcclProxyConnection *connection,
                                       struct sdcclProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize, int *done);

sdcclResult_t
sdcclP2pSendProxyConnect(struct sdcclProxyConnection *connection,
                          struct sdcclProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize, int *done);

sdcclResult_t
sdcclP2pRecvProxyConnect(struct sdcclProxyConnection *connection,
                          struct sdcclProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize, int *done);

sdcclResult_t sdcclP2pProxyRegister(struct sdcclProxyConnection *connection,
                                      struct sdcclProxyState *proxyState,
                                      void *reqBuff, int reqSize,
                                      void *respBuff, int respSize, int *done);

sdcclResult_t
sdcclP2pProxyDeregister(struct sdcclProxyConnection *connection,
                         struct sdcclProxyState *proxyState, void *reqBuff,
                         int reqSize, int *done);

sdcclResult_t
sdcclP2pAllocateShareableBuffer(size_t size, int directMap,
                                 struct sdcclP2pIpcDesc *ipcDesc, void **ptr);

sdcclResult_t sdcclP2pImportShareableBuffer(struct sdcclHeteroComm *comm,
                                              int peer, size_t size,
                                              struct sdcclP2pIpcDesc *ipcDesc,
                                              void **devMemPtr);

sdcclResult_t
sdcclP2pRegisterBuffer(struct sdcclHeteroComm *comm, const void *userbuff,
                        size_t buffSize, struct sdcclConnector **peerConns,
                        int *peerRanks, int nPeers, bool isSender,
                        int *regBufFlag, uintptr_t *offsetOut,
                        uintptr_t **peerRmtAddrsOut, size_t shmRegSlotIdx);

sdcclResult_t sdcclP2pDeregisterBuffer(struct sdcclHeteroComm *comm,
                                         struct sdcclIpcRegInfo *info);

sdcclResult_t sdcclP2pSendProxyFree(struct sdcclP2pResources *resources);

sdcclResult_t sdcclP2pRecvProxyFree(struct sdcclP2pResources *resources);

void setP2pSlotInfo(int rank, int peerRank, size_t size, sdcclDataType_t dtype,
                    int isRecv, uint64_t *opHash, size_t *slotIdx);

#ifdef __cplusplus
}
#endif

#endif // SDCCL_INT_P2P_H_