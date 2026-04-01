#ifndef SDCCL_REGISTER_H_
#define SDCCL_REGISTER_H_

#include "core.h"
#include "device.h"
#include <list>

#define SDCCL_IPC_HANDLE_SIZE 64

typedef union {
  char reserved[SDCCL_IPC_HANDLE_SIZE];
} sdcclIpcHandleData;

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct netRegInfo {
  uintptr_t buffer;
  size_t size;
};

struct sdcclRegNetHandle {
  void *handle = NULL;
  struct sdcclProxyConnector *proxyConn = NULL;
};

struct sdcclRegP2pHandle {
  void *handle = NULL;
  struct sdcclProxyConnector *proxyConn = NULL;
};

struct sdcclIpcImpInfo {
  void *rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
  // userOffset removed — sent fresh via SHM each call, never cached
};

struct sdcclPeerRegIpcAddr {
  uintptr_t *devPeerRmtAddrs;
  uintptr_t *hostPeerRmtAddrs;
};

struct sdcclIpcRegInfo {
  int peerRank;
  void *baseAddr;
  struct sdcclProxyConnector *ipcProxyconn;
  struct sdcclIpcImpInfo impInfo;
  bool handleReady;
};

struct sdcclRegItem {
  uintptr_t beginAddr = 0;
  uintptr_t endAddr = 0;
  int refCount = 1;
  std::list<std::pair<sdcclRegNetHandle, sdcclRegP2pHandle>> handles;
  void *homoRegHandle = nullptr;          // backend CCL handle (homo path only)
  sdcclIpcHandleData ipcHandleData = {}; // IPC handle bytes (both paths)
};

struct sdcclReg {
  // common attributes
  size_t pages;
  int refs;
  uintptr_t addr;
  uint32_t state;
  // net reg
  int nDevs;
  int devs[MAXCHANNELS];
  void **handles;
  // nvls reg
  uintptr_t baseAddr;
  size_t baseSize;
  size_t regSize;
  int dev;
  // collnet reg
  void *collnetHandle;
  struct sdcclProxyConnector *proxyconn;
};

struct sdcclRegCache {
  struct sdcclReg **slots;
  int capacity, population;
  uintptr_t pageSize;
  void *sComms[MAXCHANNELS];
  void *rComms[MAXCHANNELS];
};

sdcclResult_t sdcclRegCleanup(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclRegFind(struct sdcclHeteroComm *comm, const void *data,
                             size_t size, struct sdcclReg **reg);

#endif
