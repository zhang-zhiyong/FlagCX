/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_NET_ADAPTOR_H_
#define SDCCL_NET_ADAPTOR_H_

#include "sdccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// MR registration flags for one-sided strong ordering
typedef enum {
  SDCCL_NET_MR_FLAG_NONE = 0,
  SDCCL_NET_MR_FLAG_FORCE_SO =
      (1 << 0), // Force strong ordering (disable relaxed ordering)
} sdcclNetMrFlag_t;

// Version history:
//   v1 — 22 function pointers: name, init, devices, getProperties,
//         listen, connect, accept, closeSend, closeRecv, closeListen,
//         regMr, regMrDmaBuf, deregMr, isend, irecv, iflush, test,
//         iput, iget, iputSignal, getDevFromName
struct sdcclNetAdaptor_v1 {
  // Basic functions
  const char *name;
  sdcclResult_t (*init)();
  sdcclResult_t (*devices)(int *ndev);
  sdcclResult_t (*getProperties)(int dev, void *props);

  // Setup functions
  sdcclResult_t (*listen)(int dev, void *handle, void **listenComm);
  sdcclResult_t (*connect)(int dev, void *handle, void **sendComm);
  sdcclResult_t (*accept)(void *listenComm, void **recvComm);
  sdcclResult_t (*closeSend)(void *sendComm);
  sdcclResult_t (*closeRecv)(void *recvComm);
  sdcclResult_t (*closeListen)(void *listenComm);

  // Memory region functions
  sdcclResult_t (*regMr)(void *comm, void *data, size_t size, int type,
                          int mrFlags, void **mhandle);
  sdcclResult_t (*regMrDmaBuf)(void *comm, void *data, size_t size, int type,
                                uint64_t offset, int fd, int mrFlags,
                                void **mhandle);
  sdcclResult_t (*deregMr)(void *comm, void *mhandle);

  // Two-sided functions
  sdcclResult_t (*isend)(void *sendComm, void *data, size_t size, int tag,
                          void *mhandle, void *phandle, void **request);
  sdcclResult_t (*irecv)(void *recvComm, int n, void **data, size_t *sizes,
                          int *tags, void **mhandles, void **phandles,
                          void **request);
  sdcclResult_t (*iflush)(void *recvComm, int n, void **data, int *sizes,
                           void **mhandles, void **request);
  sdcclResult_t (*test)(void *request, int *done, int *sizes);

  // One-sided (per-window MR: separate src/dst handles for independent buffers)
  sdcclResult_t (*iput)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                         size_t size, int srcRank, int dstRank,
                         void **srcHandles, void **dstHandles, void **request);
  // RDMA READ: pull data from remote srcRank into local dstRank buffer
  sdcclResult_t (*iget)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                         size_t size, int srcRank, int dstRank,
                         void **srcHandles, void **dstHandles, void **request);
  // Data + signal combined (NCCL GIN-aligned: enables chained WRITE + ATOMIC)
  // When size == 0, only signal ATOMIC is posted (signal-only mode)
  sdcclResult_t (*iputSignal)(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                               size_t size, int srcRank, int dstRank,
                               void **srcHandles, void **dstHandles,
                               uint64_t signalOff, void **signalHandles,
                               uint64_t signalValue, void **request);

  // Device name lookup
  sdcclResult_t (*getDevFromName)(char *name, int *dev);
};
#define sdcclNetAdaptor sdcclNetAdaptor_v1

// Versioned export symbol name
#define SDCCL_NET_ADAPTOR_PLUGIN_SYMBOL_V1 sdcclNetAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // SDCCL_NET_ADAPTOR_H_
