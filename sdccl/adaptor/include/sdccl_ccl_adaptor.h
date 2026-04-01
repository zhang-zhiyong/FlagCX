/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_CCL_ADAPTOR_H_
#define SDCCL_CCL_ADAPTOR_H_

#include "sdccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for opaque types not in sdccl.h
// (sdcclStream_t, sdcclWindow_t are already typedef'd in sdccl.h)
typedef struct sdcclInnerComm *sdcclInnerComm_t;
typedef struct sdcclInnerDevComm *sdcclInnerDevComm_t;
struct sdcclDevCommRequirements;
struct bootstrapState;

// Version history:
//   v1 — 34 function pointers: getVersion, getUniqueId, getErrorString,
//         getLastError, getStagedBuffer, commInitRank, commFinalize,
//         commDestroy, commAbort, commResume, commSuspend, commCount,
//         commGetDeviceNumber, commUserRank, commGetAsyncError, memAlloc,
//         memFree, commRegister, commDeregister, commWindowRegister,
//         commWindowDeregister, reduce, gather, scatter, broadcast,
//         allReduce, reduceScatter, allGather, alltoAll, alltoAllv,
//         send, recv, groupStart, groupEnd
struct sdcclCCLAdaptor_v1 {
  const char *name;
  // Basic functions
  sdcclResult_t (*getVersion)(int *version);
  sdcclResult_t (*getUniqueId)(sdcclUniqueId_t *uniqueId);
  const char *(*getErrorString)(sdcclResult_t result);
  const char *(*getLastError)(sdcclInnerComm_t comm);
  sdcclResult_t (*getStagedBuffer)(const sdcclInnerComm_t comm, void **buff,
                                    size_t size, int isRecv);

  // Communicator functions
  sdcclResult_t (*commInitRank)(sdcclInnerComm_t *comm, int nranks,
                                 sdcclUniqueId *commId, int rank,
                                 bootstrapState *bootstrap);
  sdcclResult_t (*commFinalize)(sdcclInnerComm_t comm);
  sdcclResult_t (*commDestroy)(sdcclInnerComm_t comm);
  sdcclResult_t (*commAbort)(sdcclInnerComm_t comm);
  sdcclResult_t (*commResume)(sdcclInnerComm_t comm);
  sdcclResult_t (*commSuspend)(sdcclInnerComm_t comm);
  sdcclResult_t (*commCount)(const sdcclInnerComm_t comm, int *count);
  sdcclResult_t (*commGetDeviceNumber)(const sdcclInnerComm_t comm,
                                        int *device);
  sdcclResult_t (*commUserRank)(const sdcclInnerComm_t comm, int *rank);
  sdcclResult_t (*commGetAsyncError)(sdcclInnerComm_t comm,
                                      sdcclResult_t *asyncError);
  sdcclResult_t (*memAlloc)(void **ptr, size_t size);
  sdcclResult_t (*memFree)(void *ptr);
  sdcclResult_t (*commRegister)(const sdcclInnerComm_t comm, void *buff,
                                 size_t size, void **handle);
  sdcclResult_t (*commDeregister)(const sdcclInnerComm_t comm, void *handle);
  // Symmetric functions
  sdcclResult_t (*commWindowRegister)(sdcclInnerComm_t comm, void *buff,
                                       size_t size, sdcclWindow_t *win,
                                       int winFlags);
  sdcclResult_t (*commWindowDeregister)(sdcclInnerComm_t comm,
                                         sdcclWindow_t win);

  // Communication functions
  sdcclResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, sdcclRedOp_t op,
                           int root, sdcclInnerComm_t comm,
                           sdcclStream_t stream);
  sdcclResult_t (*gather)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, int root,
                           sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*scatter)(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, int root,
                            sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype, int root,
                              sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*allReduce)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype,
                              sdcclRedOp_t op, sdcclInnerComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*reduceScatter)(const void *sendbuff, void *recvbuff,
                                  size_t recvcount, sdcclDataType_t datatype,
                                  sdcclRedOp_t op, sdcclInnerComm_t comm,
                                  sdcclStream_t stream);
  sdcclResult_t (*allGather)(const void *sendbuff, void *recvbuff,
                              size_t sendcount, sdcclDataType_t datatype,
                              sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*alltoAll)(const void *sendbuff, void *recvbuff, size_t count,
                             sdcclDataType_t datatype, sdcclInnerComm_t comm,
                             sdcclStream_t stream);
  sdcclResult_t (*alltoAllv)(const void *sendbuff, size_t *sendcounts,
                              size_t *sdispls, void *recvbuff,
                              size_t *recvcounts, size_t *rdispls,
                              sdcclDataType_t datatype, sdcclInnerComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*send)(const void *sendbuff, size_t count,
                         sdcclDataType_t datatype, int peer,
                         sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*recv)(void *recvbuff, size_t count,
                         sdcclDataType_t datatype, int peer,
                         sdcclInnerComm_t comm, sdcclStream_t stream);

  // Group semantics
  sdcclResult_t (*groupStart)();
  sdcclResult_t (*groupEnd)();

  // Device API - Host-side management (NCCL > 2.28, CNCL device API, etc.)
  sdcclResult_t (*devCommCreate)(sdcclInnerComm_t comm,
                                  const sdcclDevCommRequirements *reqs,
                                  sdcclInnerDevComm_t *devComm);
  sdcclResult_t (*devCommDestroy)(sdcclInnerComm_t comm,
                                   sdcclInnerDevComm_t devComm);
};
#define sdcclCCLAdaptor sdcclCCLAdaptor_v1

// Versioned export symbol name
#define SDCCL_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 sdcclCCLAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // SDCCL_CCL_ADAPTOR_H_
