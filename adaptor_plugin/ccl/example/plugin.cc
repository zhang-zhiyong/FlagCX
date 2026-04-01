/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example CCL adaptor plugin for SDCCL.
 * This is a minimal skeleton: all operations return sdcclInternalError,
 * so this plugin is only useful for verifying that the loading mechanism
 * works. A real plugin would wrap a CCL library (e.g. NCCL, RCCL).
 ************************************************************************/

#include "sdccl/sdccl_ccl_adaptor.h"
#include "sdccl/nvidia_adaptor.h"

static sdcclResult_t pluginGetVersion(int *version) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetUniqueId(sdcclUniqueId_t *uniqueId) {
  return sdcclInternalError;
}

static const char *pluginGetErrorString(sdcclResult_t result) {
  return "Example CCL plugin: not implemented";
}

static const char *pluginGetLastError(sdcclInnerComm_t comm) {
  return "Example CCL plugin: not implemented";
}

static sdcclResult_t pluginGetStagedBuffer(const sdcclInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                         sdcclUniqueId *commId, int rank,
                                         struct bootstrapState *bootstrap) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommFinalize(sdcclInnerComm_t comm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommDestroy(sdcclInnerComm_t comm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommAbort(sdcclInnerComm_t comm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommResume(sdcclInnerComm_t comm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommSuspend(sdcclInnerComm_t comm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommCount(const sdcclInnerComm_t comm,
                                      int *count) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommGetDeviceNumber(const sdcclInnerComm_t comm,
                                                int *device) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommUserRank(const sdcclInnerComm_t comm,
                                         int *rank) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommGetAsyncError(sdcclInnerComm_t comm,
                                              sdcclResult_t *asyncError) {
  return sdcclInternalError;
}

static sdcclResult_t pluginMemAlloc(void **ptr, size_t size) {
  return sdcclInternalError;
}

static sdcclResult_t pluginMemFree(void *ptr) { return sdcclInternalError; }

static sdcclResult_t pluginCommRegister(const sdcclInnerComm_t comm,
                                         void *buff, size_t size,
                                         void **handle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommDeregister(const sdcclInnerComm_t comm,
                                           void *handle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommWindowRegister(sdcclInnerComm_t comm,
                                               void *buff, size_t size,
                                               sdcclWindow_t *win,
                                               int winFlags) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCommWindowDeregister(sdcclInnerComm_t comm,
                                                 sdcclWindow_t win) {
  return sdcclInternalError;
}

static sdcclResult_t pluginReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, int root,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGather(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginScatter(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      int root, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclRedOp_t op, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t
pluginReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                    sdcclDataType_t datatype, sdcclRedOp_t op,
                    sdcclInnerComm_t comm, sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      sdcclDataType_t datatype,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginAlltoAll(const void *sendbuff, void *recvbuff,
                                     size_t count, sdcclDataType_t datatype,
                                     sdcclInnerComm_t comm,
                                     sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                      size_t *sdispls, void *recvbuff,
                                      size_t *recvcounts, size_t *rdispls,
                                      sdcclDataType_t datatype,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginSend(const void *sendbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginRecv(void *recvbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGroupStart() { return sdcclInternalError; }

static sdcclResult_t pluginGroupEnd() { return sdcclInternalError; }

__attribute__((visibility("default"))) struct sdcclCCLAdaptor_v1
    SDCCL_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",
        pluginGetVersion,
        pluginGetUniqueId,
        pluginGetErrorString,
        pluginGetLastError,
        pluginGetStagedBuffer,
        pluginCommInitRank,
        pluginCommFinalize,
        pluginCommDestroy,
        pluginCommAbort,
        pluginCommResume,
        pluginCommSuspend,
        pluginCommCount,
        pluginCommGetDeviceNumber,
        pluginCommUserRank,
        pluginCommGetAsyncError,
        pluginMemAlloc,
        pluginMemFree,
        pluginCommRegister,
        pluginCommDeregister,
        pluginCommWindowRegister,
        pluginCommWindowDeregister,
        pluginReduce,
        pluginGather,
        pluginScatter,
        pluginBroadcast,
        pluginAllReduce,
        pluginReduceScatter,
        pluginAllGather,
        pluginAlltoAll,
        pluginAlltoAllv,
        pluginSend,
        pluginRecv,
        pluginGroupStart,
        pluginGroupEnd,
};
