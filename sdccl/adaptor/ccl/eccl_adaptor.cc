/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#include "enflame_adaptor.h"

#ifdef USE_ENFLAME_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

sdcclResult_t ecclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)ecclGetVersion(version);
}

sdcclResult_t ecclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)ecclGetUniqueId((ecclUniqueId *)(*uniqueId));
}

const char *ecclAdaptorGetErrorString(sdcclResult_t result) {
  return ecclGetErrorString((ecclResult_t)result);
}

const char *ecclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return ecclGetLastError(comm->base);
}

sdcclResult_t ecclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)ecclCommInitRank(&(*comm)->base, nranks,
                                          *(ecclUniqueId *)commId, rank);
}

sdcclResult_t ecclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  // ECCL does not have a separate finalize function, use destroy
  return sdcclSuccess;
}

sdcclResult_t ecclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ecclCommDestroy(comm->base);
}

sdcclResult_t ecclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ecclCommAbort(comm->base);
}

sdcclResult_t ecclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ecclInvalidUsage;
}

sdcclResult_t ecclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ecclInvalidUsage;
}

sdcclResult_t ecclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)ecclCommCount(comm->base, count);
}

sdcclResult_t ecclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return (sdcclResult_t)ecclCommDevice(comm->base, device);
}

sdcclResult_t ecclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)ecclCommUserRank(comm->base, rank);
}

sdcclResult_t ecclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return (sdcclResult_t)ecclCommGetAsyncError(comm->base,
                                               (ecclResult_t *)asyncError);
}

sdcclResult_t ecclAdaptorMemAlloc(void **ptr, size_t size) {
  topsError_t err = topsMalloc(ptr, size);
  if (err != topsSuccess) {
    return sdcclUnhandledDeviceError;
  }
  return sdcclSuccess;
}

sdcclResult_t ecclAdaptorMemFree(void *ptr) {
  topsError_t err = topsFree(ptr);
  if (err != topsSuccess) {
    return sdcclUnhandledDeviceError;
  }
  return sdcclSuccess;
}

sdcclResult_t ecclAdaptorCommRegister(const sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorCommDeregister(const sdcclInnerComm_t comm,
                                         void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ecclReduce(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, (ecclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t ecclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ecclGather(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, root, comm->base,
                                    stream->base);
}

sdcclResult_t ecclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  return (sdcclResult_t)ecclScatter(sendbuff, recvbuff, count,
                                     (ecclDataType_t)datatype, root, comm->base,
                                     stream->base);
}

sdcclResult_t ecclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ecclBroadcast(sendbuff, recvbuff, count,
                                       (ecclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t ecclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ecclAllReduce(
      sendbuff, recvbuff, count, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t
ecclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ecclReduceScatter(
      sendbuff, recvbuff, recvcount, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t ecclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ecclAllGather(sendbuff, recvbuff, sendcount,
                                       (ecclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t ecclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return (sdcclResult_t)ecclAlltoall(sendbuff, count, (ecclDataType_t)datatype,
                                      recvbuff, count, (ecclDataType_t)datatype,
                                      comm->base, stream->base);
}

sdcclResult_t ecclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ecclAlltoAllv(
      (void *)sendbuff, sendcounts, sdispls, (ecclDataType_t)datatype, recvbuff,
      recvcounts, rdispls, (ecclDataType_t)datatype, comm->base, stream->base);
}

sdcclResult_t ecclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ecclSend(sendbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ecclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ecclRecv(recvbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ecclAdaptorGroupStart() {
  return (sdcclResult_t)ecclGroupStart();
}

sdcclResult_t ecclAdaptorGroupEnd() { return (sdcclResult_t)ecclGroupEnd(); }

sdcclResult_t
ecclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t ecclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor ecclAdaptor = {
    "ECCL",
    // Basic functions
    ecclAdaptorGetVersion, ecclAdaptorGetUniqueId, ecclAdaptorGetErrorString,
    ecclAdaptorGetLastError, ecclAdaptorGetStagedBuffer,
    // Communicator functions
    ecclAdaptorCommInitRank, ecclAdaptorCommFinalize, ecclAdaptorCommDestroy,
    ecclAdaptorCommAbort, ecclAdaptorCommResume, ecclAdaptorCommSuspend,
    ecclAdaptorCommCount, ecclAdaptorCommCuDevice, ecclAdaptorCommUserRank,
    ecclAdaptorCommGetAsyncError, ecclAdaptorMemAlloc, ecclAdaptorMemFree,
    ecclAdaptorCommRegister, ecclAdaptorCommDeregister,
    // Symmetric functions
    ecclAdaptorCommWindowRegister, ecclAdaptorCommWindowDeregister,
    // Communication functions
    ecclAdaptorReduce, ecclAdaptorGather, ecclAdaptorScatter,
    ecclAdaptorBroadcast, ecclAdaptorAllReduce, ecclAdaptorReduceScatter,
    ecclAdaptorAllGather, ecclAdaptorAlltoAll, ecclAdaptorAlltoAllv,
    ecclAdaptorSend, ecclAdaptorRecv,
    // Group semantics
    ecclAdaptorGroupStart, ecclAdaptorGroupEnd,
    // Device API
    ecclAdaptorDevCommCreate, ecclAdaptorDevCommDestroy};

#endif // USE_ENFLAME_ADAPTOR
