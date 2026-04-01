#include "iluvatar_corex_adaptor.h"

#ifdef USE_ILUVATAR_COREX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

sdcclResult_t ixncclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)ncclGetVersion(version);
}

sdcclResult_t ixncclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

sdcclResult_t ixncclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return sdcclNotSupported;
}

const char *ixncclAdaptorGetErrorString(sdcclResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ixncclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

sdcclResult_t ixncclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                         sdcclUniqueId_t commId, int rank,
                                         bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

sdcclResult_t ixncclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommFinalize(comm->base);
}

sdcclResult_t ixncclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommDestroy(comm->base);
}

sdcclResult_t ixncclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommAbort(comm->base);
}

sdcclResult_t ixncclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t ixncclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t ixncclAdaptorCommCount(const sdcclInnerComm_t comm,
                                      int *count) {
  return (sdcclResult_t)ncclCommCount(comm->base, count);
}

sdcclResult_t ixncclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                         int *device) {
  return (sdcclResult_t)ncclCommCuDevice(comm->base, device);
}

sdcclResult_t ixncclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                         int *rank) {
  return (sdcclResult_t)ncclCommUserRank(comm->base, rank);
}

sdcclResult_t ixncclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                              sdcclResult_t *asyncError) {
  return (sdcclResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
sdcclResult_t ixncclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t ixncclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t ixncclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                         size_t size, void **handle) {
  return sdcclNotSupported;
}

sdcclResult_t ixncclAdaptorCommWindowRegister(sdcclInnerComm_t comm,
                                               void *buff, size_t size,
                                               sdcclWindow_t *win,
                                               int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t ixncclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                                 sdcclWindow_t win) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t ixncclAdaptorCommDeregister(sdcclInnerComm_t comm,
                                           void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t ixncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, int root,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorGather(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclRecv(static_cast<void *>(buffer + r * size), size, ncclChar, r,
                     comm->base, stream->base);
    }
  }
  res = ncclSend(sendbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t ixncclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclSend(static_cast<const void *>(buffer + r * size), size,
                     ncclChar, r, comm->base, stream->base);
    }
  }
  res = ncclRecv(recvbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t ixncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      int root, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclRedOp_t op, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          sdcclDataType_t datatype,
                                          sdcclRedOp_t op,
                                          sdcclInnerComm_t comm,
                                          sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      sdcclDataType_t datatype,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t ixncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                     size_t count, sdcclDataType_t datatype,
                                     sdcclInnerComm_t comm,
                                     sdcclStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = ncclSend(static_cast<const void *>(bufferIn + r * size), size,
                   ncclChar, r, comm->base, stream->base);
    res = ncclRecv(static_cast<void *>(bufferOut + r * size), size, ncclChar, r,
                   comm->base, stream->base);
  }
  res = ncclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t ixncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                      size_t *sdispls, void *recvbuff,
                                      size_t *recvcounts, size_t *rdispls,
                                      sdcclDataType_t datatype,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommCount(comm->base, &nranks);

  size_t size = getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (sdcclCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = ncclSend(static_cast<const void *>(bufferIn + sdispls[r] * size),
                     sendcounts[r], (ncclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
    if (sdcclCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = ncclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                     recvcounts[r], (ncclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
  }
  res = ncclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t ixncclAdaptorSend(const void *sendbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorRecv(void *recvbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ixncclAdaptorGroupStart() {
  return (sdcclResult_t)ncclGroupStart();
}

sdcclResult_t ixncclAdaptorGroupEnd() {
  return (sdcclResult_t)ncclGroupEnd();
}

sdcclResult_t
ixncclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                           const sdcclDevCommRequirements * /*reqs*/,
                           sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t ixncclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                           sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor ixncclAdaptor = {
    "IXNCCL",
    // Basic functions
    ixncclAdaptorGetVersion, ixncclAdaptorGetUniqueId,
    ixncclAdaptorGetErrorString, ixncclAdaptorGetLastError,
    ixncclAdaptorGetStagedBuffer,
    // Communicator functions
    ixncclAdaptorCommInitRank, ixncclAdaptorCommFinalize,
    ixncclAdaptorCommDestroy, ixncclAdaptorCommAbort, ixncclAdaptorCommResume,
    ixncclAdaptorCommSuspend, ixncclAdaptorCommCount, ixncclAdaptorCommCuDevice,
    ixncclAdaptorCommUserRank, ixncclAdaptorCommGetAsyncError,
    ixncclAdaptorMemAlloc, ixncclAdaptorMemFree, ixncclAdaptorCommRegister,
    ixncclAdaptorCommDeregister,
    // Symmetric functions
    ixncclAdaptorCommWindowRegister, ixncclAdaptorCommWindowDeregister,
    // Communication functions
    ixncclAdaptorReduce, ixncclAdaptorGather, ixncclAdaptorScatter,
    ixncclAdaptorBroadcast, ixncclAdaptorAllReduce, ixncclAdaptorReduceScatter,
    ixncclAdaptorAllGather, ixncclAdaptorAlltoAll, ixncclAdaptorAlltoAllv,
    ixncclAdaptorSend, ixncclAdaptorRecv,
    // Group semantics
    ixncclAdaptorGroupStart, ixncclAdaptorGroupEnd,
    // Device API
    ixncclAdaptorDevCommCreate, ixncclAdaptorDevCommDestroy};

#endif // USE_ILUVATAR_COREX_ADAPTOR
