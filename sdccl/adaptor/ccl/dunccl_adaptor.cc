#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

sdcclResult_t duncclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)ncclGetVersion(version);
}

sdcclResult_t duncclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

sdcclResult_t duncclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return sdcclNotSupported;
}

const char *duncclAdaptorGetErrorString(sdcclResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *duncclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

sdcclResult_t duncclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                         sdcclUniqueId_t commId, int rank,
                                         bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

sdcclResult_t duncclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommFinalize(comm->base);
}

sdcclResult_t duncclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommDestroy(comm->base);
}

sdcclResult_t duncclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommAbort(comm->base);
}

sdcclResult_t duncclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t duncclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t duncclAdaptorCommCount(const sdcclInnerComm_t comm,
                                      int *count) {
  return (sdcclResult_t)ncclCommCount(comm->base, count);
}

sdcclResult_t duncclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                         int *device) {
  return (sdcclResult_t)ncclCommCuDevice(comm->base, device);
}

sdcclResult_t duncclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                         int *rank) {
  return (sdcclResult_t)ncclCommUserRank(comm->base, rank);
}

sdcclResult_t duncclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                              sdcclResult_t *asyncError) {
  return (sdcclResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
sdcclResult_t duncclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t duncclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t duncclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                         size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t duncclAdaptorCommDeregister(sdcclInnerComm_t comm,
                                           void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t duncclAdaptorCommWindowRegister(sdcclInnerComm_t comm,
                                               void *buff, size_t size,
                                               sdcclWindow_t *win,
                                               int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t duncclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                                 sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t duncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, int root,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t duncclAdaptorGather(const void *sendbuff, void *recvbuff,
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

sdcclResult_t duncclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

sdcclResult_t duncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      int root, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t duncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclRedOp_t op, sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t duncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          sdcclDataType_t datatype,
                                          sdcclRedOp_t op,
                                          sdcclInnerComm_t comm,
                                          sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t duncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      sdcclDataType_t datatype,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t duncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                     size_t count, sdcclDataType_t datatype,
                                     sdcclInnerComm_t comm,
                                     sdcclStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
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

sdcclResult_t duncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
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

sdcclResult_t duncclAdaptorSend(const void *sendbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t duncclAdaptorRecv(void *recvbuff, size_t count,
                                 sdcclDataType_t datatype, int peer,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t duncclAdaptorGroupStart() {
  return (sdcclResult_t)ncclGroupStart();
}

sdcclResult_t duncclAdaptorGroupEnd() {
  return (sdcclResult_t)ncclGroupEnd();
}

sdcclResult_t
duncclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                           const sdcclDevCommRequirements * /*reqs*/,
                           sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t duncclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                           sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor duncclAdaptor = {
    "DUNCCL",
    // Basic functions
    duncclAdaptorGetVersion, duncclAdaptorGetUniqueId,
    duncclAdaptorGetErrorString, duncclAdaptorGetLastError,
    duncclAdaptorGetStagedBuffer,
    // Communicator functions
    duncclAdaptorCommInitRank, duncclAdaptorCommFinalize,
    duncclAdaptorCommDestroy, duncclAdaptorCommAbort, duncclAdaptorCommResume,
    duncclAdaptorCommSuspend, duncclAdaptorCommCount, duncclAdaptorCommCuDevice,
    duncclAdaptorCommUserRank, duncclAdaptorCommGetAsyncError,
    duncclAdaptorMemAlloc, duncclAdaptorMemFree, duncclAdaptorCommRegister,
    duncclAdaptorCommDeregister,
    // Symmetric functions
    duncclAdaptorCommWindowRegister, duncclAdaptorCommWindowDeregister,
    // Communication functions
    duncclAdaptorReduce, duncclAdaptorGather, duncclAdaptorScatter,
    duncclAdaptorBroadcast, duncclAdaptorAllReduce, duncclAdaptorReduceScatter,
    duncclAdaptorAllGather, duncclAdaptorAlltoAll, duncclAdaptorAlltoAllv,
    duncclAdaptorSend, duncclAdaptorRecv,
    // Group semantics
    duncclAdaptorGroupStart, duncclAdaptorGroupEnd,
    // Device API
    duncclAdaptorDevCommCreate, duncclAdaptorDevCommDestroy};

#endif // USE_DU_ADAPTOR
