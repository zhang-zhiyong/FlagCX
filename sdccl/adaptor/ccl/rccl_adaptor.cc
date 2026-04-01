#include "amd_adaptor.h"

#ifdef USE_AMD_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

sdcclResult_t rcclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)ncclGetVersion(version);
}

sdcclResult_t rcclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

sdcclResult_t rcclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

const char *rcclAdaptorGetErrorString(sdcclResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *rcclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

sdcclResult_t rcclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

sdcclResult_t rcclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommFinalize(comm->base);
}

sdcclResult_t rcclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommDestroy(comm->base);
}

sdcclResult_t rcclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclCommAbort(comm->base);
}

sdcclResult_t rcclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t rcclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t rcclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)ncclCommCount(comm->base, count);
}

sdcclResult_t rcclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return (sdcclResult_t)ncclCommCuDevice(comm->base, device);
}

sdcclResult_t rcclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)ncclCommUserRank(comm->base, rank);
}

sdcclResult_t rcclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return (sdcclResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
sdcclResult_t rcclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t rcclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t rcclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t rcclAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t rcclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t rcclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t rcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t rcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  int rank, nranks;
  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommUserRank(comm->base, &rank), res, fail);
  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      CCLCHECKGOTO(ncclRecv(static_cast<void *>(buffer + r * size), size,
                            ncclChar, r, comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(
      ncclSend(sendbuff, size, ncclChar, root, comm->base, stream->base), res,
      fail);
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return sdcclSuccess;
fail:
  return (sdcclResult_t)res;
}

sdcclResult_t rcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  int rank, nranks;
  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommUserRank(comm->base, &rank), res, fail);
  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      CCLCHECKGOTO(ncclSend(static_cast<const void *>(buffer + r * size), size,
                            ncclChar, r, comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(
      ncclRecv(recvbuff, size, ncclChar, root, comm->base, stream->base), res,
      fail);
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return sdcclSuccess;
fail:
  return (sdcclResult_t)res;
}

sdcclResult_t rcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t rcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t
rcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t rcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t rcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  int nranks;
  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  for (int r = 0; r < nranks; r++) {
    CCLCHECKGOTO(ncclSend(static_cast<const void *>(bufferIn + r * size), size,
                          ncclChar, r, comm->base, stream->base),
                 res, fail);
    CCLCHECKGOTO(ncclRecv(static_cast<void *>(bufferOut + r * size), size,
                          ncclChar, r, comm->base, stream->base),
                 res, fail);
  }
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return sdcclSuccess;
fail:
  return (sdcclResult_t)res;
}

sdcclResult_t rcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  int nranks;
  size_t size = getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  for (int r = 0; r < nranks; r++) {
    if (sdcclCCLAdaptorNeedSendrecv(sendcounts[r])) {
      CCLCHECKGOTO(
          ncclSend(static_cast<const void *>(bufferIn + sdispls[r] * size),
                   sendcounts[r], (ncclDataType_t)datatype, r, comm->base,
                   stream->base),
          res, fail);
    }
    if (sdcclCCLAdaptorNeedSendrecv(recvcounts[r])) {
      CCLCHECKGOTO(ncclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                            recvcounts[r], (ncclDataType_t)datatype, r,
                            comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return sdcclSuccess;
fail:
  return (sdcclResult_t)res;
}

sdcclResult_t rcclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t rcclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t rcclAdaptorGroupStart() {
  return (sdcclResult_t)ncclGroupStart();
}

sdcclResult_t rcclAdaptorGroupEnd() { return (sdcclResult_t)ncclGroupEnd(); }

sdcclResult_t
rcclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t rcclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor rcclAdaptor = {
    "RCCL",
    // Basic functions
    rcclAdaptorGetVersion, rcclAdaptorGetUniqueId, rcclAdaptorGetErrorString,
    rcclAdaptorGetLastError, rcclAdaptorGetStagedBuffer,
    // Communicator functions
    rcclAdaptorCommInitRank, rcclAdaptorCommFinalize, rcclAdaptorCommDestroy,
    rcclAdaptorCommAbort, rcclAdaptorCommResume, rcclAdaptorCommSuspend,
    rcclAdaptorCommCount, rcclAdaptorCommCuDevice, rcclAdaptorCommUserRank,
    rcclAdaptorCommGetAsyncError, rcclAdaptorMemAlloc, rcclAdaptorMemFree,
    rcclAdaptorCommRegister, rcclAdaptorCommDeregister,
    // Symmetric functions
    rcclAdaptorCommWindowRegister, rcclAdaptorCommWindowDeregister,
    // Communication functions
    rcclAdaptorReduce, rcclAdaptorGather, rcclAdaptorScatter,
    rcclAdaptorBroadcast, rcclAdaptorAllReduce, rcclAdaptorReduceScatter,
    rcclAdaptorAllGather, rcclAdaptorAlltoAll, rcclAdaptorAlltoAllv,
    rcclAdaptorSend, rcclAdaptorRecv,
    // Group semantics
    rcclAdaptorGroupStart, rcclAdaptorGroupEnd,
    // Device API
    rcclAdaptorDevCommCreate, rcclAdaptorDevCommDestroy};

#endif // USE_AMD_ADAPTOR