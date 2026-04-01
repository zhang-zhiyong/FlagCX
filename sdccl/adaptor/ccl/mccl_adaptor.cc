/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

sdcclResult_t mcclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)mcclGetVersion(version);
}

sdcclResult_t mcclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)mcclGetUniqueId((mcclUniqueId *)(*uniqueId));
}

sdcclResult_t mcclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

const char *mcclAdaptorGetErrorString(sdcclResult_t result) {
  return mcclGetErrorString((mcclResult_t)result);
}

const char *mcclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return mcclGetLastError(comm->base);
}

sdcclResult_t mcclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)mcclCommInitRank(&(*comm)->base, nranks,
                                          *(mcclUniqueId *)commId, rank);
}

sdcclResult_t mcclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return (sdcclResult_t)mcclCommFinalize(comm->base);
}

sdcclResult_t mcclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)mcclCommDestroy(comm->base);
}

sdcclResult_t mcclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)mcclCommAbort(comm->base);
}

sdcclResult_t mcclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)mcclInvalidUsage;
}

sdcclResult_t mcclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)mcclInvalidUsage;
}

sdcclResult_t mcclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)mcclCommCount(comm->base, count);
}

sdcclResult_t mcclAdaptorCommMcDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return (sdcclResult_t)mcclCommMcDevice(comm->base, device);
}

sdcclResult_t mcclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)mcclCommUserRank(comm->base, rank);
}

sdcclResult_t mcclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return (sdcclResult_t)mcclCommGetAsyncError(comm->base,
                                               (mcclResult_t *)asyncError);
}

// TODO: unsupported
sdcclResult_t mcclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t mcclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t mcclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t mcclAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t mcclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t mcclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t mcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)mcclReduce(sendbuff, recvbuff, count,
                                    (mcclDataType_t)datatype, (mcclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t mcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = mcclRecv(static_cast<void *>(buffer + r * size), size, mcclChar, r,
                     comm->base, stream->base);
    }
  }
  res = mcclSend(sendbuff, size, mcclChar, root, comm->base, stream->base);
  res = mcclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t mcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = mcclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = mcclSend(static_cast<const void *>(buffer + r * size), size,
                     mcclChar, r, comm->base, stream->base);
    }
  }
  res = mcclRecv(recvbuff, size, mcclChar, root, comm->base, stream->base);
  res = mcclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t mcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)mcclBroadcast(sendbuff, recvbuff, count,
                                       (mcclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t mcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)mcclAllReduce(
      sendbuff, recvbuff, count, (mcclDataType_t)datatype, (mcclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t
mcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)mcclReduceScatter(
      sendbuff, recvbuff, recvcount, (mcclDataType_t)datatype, (mcclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t mcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)mcclAllGather(sendbuff, recvbuff, sendcount,
                                       (mcclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t mcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = mcclSend(static_cast<const void *>(bufferIn + r * size), size,
                   mcclChar, r, comm->base, stream->base);
    res = mcclRecv(static_cast<void *>(bufferOut + r * size), size, mcclChar, r,
                   comm->base, stream->base);
  }
  res = mcclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t mcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  int nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommCount(comm->base, &nranks);

  size_t size = getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (sdcclCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = mcclSend(static_cast<const void *>(bufferIn + sdispls[r] * size),
                     sendcounts[r], (mcclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
    if (sdcclCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = mcclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                     recvcounts[r], (mcclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
  }
  res = mcclGroupEnd();

  return (sdcclResult_t)res;
}

sdcclResult_t mcclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)mcclSend(sendbuff, count, (mcclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t mcclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)mcclRecv(recvbuff, count, (mcclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t mcclAdaptorGroupStart() {
  return (sdcclResult_t)mcclGroupStart();
}

sdcclResult_t mcclAdaptorGroupEnd() { return (sdcclResult_t)mcclGroupEnd(); }

sdcclResult_t
mcclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t mcclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor mcclAdaptor = {
    "MCCL",
    // Basic functions
    mcclAdaptorGetVersion, mcclAdaptorGetUniqueId, mcclAdaptorGetErrorString,
    mcclAdaptorGetLastError, mcclAdaptorGetStagedBuffer,
    // Communicator functions
    mcclAdaptorCommInitRank, mcclAdaptorCommFinalize, mcclAdaptorCommDestroy,
    mcclAdaptorCommAbort, mcclAdaptorCommResume, mcclAdaptorCommSuspend,
    mcclAdaptorCommCount, mcclAdaptorCommMcDevice, mcclAdaptorCommUserRank,
    mcclAdaptorCommGetAsyncError, mcclAdaptorMemAlloc, mcclAdaptorMemFree,
    mcclAdaptorCommRegister, mcclAdaptorCommDeregister,
    // Symmetric functions
    mcclAdaptorCommWindowRegister, mcclAdaptorCommWindowDeregister,
    // Communication functions
    mcclAdaptorReduce, mcclAdaptorGather, mcclAdaptorScatter,
    mcclAdaptorBroadcast, mcclAdaptorAllReduce, mcclAdaptorReduceScatter,
    mcclAdaptorAllGather, mcclAdaptorAlltoAll, mcclAdaptorAlltoAllv,
    mcclAdaptorSend, mcclAdaptorRecv,
    // Group semantics
    mcclAdaptorGroupStart, mcclAdaptorGroupEnd,
    // Device API
    mcclAdaptorDevCommCreate, mcclAdaptorDevCommDestroy};

#endif // USE_METAX_ADAPTOR
