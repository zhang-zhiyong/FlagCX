#ifdef USE_CAMBRICON_ADAPTOR

#include "cambricon_adaptor.h"

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include <map>

std::map<sdcclDataType_t, cnclDataType_t> f2c_datatype_map = {
    {sdcclInt8, cnclInt8},       {sdcclUint8, cnclUint8},
    {sdcclInt, cnclInt},         {sdcclInt32, cnclInt32},
    {sdcclInt64, cnclInt64},     {sdcclHalf, cnclHalf},
    {sdcclFloat16, cnclFloat16}, {sdcclBfloat16, cnclBfloat16},
    {sdcclFloat32, cnclFloat32}, {sdcclFloat, cnclFloat},
    {sdcclDouble, cnclFloat},
};

std::map<sdcclRedOp_t, cnclReduceOp_t> f2c_reduceop_map = {
    {sdcclSum, cnclSum},
    {sdcclProd, cnclProd},
    {sdcclMax, cnclMax},
    {sdcclMin, cnclMin}};

// TODO: not match fully
std::map<cnclResult_t, sdcclResult_t> c2f_ret_map = {
    {CNCL_RET_SUCCESS, sdcclSuccess},
    {CNCL_RET_ERR_UNSUPPORTED, sdcclUnhandledDeviceError},
    {CNCL_RET_ASYNC_ERROR, sdcclRemoteError}};

std::map<sdcclResult_t, cnclResult_t> f2c_ret_map = {
    {sdcclSuccess, CNCL_RET_SUCCESS},
    {sdcclUnhandledDeviceError, CNCL_RET_ERR_UNSUPPORTED}};

// TODO: unsupported
sdcclResult_t cnclAdaptorGetVersion(int *version) {
  // return (sdcclResult_t)cnclGetVersion(version);
  return sdcclUnhandledDeviceError;
}

sdcclResult_t cnclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (
      sdcclResult_t)c2f_ret_map[cnclGetCliqueId((cnclCliqueId *)(*uniqueId))];
}

sdcclResult_t cnclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

const char *cnclAdaptorGetErrorString(sdcclResult_t result) {
  return cnclGetErrorStr((cnclResult_t)f2c_ret_map[result]);
}

// TODO: unsupported
const char *cnclAdaptorGetLastError(sdcclInnerComm_t comm) {
  // return cnclGetLastError(comm->base);
  return "Not Implemented";
}

sdcclResult_t cnclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  int devId = 0;
  DEVCHECK(cnrtGetDevice(&devId));
  return (sdcclResult_t)c2f_ret_map[cnclInitComms(
      &(*comm)->base, 1 /*num_comm*/, &devId /*dev_list*/, &rank /*rank_list*/,
      nranks, (cnclCliqueId *)commId)];
}

// TODO: unsupported
sdcclResult_t cnclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  // return (sdcclResult_t)cnclCommFinalize(comm->base);
  return sdcclUnhandledDeviceError;
}

sdcclResult_t cnclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)
      c2f_ret_map[cnclDestroyComms(&(comm->base), 1 /*num_comm*/)];
}

sdcclResult_t cnclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)c2f_ret_map[cnclAbortComm(comm->base)];
}

// TODO: not match
sdcclResult_t cnclAdaptorCommResume(sdcclInnerComm_t comm) {
  // return (sdcclResult_t)ncclInvalidUsage;
  return (sdcclResult_t)c2f_ret_map[CNCL_RET_ERR_ARGUMENTS];
}

// TODO: not match
sdcclResult_t cnclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  // return (sdcclResult_t)ncclInvalidUsage;
  return (sdcclResult_t)c2f_ret_map[CNCL_RET_ERR_ARGUMENTS];
}

sdcclResult_t cnclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)c2f_ret_map[cnclGetCommCount(count, comm->base)];
}

sdcclResult_t cnclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return (sdcclResult_t)c2f_ret_map[cnclGetCommDevice(device, comm->base)];
}

sdcclResult_t cnclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)c2f_ret_map[cnclGetCommRank(rank, comm->base)];
}

sdcclResult_t cnclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  *asyncError = c2f_ret_map[cnclGetCommAsyncError(comm->base)];
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t cnclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t cnclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t cnclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t cnclAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t cnclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t cnclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t cnclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclReduce(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], root, comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = cnclRecv(static_cast<void *>(buffer + r * size), size, cnclChar, r,
                     comm->base, stream->base);
    }
  }
  res = cnclSend(const_cast<void *>(sendbuff), size, cnclChar, root, comm->base,
                 stream->base);
  res = cnclGroupEnd();

  return (sdcclResult_t)c2f_ret_map[res];
}

sdcclResult_t cnclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = cnclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = cnclSend(
          const_cast<void *>(static_cast<const void *>(buffer + r * size)),
          size, cnclChar, r, comm->base, stream->base);
    }
  }
  res = cnclRecv(recvbuff, size, cnclChar, root, comm->base, stream->base);
  res = cnclGroupEnd();

  return (sdcclResult_t)c2f_ret_map[res];
}

sdcclResult_t cnclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclBroadcast(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      root, comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclAllReduce(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], comm->base, stream->base)];
}

sdcclResult_t
cnclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclReduceScatter(
      sendbuff, recvbuff, recvcount, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclAllGather(
      sendbuff, recvbuff, sendcount, (cnclDataType_t)f2c_datatype_map[datatype],
      comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = cnclSend(
        const_cast<void *>(static_cast<const void *>(bufferIn + r * size)),
        size, cnclChar, r, comm->base, stream->base);
    res = cnclRecv(static_cast<void *>(bufferOut + r * size), size, cnclChar, r,
                   comm->base, stream->base);
  }
  res = cnclGroupEnd();

  return (sdcclResult_t)c2f_ret_map[res];
}

sdcclResult_t cnclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  int nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (sdcclCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = cnclSend(const_cast<void *>(static_cast<const void *>(
                         bufferIn + sdispls[r] * size)),
                     sendcounts[r], f2c_datatype_map[datatype], r, comm->base,
                     stream->base);
    }
    if (sdcclCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = cnclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                     recvcounts[r], f2c_datatype_map[datatype], r, comm->base,
                     stream->base);
    }
  }
  res = cnclGroupEnd();

  return (sdcclResult_t)c2f_ret_map[res];
}

sdcclResult_t cnclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  // TODO: const_cast will be removed in the future
  return (sdcclResult_t)
      c2f_ret_map[cnclSend(const_cast<void *>(sendbuff), count,
                           (cnclDataType_t)f2c_datatype_map[datatype], peer,
                           comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)c2f_ret_map[cnclRecv(
      recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype], peer,
      comm->base, stream->base)];
}

sdcclResult_t cnclAdaptorGroupStart() {
  return (sdcclResult_t)c2f_ret_map[cnclGroupStart()];
}

sdcclResult_t cnclAdaptorGroupEnd() {
  return (sdcclResult_t)c2f_ret_map[cnclGroupEnd()];
}

sdcclResult_t
cnclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t cnclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor cnclAdaptor = {
    "CNCL",
    // Basic functions
    cnclAdaptorGetVersion, cnclAdaptorGetUniqueId, cnclAdaptorGetErrorString,
    cnclAdaptorGetLastError, cnclAdaptorGetStagedBuffer,
    // Communicator functions
    cnclAdaptorCommInitRank, cnclAdaptorCommFinalize, cnclAdaptorCommDestroy,
    cnclAdaptorCommAbort, cnclAdaptorCommResume, cnclAdaptorCommSuspend,
    cnclAdaptorCommCount, cnclAdaptorCommCuDevice, cnclAdaptorCommUserRank,
    cnclAdaptorCommGetAsyncError, cnclAdaptorMemAlloc, cnclAdaptorMemFree,
    cnclAdaptorCommRegister, cnclAdaptorCommDeregister,
    // Symmetric functions
    cnclAdaptorCommWindowRegister, cnclAdaptorCommWindowDeregister,
    // Communication functions
    cnclAdaptorReduce, cnclAdaptorGather, cnclAdaptorScatter,
    cnclAdaptorBroadcast, cnclAdaptorAllReduce, cnclAdaptorReduceScatter,
    cnclAdaptorAllGather, cnclAdaptorAlltoAll, cnclAdaptorAlltoAllv,
    cnclAdaptorSend, cnclAdaptorRecv,
    // Group semantics
    cnclAdaptorGroupStart, cnclAdaptorGroupEnd,
    // Device API
    cnclAdaptorDevCommCreate, cnclAdaptorDevCommDestroy};

#endif // USE_CAMBRICON_ADAPTOR
