#include "tsmicro_adaptor.h"

#ifdef USE_TSM_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

#include <cstring>
#include <map>

static const std::map<tcclResult_t, sdcclResult_t> tcclToSdcclResultMap = {
    {tcclSuccess, sdcclSuccess},
    {tcclUnhandledDeviceError, sdcclUnhandledDeviceError},
    {tcclSystemError, sdcclSystemError},
    {tcclInvalidArgument, sdcclInvalidArgument},
    {tcclInvalidUsage, sdcclInvalidUsage},
    {tcclRemoteError, sdcclRemoteError},
    {tcclInProgress, sdcclInProgress},
    {tcclUnhandledCCLError, sdcclUnhandledCCLError},
    {tcclNotSupported, sdcclNotSupported},
    {tcclNumResults, sdcclNumResults},
    {tcclInternalError, sdcclInternalError}};

// Data type mapping
static const std::map<sdcclDataType_t, tcclDataType_t>
    sdcclToTcclDatatypeMap = {
        {sdcclInt8, tcclInt8},         {sdcclChar, tcclChar},
        {sdcclUint8, tcclUint8},       {sdcclInt32, tcclInt32},
        {sdcclInt, tcclInt},           {sdcclUint32, tcclUint32},
        {sdcclInt64, tcclInt64},       {sdcclUint64, tcclUint64},
        {sdcclFloat16, tcclFloat16},   {sdcclHalf, tcclHalf},
        {sdcclFloat32, tcclFloat32},   {sdcclFloat, tcclFloat},
        {sdcclFloat64, tcclFloat64},   {sdcclDouble, tcclDouble},
        {sdcclBfloat16, tcclBfloat16}, {sdcclNumTypes, tcclNumTypes}};

// Reduction operation mapping
static const std::map<sdcclRedOp_t, tcclRedOp_t> sdcclToTcclRedopMap = {
    {sdcclSum, tcclSum},           {sdcclProd, tcclProd},
    {sdcclMax, tcclMax},           {sdcclMin, tcclMin},
    {sdcclAvg, tcclAvg},           {sdcclNumRedOps, tcclNumRedOps},
    {sdcclMaxRedOp, tcclMaxRedOp}, {sdcclRedNoOp, tcclRedNoOp}};

// Type conversion functions using maps
static inline sdcclResult_t fromTcclResult(tcclResult_t result) {
  auto it = tcclToSdcclResultMap.find(result);
  if (it != tcclToSdcclResultMap.end()) {
    return it->second;
  }
  return sdcclInternalError; // Default error if not found
}

static inline tcclDataType_t toTcclDataType(sdcclDataType_t dtype) {
  auto it = sdcclToTcclDatatypeMap.find(dtype);
  if (it != sdcclToTcclDatatypeMap.end()) {
    return it->second;
  }
  return tcclNumTypes; // Default enum value if not found
}

static inline tcclRedOp_t toTcclRedOp(sdcclRedOp_t op) {
  auto it = sdcclToTcclRedopMap.find(op);
  if (it != sdcclToTcclRedopMap.end()) {
    return it->second;
  }
  return tcclRedNoOp; // Default enum value if not found
}

sdcclResult_t tcclAdaptorGetVersion(int *version) {
  tcclResult_t result = tcclGetVersion(version);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  tcclResult_t result = tcclGetUniqueId((tcclUniqueId *)(*uniqueId));
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

const char *tcclAdaptorGetErrorString(sdcclResult_t result) {
  // TODO: supported later
  return "Not Implemented";
}

const char *tcclAdaptorGetLastError(sdcclInnerComm_t comm) {
  if (!comm) {
    return "sdcclInvalidArgument";
  }
  return tcclGetLastError(comm->base);
}

sdcclResult_t tcclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  tcclResult_t result =
      tcclCommInitRank(&(*comm)->base, nranks, *(tcclUniqueId *)commId, rank);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommFinalize(comm->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommDestroy(comm->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommAbort(sdcclInnerComm_t comm) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommAbort(comm->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommResume(sdcclInnerComm_t comm) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommResume(comm->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommSuspend(comm->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommCount(comm->base, count);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommGetDeviceNumber(comm->base, device);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommUserRank(comm->base, rank);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t tccl_error;
  tcclResult_t result = tcclCommGetAsyncError(comm->base, &tccl_error);
  if (asyncError) {
    *asyncError = fromTcclResult(tccl_error);
  }
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

sdcclResult_t tcclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

sdcclResult_t tcclAdaptorCommRegister(const sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommRegister(comm->base, buff, size, handle);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommDeregister(const sdcclInnerComm_t comm,
                                         void *handle) {
  if (!comm) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclCommDeregister(comm->base, handle);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t tcclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t tcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclReduce(sendbuff, recvbuff, count, toTcclDataType(datatype),
                 toTcclRedOp(op), root, comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclGather(sendbuff, recvbuff, count, toTcclDataType(datatype), root,
                 comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclScatter(sendbuff, recvbuff, count, toTcclDataType(datatype), root,
                  comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclBroadcast(sendbuff, recvbuff, count, toTcclDataType(datatype), root,
                    comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclAllReduce(sendbuff, recvbuff, count, toTcclDataType(datatype),
                    toTcclRedOp(op), comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t
tcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclReduceScatter(sendbuff, recvbuff, recvcount, toTcclDataType(datatype),
                        toTcclRedOp(op), comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclAllGather(sendbuff, recvbuff, sendcount, toTcclDataType(datatype),
                    comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result =
      tcclAlltoAll(sendbuff, recvbuff, count, toTcclDataType(datatype),
                   comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclAlltoAllv(
      sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls,
      toTcclDataType(datatype), comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclSend(sendbuff, count, toTcclDataType(datatype),
                                 peer, comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  if (!comm || !stream) {
    return sdcclInvalidArgument;
  }
  tcclResult_t result = tcclRecv(recvbuff, count, toTcclDataType(datatype),
                                 peer, comm->base, stream->base);
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorGroupStart() {
  tcclResult_t result = tcclGroupStart();
  return fromTcclResult(result);
}

sdcclResult_t tcclAdaptorGroupEnd() {
  tcclResult_t result = tcclGroupEnd();
  return fromTcclResult(result);
}

sdcclResult_t
tcclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t tcclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor tcclAdaptor = {
    "TCCL",
    // Basic functions
    tcclAdaptorGetVersion, tcclAdaptorGetUniqueId, tcclAdaptorGetErrorString,
    tcclAdaptorGetLastError, tcclAdaptorGetStagedBuffer,
    // Communicator functions
    tcclAdaptorCommInitRank, tcclAdaptorCommFinalize, tcclAdaptorCommDestroy,
    tcclAdaptorCommAbort, tcclAdaptorCommResume, tcclAdaptorCommSuspend,
    tcclAdaptorCommCount, tcclAdaptorCommCuDevice, tcclAdaptorCommUserRank,
    tcclAdaptorCommGetAsyncError, tcclAdaptorMemAlloc, tcclAdaptorMemFree,
    tcclAdaptorCommRegister, tcclAdaptorCommDeregister,
    // Symmetric functions
    tcclAdaptorCommWindowRegister, tcclAdaptorCommWindowDeregister,
    // Communication functions
    tcclAdaptorReduce, tcclAdaptorGather, tcclAdaptorScatter,
    tcclAdaptorBroadcast, tcclAdaptorAllReduce, tcclAdaptorReduceScatter,
    tcclAdaptorAllGather, tcclAdaptorAlltoAll, tcclAdaptorAlltoAllv,
    tcclAdaptorSend, tcclAdaptorRecv,
    // Group semantics
    tcclAdaptorGroupStart, tcclAdaptorGroupEnd,
    // Device API
    tcclAdaptorDevCommCreate, tcclAdaptorDevCommDestroy};

#endif // USE_TSM_ADAPTOR