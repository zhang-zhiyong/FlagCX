#include "ascend_adaptor.h"

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include <map>
#include <vector>
std::map<sdcclDataType_t, HcclDataType> f2h_datatype_map = {
    {sdcclInt8, HCCL_DATA_TYPE_INT8},
    {sdcclUint8, HCCL_DATA_TYPE_UINT8},
    {sdcclInt, HCCL_DATA_TYPE_INT32},
    {sdcclInt32, HCCL_DATA_TYPE_INT32},
    {sdcclInt64, HCCL_DATA_TYPE_INT64},
    {sdcclHalf, HCCL_DATA_TYPE_FP16},
    {sdcclFloat16, HCCL_DATA_TYPE_FP16},
    {sdcclBfloat16, HCCL_DATA_TYPE_BFP16},
    {sdcclFloat32, HCCL_DATA_TYPE_FP32},
    {sdcclFloat, HCCL_DATA_TYPE_FP32},
    {sdcclDouble, HCCL_DATA_TYPE_FP64},
};

std::map<sdcclRedOp_t, HcclReduceOp> f2h_reduceop_map = {
    {sdcclSum, HCCL_REDUCE_SUM},
    {sdcclProd, HCCL_REDUCE_PROD},
    {sdcclMax, HCCL_REDUCE_MAX},
    {sdcclMin, HCCL_REDUCE_MIN}};

// TODO: not match fully
std::map<HcclResult, sdcclResult_t> h2f_ret_map = {
    {HCCL_SUCCESS, sdcclSuccess},
    {HCCL_E_PARA, sdcclInvalidArgument},
    {HCCL_E_PTR, sdcclUnhandledDeviceError},
    {HCCL_E_MEMORY, sdcclUnhandledDeviceError},
    {HCCL_E_INTERNAL, sdcclInternalError},
    {HCCL_E_NOT_SUPPORT, sdcclNotSupported},
    {HCCL_E_NOT_FOUND, sdcclUnhandledDeviceError},
    {HCCL_E_UNAVAIL, sdcclUnhandledDeviceError},
    {HCCL_E_SYSCALL, sdcclUnhandledDeviceError},
    {HCCL_E_TIMEOUT, sdcclUnhandledDeviceError},
    {HCCL_E_OPEN_FILE_FAILURE, sdcclUnhandledDeviceError},
    {HCCL_E_TCP_CONNECT, sdcclUnhandledDeviceError},
    {HCCL_E_ROCE_CONNECT, sdcclUnhandledDeviceError},
    {HCCL_E_TCP_TRANSFER, sdcclUnhandledDeviceError},
    {HCCL_E_ROCE_TRANSFER, sdcclUnhandledDeviceError},
    {HCCL_E_RUNTIME, sdcclUnhandledDeviceError},
    {HCCL_E_DRV, sdcclUnhandledDeviceError},
    {HCCL_E_PROFILING, sdcclUnhandledDeviceError},
    {HCCL_E_CCE, sdcclUnhandledDeviceError},
    {HCCL_E_NETWORK, sdcclUnhandledDeviceError},
    {HCCL_E_AGAIN, sdcclUnhandledDeviceError},
    {HCCL_E_REMOTE, sdcclRemoteError},
    {HCCL_E_SUSPENDING, sdcclUnhandledDeviceError},
    {HCCL_E_RESERVED, sdcclUnhandledDeviceError}};

std::map<sdcclResult_t, HcclResult> f2h_ret_map = {
    {sdcclSuccess, HCCL_SUCCESS},
    {sdcclInternalError, HCCL_E_INTERNAL},
    {sdcclNotSupported, HCCL_E_NOT_SUPPORT},
    {sdcclInvalidArgument, HCCL_E_PARA},
    {sdcclRemoteError, HCCL_E_REMOTE},
    {sdcclUnhandledDeviceError, HCCL_E_RESERVED}};

struct HcclSendRecvItemEx {
  sdcclInnerComm_t comm;
  sdcclStream_t stream;
};
HcclSendRecvItemEx item;
std::vector<HcclSendRecvItem> sendRecvInfo;

// TODO: unsupported
sdcclResult_t hcclAdaptorGetVersion(int *version) {
  return sdcclNotSupported;
}

sdcclResult_t hcclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  return (
      sdcclResult_t)h2f_ret_map[HcclGetRootInfo((HcclRootInfo *)(*uniqueId))];
}

sdcclResult_t hcclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

const char *hcclAdaptorGetErrorString(sdcclResult_t result) {
  return HcclGetErrorString((HcclResult)f2h_ret_map[result]);
}

// TODO: unsupported
const char *hcclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return "Not Implemented";
}

sdcclResult_t hcclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)h2f_ret_map[HcclCommInitRootInfo(
      nranks, (HcclRootInfo *)commId, rank, &(*comm)->base)];
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return sdcclUnhandledDeviceError;
}

sdcclResult_t hcclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)h2f_ret_map[HcclCommDestroy(comm->base)];
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommResume(sdcclInnerComm_t comm) {
  return sdcclUnhandledDeviceError;
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return sdcclUnhandledDeviceError;
}

sdcclResult_t hcclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)
      h2f_ret_map[HcclGetRankSize(comm->base, (uint32_t *)count)];
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return sdcclUnhandledDeviceError;
}

sdcclResult_t hcclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (
      sdcclResult_t)h2f_ret_map[HcclGetRankId(comm->base, (uint32_t *)rank)];
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t hcclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t hcclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t hcclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t hcclAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t hcclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t hcclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t hcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclReduce(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], root, comm->base, stream->base)];
}

// TODO: unsupported
sdcclResult_t hcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  uint32_t rank, nranks;
  HcclResult res = HCCL_SUCCESS;
  res = HcclGetRankSize(comm->base, &nranks);
  res = HcclGetRankId(comm->base, &rank);
  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);
  std::vector<HcclSendRecvItem> sendRecvInfo;
  if (rank == root) {
    for (uint32_t r = 0; r < nranks; r++) {
      sendRecvInfo.emplace_back(HcclSendRecvItem{
          HcclSendRecvType::HCCL_RECV, static_cast<void *>(buffer + r * size),
          size, HcclDataType::HCCL_DATA_TYPE_INT8, r});
    }
  }
  void *sendbuffptr = (void *)sendbuff;
  sendRecvInfo.emplace_back(
      HcclSendRecvItem{HcclSendRecvType::HCCL_SEND, sendbuffptr, size,
                       HcclDataType::HCCL_DATA_TYPE_INT8, (uint32_t)root});
  uint32_t itemNum = sendRecvInfo.size();
  HcclBatchSendRecv(sendRecvInfo.data(), itemNum, comm->base, stream->base);

  return (sdcclResult_t)h2f_ret_map[res];
}

sdcclResult_t hcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclScatter(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      root, comm->base, stream->base)];
}

sdcclResult_t hcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {

  uint32_t rank;
  HcclGetRankId(comm->base, &rank);
  if (rank == root) {
    aclrtMemcpy(recvbuff, count, sendbuff, count, ACL_MEMCPY_DEVICE_TO_DEVICE);
  }
  void *buffer = (rank == root) ? const_cast<void *>(sendbuff) : recvbuff;
  return (sdcclResult_t)h2f_ret_map[HcclBroadcast(
      buffer, count, (HcclDataType)f2h_datatype_map[datatype], root, comm->base,
      stream->base)];
}

sdcclResult_t hcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclAllReduce(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], comm->base, stream->base)];
}

sdcclResult_t
hcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclReduceScatter(
      sendbuffptr, recvbuff, recvcount,
      (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], comm->base, stream->base)];
}

sdcclResult_t hcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, stream->base)];
}

sdcclResult_t hcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclAlltoAll(
      sendbuffptr, count, (HcclDataType)f2h_datatype_map[datatype], recvbuff,
      count, (HcclDataType)f2h_datatype_map[datatype], comm->base,
      stream->base)];
}

sdcclResult_t hcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (sdcclResult_t)h2f_ret_map[HcclAlltoAllV(
      sendbuffptr, sendcounts, sdispls,
      (HcclDataType)f2h_datatype_map[datatype], recvbuff, recvcounts, rdispls,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, stream->base)];
}

sdcclResult_t hcclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  item.comm = comm;
  item.stream = stream;
  sendRecvInfo.emplace_back(HcclSendRecvItem{
      HcclSendRecvType::HCCL_SEND, sendbuffptr, count,
      (HcclDataType)f2h_datatype_map[datatype], (uint32_t)peer});
  return sdcclSuccess;
}

sdcclResult_t hcclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  sendRecvInfo.emplace_back(HcclSendRecvItem{
      HcclSendRecvType::HCCL_RECV, recvbuff, count,
      (HcclDataType)f2h_datatype_map[datatype], (uint32_t)peer});
  return sdcclSuccess;
}

sdcclResult_t hcclAdaptorGroupStart() {
  sendRecvInfo.clear();
  return sdcclSuccess;
}

sdcclResult_t hcclAdaptorGroupEnd() {
  uint32_t itemNum = 0;
  itemNum = sendRecvInfo.size();
  if (itemNum > 0) {
    return (sdcclResult_t)h2f_ret_map[HcclBatchSendRecv(
        sendRecvInfo.data(), itemNum, item.comm->base, item.stream->base)];
  }
  return sdcclSuccess;
}

sdcclResult_t
hcclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t hcclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor hcclAdaptor = {
    "HCCL",
    // Basic functions
    hcclAdaptorGetVersion, hcclAdaptorGetUniqueId, hcclAdaptorGetErrorString,
    hcclAdaptorGetLastError, hcclAdaptorGetStagedBuffer,
    // Communicator functions
    hcclAdaptorCommInitRank, hcclAdaptorCommFinalize, hcclAdaptorCommDestroy,
    hcclAdaptorCommAbort, hcclAdaptorCommResume, hcclAdaptorCommSuspend,
    hcclAdaptorCommCount, hcclAdaptorCommCuDevice, hcclAdaptorCommUserRank,
    hcclAdaptorCommGetAsyncError, hcclAdaptorMemAlloc, hcclAdaptorMemFree,
    hcclAdaptorCommRegister, hcclAdaptorCommDeregister,
    // Symmetric functions
    hcclAdaptorCommWindowRegister, hcclAdaptorCommWindowDeregister,
    // Communication functions
    hcclAdaptorReduce, hcclAdaptorGather, hcclAdaptorScatter,
    hcclAdaptorBroadcast, hcclAdaptorAllReduce, hcclAdaptorReduceScatter,
    hcclAdaptorAllGather, hcclAdaptorAlltoAll, hcclAdaptorAlltoAllv,
    hcclAdaptorSend, hcclAdaptorRecv,
    // Group semantics
    hcclAdaptorGroupStart, hcclAdaptorGroupEnd,
    // Device API
    hcclAdaptorDevCommCreate, hcclAdaptorDevCommDestroy};

#endif // USE_ASCEND_ADAPTOR
