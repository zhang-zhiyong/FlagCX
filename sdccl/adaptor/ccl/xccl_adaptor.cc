#include "kunlunxin_adaptor.h"
#include <iostream>

#ifdef USE_KUNLUNXIN_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

BKCLDataType sdcclToXcclDataType(sdcclDataType_t type) {
  // use BKCL_UINT8 as unknown data type
  static const struct {
    sdcclDataType_t sdcclType;
    BKCLDataType bkclType;
  } typeMap[] = {
      {sdcclInt8, BKCL_UINT8},        {sdcclChar, BKCL_UINT8},
      {sdcclUint8, BKCL_UINT8},       {sdcclInt32, BKCL_INT32},
      {sdcclInt, BKCL_INT32},         {sdcclUint32, BKCL_INT32},
      {sdcclUint64, BKCL_INT64},      {sdcclInt64, BKCL_INT64},
      {sdcclFloat16, BKCL_FLOAT16},   {sdcclHalf, BKCL_FLOAT16},
      {sdcclFloat32, BKCL_FLOAT},     {sdcclFloat, BKCL_FLOAT},
      {sdcclFloat64, BKCL_FLOAT64},   {sdcclDouble, BKCL_FLOAT64},
      {sdcclBfloat16, BKCL_BFLOAT16},
  };

  const size_t mapSize = sizeof(typeMap) / sizeof(typeMap[0]);

  for (size_t i = 0; i < mapSize; ++i) {
    if (typeMap[i].sdcclType == type) {
      return typeMap[i].bkclType;
    }
  }

  // return unknown data type if not found
  return BKCL_UINT8;
}

BKCLOp sdcclRedOpToBKCLOp(sdcclRedOp_t op) {
  switch (op) {
    case sdcclSum:
      return BKCLOp::BKCL_ADD;
    case sdcclProd:
      return BKCLOp::BKCL_PRODUCT;
    case sdcclMax:
      return BKCLOp::BKCL_MAX;
    case sdcclMin:
      return BKCLOp::BKCL_MIN;
    default:
      // return BKCLOp::BKCL_NUM_OPS to account for unknown redOp type
      return BKCLOp::BKCL_NUM_OPS;
  }
}

// Unsupported
sdcclResult_t xcclAdaptorGetVersion(int *version) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  // Note that when performing heterogeneous communication between Kunlunxin and
  // other devices, XCCL must be used to generate the unique ID.
  return (sdcclResult_t)bkcl_get_unique_id(
      (BKCLUniqueId *)(((char *)*uniqueId) + sizeof(int)));
}

sdcclResult_t xcclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return sdcclNotSupported;
}

// Unsupported
const char *xcclAdaptorGetErrorString(sdcclResult_t result) {
  return "sdcclNotSupported";
}

// Unsupported
const char *xcclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return "sdcclNotSupported";
}

sdcclResult_t xcclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    sdcclCalloc(comm, 1);
  }
  return (sdcclResult_t)bkcl_init_rank(
      &(*comm)->base, rank, nranks,
      (BKCLUniqueId *)((char *)commId + sizeof(int)));
}

// Unsupported
sdcclResult_t xcclAdaptorCommFinalize(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorCommDestroy(sdcclInnerComm_t comm) {
  return (sdcclResult_t)bkcl_destroy_context(comm->base);
}

sdcclResult_t xcclAdaptorCommAbort(sdcclInnerComm_t comm) {
  return (sdcclResult_t)bkcl_comm_abort(comm->base);
}

// Unsupported
sdcclResult_t xcclAdaptorCommResume(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

// Unsupported
sdcclResult_t xcclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)bkcl_comm_count(comm->base, count);
}

// Unsupported
sdcclResult_t xcclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)bkcl_comm_user_rank(comm->base, rank);
}

// TODO: unsupported
sdcclResult_t xcclAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t xcclAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// Unsupported
sdcclResult_t xcclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t xcclAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t xcclAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_reduce(
      comm->base, sendbuff, recvbuff, count, sdcclToXcclDataType(datatype),
      sdcclRedOpToBKCLOp(op), root, stream->base);
}

// Unsupported
sdcclResult_t xcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t stream) {
  int rank, nranks;
  BKCLResult_t res = BKCL_SUCCESS;
  res = bkcl_comm_user_rank(comm->base, &rank);
  res = bkcl_comm_count(comm->base, &nranks);

  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = bkcl_group_start();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = bkcl_send(comm->base, static_cast<const void *>(buffer + r * size),
                      size, r, BKCL_UINT8, stream->base);
    }
  }
  res = bkcl_recv(comm->base, recvbuff, size, root, BKCL_UINT8, stream->base);
  res = bkcl_group_end();

  return (sdcclResult_t)res;
}

sdcclResult_t xcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_broadcast(comm->base, sendbuff, recvbuff, count,
                                        sdcclToXcclDataType(datatype), root,
                                        stream->base);
}

sdcclResult_t xcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_all_reduce(comm->base, sendbuff, recvbuff, count,
                                         sdcclToXcclDataType(datatype),
                                         sdcclRedOpToBKCLOp(op), stream->base);
}

sdcclResult_t
xcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_reduce_scatter(
      comm->base, sendbuff, recvbuff, recvcount, sdcclToXcclDataType(datatype),
      sdcclRedOpToBKCLOp(op), stream->base);
}

sdcclResult_t xcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_all_gather(
      comm->base, sendbuff, sendcount, recvbuff, sdcclToXcclDataType(datatype),
      stream->base);
}

sdcclResult_t xcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_all_to_all(comm->base, sendbuff, count, recvbuff,
                                         sdcclToXcclDataType(datatype),
                                         stream->base);
}

sdcclResult_t xcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  int nranks;
  bkcl_comm_count(comm->base, &nranks);

  size_t *sendcountsDev = NULL;
  size_t *sdisplsDev = NULL;
  size_t *recvcountsDev = NULL;
  size_t *rdisplsDev = NULL;

  xpu_malloc((void **)(&sendcountsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&sdisplsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&recvcountsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&rdisplsDev), nranks * sizeof(size_t));
  xpu_memcpy_async((void *)sendcountsDev, (void *)sendcounts,
                   nranks * sizeof(size_t), XPUMemcpyKind::XPU_HOST_TO_DEVICE,
                   stream->base);
  xpu_memcpy_async((void *)sdisplsDev, (void *)sdispls, nranks * sizeof(size_t),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE, stream->base);
  xpu_memcpy_async((void *)recvcountsDev, (void *)recvcounts,
                   nranks * sizeof(size_t), XPUMemcpyKind::XPU_HOST_TO_DEVICE,
                   stream->base);
  xpu_memcpy_async((void *)rdisplsDev, (void *)rdispls, nranks * sizeof(size_t),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE, stream->base);

  sdcclResult_t res = (sdcclResult_t)bkcl_all_to_all_v(
      comm->base, sendbuff, sendcountsDev, sdisplsDev,
      sdcclToXcclDataType(datatype), recvbuff, recvcountsDev, rdisplsDev,
      sdcclToXcclDataType(datatype), stream->base);
  cudaStreamSynchronize(stream->base);
  xpu_free(sendcountsDev);
  xpu_free(sdisplsDev);
  xpu_free(recvcountsDev);
  xpu_free(rdisplsDev);
  return res;
}

sdcclResult_t xcclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_send(comm->base, sendbuff, count, peer,
                                   sdcclToXcclDataType(datatype),
                                   stream->base);
}

sdcclResult_t xcclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)bkcl_recv(comm->base, recvbuff, count, peer,
                                   sdcclToXcclDataType(datatype),
                                   stream->base);
}

sdcclResult_t xcclAdaptorGroupStart() {
  return (sdcclResult_t)bkcl_group_start();
}

sdcclResult_t xcclAdaptorGroupEnd() {
  return (sdcclResult_t)bkcl_group_end();
}

sdcclResult_t
xcclAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t xcclAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor xcclAdaptor = {
    "XCCL",
    // Basic functions
    xcclAdaptorGetVersion, xcclAdaptorGetUniqueId, xcclAdaptorGetErrorString,
    xcclAdaptorGetLastError, xcclAdaptorGetStagedBuffer,
    // Communicator functions
    xcclAdaptorCommInitRank, xcclAdaptorCommFinalize, xcclAdaptorCommDestroy,
    xcclAdaptorCommAbort, xcclAdaptorCommResume, xcclAdaptorCommSuspend,
    xcclAdaptorCommCount, xcclAdaptorCommCuDevice, xcclAdaptorCommUserRank,
    xcclAdaptorCommGetAsyncError, xcclAdaptorMemAlloc, xcclAdaptorMemFree,
    xcclAdaptorCommRegister, xcclAdaptorCommDeregister,
    // Symmetric functions
    xcclAdaptorCommWindowRegister, xcclAdaptorCommWindowDeregister,
    // Communication functions
    xcclAdaptorReduce, xcclAdaptorGather, xcclAdaptorScatter,
    xcclAdaptorBroadcast, xcclAdaptorAllReduce, xcclAdaptorReduceScatter,
    xcclAdaptorAllGather, xcclAdaptorAlltoAll, xcclAdaptorAlltoAllv,
    xcclAdaptorSend, xcclAdaptorRecv,
    // Group semantics
    xcclAdaptorGroupStart, xcclAdaptorGroupEnd,
    // Device API
    xcclAdaptorDevCommCreate, xcclAdaptorDevCommDestroy};

#endif // USE_KUNLUNXIN_ADAPTOR