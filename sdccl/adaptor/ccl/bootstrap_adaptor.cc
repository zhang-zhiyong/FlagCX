#include "bootstrap_adaptor.h"
#include "bootstrap.h"

#ifdef USE_BOOTSTRAP_ADAPTOR

static int groupDepth = 0;
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;

// TODO: unsupported
sdcclResult_t bootstrapAdaptorGetVersion(int *version) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  return sdcclNotSupported;
}

sdcclResult_t bootstrapAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                               void **buff, size_t size,
                                               int isRecv) {
  stagedBuffer *sbuff = NULL;
  if (isRecv) {
    for (auto it = recvStagedBufferList.begin();
         it != recvStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  } else {
    for (auto it = sendStagedBufferList.begin();
         it != sendStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  }
  if (sbuff == NULL) {
    SDCCLCHECK(sdcclCalloc(&sbuff, 1));
    sbuff->offset = 0;
    int newSize = BOOTSTRAP_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      return sdcclSystemError;
    }
    sbuff->size = newSize;
    if (isRecv) {
      recvStagedBufferList.push_back(sbuff);
    } else {
      sendStagedBufferList.push_back(sbuff);
    }
  }
  *buff = (void *)((char *)sbuff->buffer + sbuff->offset);
  return sdcclSuccess;
}

// TODO: unsupported
const char *bootstrapAdaptorGetErrorString(sdcclResult_t result) {
  return "Not Implemented";
}

// TODO: unsupported
const char *bootstrapAdaptorGetLastError(sdcclInnerComm_t comm) {
  return "Not Implemented";
}

sdcclResult_t bootstrapAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                            sdcclUniqueId_t /*commId*/,
                                            int rank,
                                            bootstrapState *bootstrap) {
  if (*comm == NULL) {
    SDCCLCHECK(sdcclCalloc(comm, 1));
  }
  (*comm)->base = bootstrap;

  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorCommFinalize(sdcclInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorCommDestroy(sdcclInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorCommAbort(sdcclInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorCommResume(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t bootstrapAdaptorCommCount(const sdcclInnerComm_t comm,
                                         int *count) {
  *count = comm->base->nranks;
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                            int *device) {
  device = NULL;
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                            int *rank) {
  *rank = comm->base->rank;
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                                 sdcclResult_t *asyncError) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t bootstrapAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                            size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t bootstrapAdaptorCommDeregister(sdcclInnerComm_t comm,
                                              void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t bootstrapAdaptorCommWindowRegister(sdcclInnerComm_t comm,
                                                  void *buff, size_t size,
                                                  sdcclWindow_t *win,
                                                  int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t bootstrapAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                                    sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t bootstrapAdaptorGather(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      int root, sdcclInnerComm_t comm,
                                      sdcclStream_t /*stream*/) {
  SDCCLCHECK(
      GatherBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorScatter(const void *sendbuff, void *recvbuff,
                                       size_t count, sdcclDataType_t datatype,
                                       int root, sdcclInnerComm_t comm,
                                       sdcclStream_t /*stream*/) {
  SDCCLCHECK(
      ScatterBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                         size_t count,
                                         sdcclDataType_t datatype, int root,
                                         sdcclInnerComm_t comm,
                                         sdcclStream_t /*stream*/) {
  SDCCLCHECK(BroadcastBootstrap(comm->base, sendbuff, recvbuff, count,
                                 datatype, root));
  return sdcclSuccess;
}

sdcclResult_t
bootstrapAdaptorAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                          sdcclDataType_t datatype, sdcclRedOp_t op,
                          sdcclInnerComm_t comm, sdcclStream_t /*stream*/) {
  SDCCLCHECK(
      AllReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype, op));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclRedOp_t op, int root,
                                      sdcclInnerComm_t comm,
                                      sdcclStream_t /*stream*/) {
  SDCCLCHECK(ReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype,
                              op, root));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorReduceScatter(const void *sendbuff,
                                             void *recvbuff, size_t recvcount,
                                             sdcclDataType_t datatype,
                                             sdcclRedOp_t op,
                                             sdcclInnerComm_t comm,
                                             sdcclStream_t /*stream*/) {
  SDCCLCHECK(ReduceScatterBootstrap(comm->base, sendbuff, recvbuff, recvcount,
                                     datatype, op));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                         size_t sendcount,
                                         sdcclDataType_t datatype,
                                         sdcclInnerComm_t comm,
                                         sdcclStream_t /*stream*/) {
  SDCCLCHECK(
      AllGatherBootstrap(comm->base, sendbuff, recvbuff, sendcount, datatype));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                        size_t count, sdcclDataType_t datatype,
                                        sdcclInnerComm_t comm,
                                        sdcclStream_t /*stream*/) {
  SDCCLCHECK(
      AlltoAllBootstrap(comm->base, sendbuff, recvbuff, count, datatype));
  return sdcclSuccess;
}

sdcclResult_t
bootstrapAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                          size_t *sdispls, void *recvbuff, size_t *recvcounts,
                          size_t *rdispls, sdcclDataType_t datatype,
                          sdcclInnerComm_t comm, sdcclStream_t /*stream*/) {
  SDCCLCHECK(AlltoAllvBootstrap(comm->base, sendbuff, sendcounts, sdispls,
                                 recvbuff, recvcounts, rdispls, datatype));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorSend(const void *sendbuff, size_t count,
                                    sdcclDataType_t datatype, int peer,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  // TODO(MC952-arch): implement out-of-order sends
  size_t size = count * getSdcclDataTypeSize(datatype);
  SDCCLCHECK(bootstrapSend(comm->base, peer, BOOTSTRAP_ADAPTOR_SEND_RECV_TAG,
                            (void *)sendbuff, size));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorRecv(void *recvbuff, size_t count,
                                    sdcclDataType_t datatype, int peer,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  // TODO(MC952-arch): implement out-of-order recvs
  size_t size = count * getSdcclDataTypeSize(datatype);
  SDCCLCHECK(bootstrapRecv(comm->base, peer, BOOTSTRAP_ADAPTOR_SEND_RECV_TAG,
                            recvbuff, size));
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorGroupStart() {
  groupDepth++;
  return sdcclSuccess;
}

sdcclResult_t bootstrapAdaptorGroupEnd() {
  groupDepth--;
  if (groupDepth == 0) {
    for (size_t i = 0; i < sendStagedBufferList.size(); ++i) {
      stagedBuffer *buff = sendStagedBufferList[i];
      buff->offset = 0;
    }
    for (size_t i = 0; i < recvStagedBufferList.size(); ++i) {
      stagedBuffer *buff = recvStagedBufferList[i];
      buff->offset = 0;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t
bootstrapAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                              const sdcclDevCommRequirements * /*reqs*/,
                              sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t
bootstrapAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                               sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor bootstrapAdaptor = {
    "BOOTSTRAP",
    // Basic functions
    bootstrapAdaptorGetVersion, bootstrapAdaptorGetUniqueId,
    bootstrapAdaptorGetErrorString, bootstrapAdaptorGetLastError,
    bootstrapAdaptorGetStagedBuffer,
    // Communicator functions
    bootstrapAdaptorCommInitRank, bootstrapAdaptorCommFinalize,
    bootstrapAdaptorCommDestroy, bootstrapAdaptorCommAbort,
    bootstrapAdaptorCommResume, bootstrapAdaptorCommSuspend,
    bootstrapAdaptorCommCount, bootstrapAdaptorCommCuDevice,
    bootstrapAdaptorCommUserRank, bootstrapAdaptorCommGetAsyncError,
    bootstrapAdaptorMemAlloc, bootstrapAdaptorMemFree,
    bootstrapAdaptorCommRegister, bootstrapAdaptorCommDeregister,
    // Symmetric functions
    bootstrapAdaptorCommWindowRegister, bootstrapAdaptorCommWindowDeregister,
    // Communication functions
    bootstrapAdaptorReduce, bootstrapAdaptorGather, bootstrapAdaptorScatter,
    bootstrapAdaptorBroadcast, bootstrapAdaptorAllReduce,
    bootstrapAdaptorReduceScatter, bootstrapAdaptorAllGather,
    bootstrapAdaptorAlltoAll, bootstrapAdaptorAlltoAllv, bootstrapAdaptorSend,
    bootstrapAdaptorRecv,
    // Group semantics
    bootstrapAdaptorGroupStart, bootstrapAdaptorGroupEnd,
    // Device API
    bootstrapAdaptorDevCommCreate, bootstrapAdaptorDevCommDestroy};

#endif // USE_BOOTSTRAP_ADAPTOR