#include "gloo_adaptor.h"

#ifdef USE_GLOO_ADAPTOR

SDCCL_PARAM(GlooIbDisable, "GLOO_IB_DISABLE", 0);

static int groupDepth = 0;
static constexpr std::chrono::milliseconds sdcclGlooDefaultTimeout =
    std::chrono::seconds(10000);
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;
static std::vector<bufferPtr> unboundBufferStorage;

// key: peer, value: tag
static std::unordered_map<int, uint32_t> sendPeerTags;
static std::unordered_map<int, uint32_t> recvPeerTags;

// TODO: unsupported
sdcclResult_t glooAdaptorGetVersion(int *version) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t glooAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
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
    sbuff->cnt = 0;
    int newSize = GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      return sdcclSystemError;
    }
    sbuff->size = newSize;
    auto unboundBuffer = comm->base->createUnboundBuffer(
        const_cast<void *>(sbuff->buffer), sbuff->size);
    sbuff->unboundBuffer = unboundBuffer.get();
    unboundBufferStorage.push_back(std::move(unboundBuffer));
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
const char *glooAdaptorGetErrorString(sdcclResult_t result) {
  return "Not Implemented";
}

// TODO: unsupported
const char *glooAdaptorGetLastError(sdcclInnerComm_t comm) {
  return "Not Implemented";
}

sdcclResult_t glooAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t /*commId*/, int rank,
                                       bootstrapState *bootstrap) {
  // Create gloo transport device
  std::shared_ptr<::gloo::transport::Device> dev;
  sdcclNetProperties_t *properties =
      (sdcclNetProperties_t *)bootstrap->properties;
  if (sdcclParamGlooIbDisable() || sdcclParamTopoDetectionDisable()) {
    // Use transport tcp
    ::gloo::transport::tcp::attr attr;
    attr.iface = std::string(bootstrap->bootstrapNetIfName);
    dev = ::gloo::transport::tcp::CreateDevice(attr);
  } else {
    // Use transport ibverbs
    ::gloo::transport::ibverbs::attr attr;
    attr.name = properties->name;
    attr.port = properties->port;
    attr.index = 3; // default index
    const char *ibGidIndex = sdcclGetEnv("SDCCL_IB_GID_INDEX");
    if (ibGidIndex != NULL) {
      attr.index = std::stoi(ibGidIndex);
    }
    dev = ::gloo::transport::ibverbs::CreateDevice(attr);
  }
  if (*comm == NULL) {
    SDCCLCHECK(sdcclCalloc(comm, 1));
  }
  // Create gloo context
  (*comm)->base = std::make_shared<sdcclGlooContext>(rank, nranks, bootstrap);
  (*comm)->base->connectFullMesh(dev);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorCommFinalize(sdcclInnerComm_t comm) {
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
  unboundBufferStorage.clear();
  comm->base.reset();
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorCommDestroy(sdcclInnerComm_t comm) {
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
  unboundBufferStorage.clear();
  comm->base.reset();
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorCommAbort(sdcclInnerComm_t comm) {
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
  unboundBufferStorage.clear();
  comm->base.reset();
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t glooAdaptorCommResume(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t glooAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  *count = comm->base->size;
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  device = NULL;
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  *rank = comm->base->rank;
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t glooAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t glooAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t glooAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t glooAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t glooAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t /*stream*/) {
  ::gloo::ReduceOptions opts(comm->base);
  opts.setRoot(root);
  opts.setReduceFunction(
      getFunction<::gloo::ReduceOptions::Func>(datatype, op));
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  ::gloo::reduce(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t /*stream*/) {
  ::gloo::GatherOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  // Set output pointer only when root
  if (root == comm->base->rank) {
    GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                        comm->base->size * count);
  }
  opts.setRoot(root);
  ::gloo::gather(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclInnerComm_t comm,
                                  sdcclStream_t /*stream*/) {
  ::gloo::ScatterOptions opts(comm->base);
  // one pointer per rank
  std::vector<void *> sendPtrs(comm->base->size);
  for (int i = 0; i < comm->base->size; ++i) {
    sendPtrs[i] = static_cast<void *>(
        (char *)sendbuff + i * count * getSdcclDataTypeSize(datatype));
  }
  GENERATE_GLOO_TYPES(datatype, setInputs, opts,
                      const_cast<void **>(sendPtrs.data()), comm->base->size,
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  opts.setRoot(root);
  ::gloo::scatter(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  ::gloo::BroadcastOptions opts(comm->base);
  // Set input pointer only when root
  if (root == comm->base->rank) {
    GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                        count);
  }
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  opts.setRoot(root);
  ::gloo::broadcast(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  ::gloo::AllreduceOptions opts(comm->base);
  opts.setReduceFunction(
      getFunction<::gloo::AllreduceOptions::Func>(datatype, op));
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, count);
  ::gloo::allreduce(opts);
  return sdcclSuccess;
}

// TODO: unsupported
sdcclResult_t
glooAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t /*stream*/) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  ::gloo::AllgatherOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      sendcount);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                      comm->base->size * sendcount);
  ::gloo::allgather(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t /*stream*/) {
  ::gloo::AlltoallOptions opts(comm->base);
  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      comm->base->size * count);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff,
                      comm->base->size * count);
  ::gloo::alltoall(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t /*stream*/) {
  // Note that sdispls and rdispls are not used in Gloo.
  ::gloo::AlltoallvOptions opts(comm->base);
  std::vector<int64_t> sendCnt(comm->base->size);
  std::vector<int64_t> recvCnt(comm->base->size);
  for (int i = 0; i < comm->base->size; ++i) {
    sendCnt[i] = sendcounts[i];
    recvCnt[i] = recvcounts[i];
  }

  GENERATE_GLOO_TYPES(datatype, setInput, opts, const_cast<void *>(sendbuff),
                      sendCnt);
  GENERATE_GLOO_TYPES(datatype, setOutput, opts, recvbuff, recvCnt);
  ::gloo::alltoallv(opts);
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm,
                               sdcclStream_t /*stream*/) {
  size_t size = count * getSdcclDataTypeSize(datatype);
  stagedBuffer_t buff = sendStagedBufferList.back();
  uint32_t utag;
  if (sendPeerTags.find(peer) != sendPeerTags.end()) {
    utag = sendPeerTags[peer];
  } else {
    utag = 0;
    sendPeerTags[peer] = 0;
  }
  buff->unboundBuffer->send(peer, utag, buff->offset, size);
  buff->offset += size;
  sendPeerTags[peer] = utag + 1;
  if (groupDepth == 0) {
    buff->unboundBuffer->waitSend(sdcclGlooDefaultTimeout);
  } else {
    buff->cnt++;
  }
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm,
                               sdcclStream_t /*stream*/) {
  size_t size = count * getSdcclDataTypeSize(datatype);
  stagedBuffer_t buff = recvStagedBufferList.back();
  uint32_t utag;
  if (recvPeerTags.find(peer) != recvPeerTags.end()) {
    utag = recvPeerTags[peer];
  } else {
    utag = 0;
    recvPeerTags[peer] = 0;
  }
  buff->unboundBuffer->recv(peer, utag, buff->offset, size);
  buff->offset += size;
  recvPeerTags[peer] = utag + 1;
  if (groupDepth == 0) {
    buff->unboundBuffer->waitRecv(sdcclGlooDefaultTimeout);
  } else {
    buff->cnt++;
  }
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorGroupStart() {
  groupDepth++;
  return sdcclSuccess;
}

sdcclResult_t glooAdaptorGroupEnd() {
  groupDepth--;
  if (groupDepth == 0) {
    for (size_t i = 0; i < sendStagedBufferList.size(); ++i) {
      stagedBuffer *buff = sendStagedBufferList[i];
      while (buff->cnt > 0) {
        buff->unboundBuffer->waitSend(sdcclGlooDefaultTimeout);
        buff->cnt--;
      }
      buff->offset = 0;
    }
    for (size_t i = 0; i < recvStagedBufferList.size(); ++i) {
      stagedBuffer *buff = recvStagedBufferList[i];
      while (buff->cnt > 0) {
        buff->unboundBuffer->waitRecv(sdcclGlooDefaultTimeout);
        buff->cnt--;
      }
      buff->offset = 0;
    }
    sendPeerTags.clear();
    recvPeerTags.clear();
  }
  return sdcclSuccess;
}

sdcclResult_t
glooAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                         const sdcclDevCommRequirements * /*reqs*/,
                         sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t glooAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                         sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor glooAdaptor = {
    "GLOO",
    // Basic functions
    glooAdaptorGetVersion, glooAdaptorGetUniqueId, glooAdaptorGetErrorString,
    glooAdaptorGetLastError, glooAdaptorGetStagedBuffer,
    // Communicator functions
    glooAdaptorCommInitRank, glooAdaptorCommFinalize, glooAdaptorCommDestroy,
    glooAdaptorCommAbort, glooAdaptorCommResume, glooAdaptorCommSuspend,
    glooAdaptorCommCount, glooAdaptorCommCuDevice, glooAdaptorCommUserRank,
    glooAdaptorCommGetAsyncError, glooAdaptorMemAlloc, glooAdaptorMemFree,
    glooAdaptorCommRegister, glooAdaptorCommDeregister,
    // Symmetric functions
    glooAdaptorCommWindowRegister, glooAdaptorCommWindowDeregister,
    // Communication functions
    glooAdaptorReduce, glooAdaptorGather, glooAdaptorScatter,
    glooAdaptorBroadcast, glooAdaptorAllReduce, glooAdaptorReduceScatter,
    glooAdaptorAllGather, glooAdaptorAlltoAll, glooAdaptorAlltoAllv,
    glooAdaptorSend, glooAdaptorRecv,
    // Group semantics
    glooAdaptorGroupStart, glooAdaptorGroupEnd,
    // Device API
    glooAdaptorDevCommCreate, glooAdaptorDevCommDestroy};

#endif // USE_GLOO_ADAPTOR