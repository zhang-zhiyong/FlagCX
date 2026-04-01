#include "mpi_adaptor.h"
#include <functional>

#ifdef USE_MPI_ADAPTOR

static int groupDepth = 0;
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;

static sdcclResult_t validateComm(sdcclInnerComm_t comm) {
  if (!comm || !comm->base) {
    return sdcclInvalidArgument;
  }

  if (!comm->base->isValidContext()) {
    printf("Error: Invalid MPI context: %s\n",
           comm->base->getLastError().c_str());
    return sdcclInternalError;
  }

  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorGetVersion(int *version) {
  int subversion;
  int result = MPI_Get_version(version, &subversion);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                         void **buff, size_t size, int isRecv) {
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
    int newSize = MPI_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
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

const char *mpiAdaptorGetErrorString(sdcclResult_t result) {
  switch (result) {
    case sdcclSuccess:
      return "MPI Success";
    case sdcclInternalError:
      return "MPI Internal Error";
    default:
      return "MPI Unknown Error";
  }
}

const char *mpiAdaptorGetLastError(sdcclInnerComm_t comm) {

  return "MPI: No Error";
}

sdcclResult_t mpiAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                      sdcclUniqueId_t /*commId*/, int rank,
                                      bootstrapState *bootstrap) {
  int initialized;
  MPI_Initialized(&initialized);

  if (!initialized) {
    int provided;
    int result = MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (result != MPI_SUCCESS) {
      return sdcclInternalError;
    }

    if (provided < MPI_THREAD_SERIALIZED) {
      printf("Warning: MPI does not support required thread level\n");
    }
  }

  int mpiRank, mpiSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  // validate parameters if bootstrap is provided
  if (bootstrap != nullptr) {
    if (rank != mpiRank || nranks != mpiSize) {
      printf("Warning: Expected rank/size (%d/%d) differs from MPI (%d/%d), "
             "using MPI values\n",
             rank, nranks, mpiRank, mpiSize);
    }
  }

  if (*comm == NULL) {
    SDCCLCHECK(sdcclCalloc(comm, 1));
  }

  // use actual MPI rank and size to create context
  (*comm)->base =
      std::make_shared<sdcclMpiContext>(mpiRank, mpiSize, bootstrap);

  // check if context is created successfully
  if (!(*comm)->base || !(*comm)->base->isValidContext()) {
    printf("Error: Failed to create MPI context: %s\n",
           (*comm)->base ? (*comm)->base->getLastError().c_str()
                         : "Unknown error");
    return sdcclInternalError;
  }
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommFinalize(sdcclInnerComm_t comm) {
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
  comm->base.reset();
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommDestroy(sdcclInnerComm_t comm) {
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
  comm->base.reset();
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommAbort(sdcclInnerComm_t comm) {
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
  MPI_Abort(comm->base->getMpiComm(), 1);
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommResume(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  *count = comm->base->getSize();
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                      int *device) {
  *device = -1;
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommUserRank(const sdcclInnerComm_t comm, int *rank) {
  *rank = comm->base->getRank();
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                           sdcclResult_t *asyncError) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t mpiAdaptorMemAlloc(void **ptr, size_t size) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t mpiAdaptorMemFree(void *ptr) { return sdcclNotSupported; }

// TODO: unsupported
sdcclResult_t mpiAdaptorCommRegister(sdcclInnerComm_t comm, void *buff,
                                      size_t size, void **handle) {
  return sdcclNotSupported;
}

// TODO: unsupported
sdcclResult_t mpiAdaptorCommDeregister(sdcclInnerComm_t comm, void *handle) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                            size_t size, sdcclWindow_t *win,
                                            int winFlags) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                              sdcclWindow_t win) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorReduce(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                sdcclRedOp_t op, int root,
                                sdcclInnerComm_t comm,
                                sdcclStream_t /*stream*/) {
  int result;
  MPI_Op mpiOp = getSdcclToMpiOp(op);
  CALL_MPI_REDUCE(datatype, sendbuff, recvbuff, count, mpiOp, root,
                  comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclInnerComm_t comm,
                                   sdcclStream_t /*stream*/) {
  int result;
  void *buffer =
      (comm->base->getRank() == root) ? const_cast<void *>(sendbuff) : recvbuff;

  CALL_MPI_BCAST(datatype, buffer, count, root, comm->base->getMpiComm(),
                 &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, sdcclInnerComm_t comm,
                                   sdcclStream_t /*stream*/) {
  SDCCLCHECK(validateComm(comm));

  int result;
  MPI_Op mpiOp = getSdcclToMpiOp(op);
  CALL_MPI_ALLREDUCE(datatype, sendbuff, recvbuff, count, mpiOp,
                     comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorGather(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                int root, sdcclInnerComm_t comm,
                                sdcclStream_t /*stream*/) {
  int result;
  CALL_MPI_GATHER(datatype, sendbuff, count, recvbuff, count, root,
                  comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclInnerComm_t comm,
                                 sdcclStream_t /*stream*/) {
  int result;
  CALL_MPI_SCATTER(datatype, sendbuff, count, recvbuff, count, root,
                   comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       sdcclDataType_t datatype,
                                       sdcclRedOp_t op, sdcclInnerComm_t comm,
                                       sdcclStream_t /*stream*/) {
  int result;
  MPI_Op mpiOp = getSdcclToMpiOp(op);
  CALL_MPI_REDUCE_SCATTER(datatype, sendbuff, recvbuff, recvcount, mpiOp,
                          comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t /*stream*/) {
  int result;
  CALL_MPI_ALLGATHER(datatype, sendbuff, sendcount, recvbuff, sendcount,
                     comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  sdcclInnerComm_t comm,
                                  sdcclStream_t /*stream*/) {
  int result;
  CALL_MPI_ALLTOALL(datatype, sendbuff, count, recvbuff, count,
                    comm->base->getMpiComm(), &result);
  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   sdcclDataType_t datatype,
                                   sdcclInnerComm_t comm,
                                   sdcclStream_t /*stream*/) {
  SDCCLCHECK(validateComm(comm));

  // validate parameters
  if (!sendcounts || !sdispls || !recvcounts || !rdispls) {
    printf(
        "Error: AlltoAllv requires non-null count and displacement arrays\n");
    return sdcclInvalidArgument;
  }

  int size = comm->base->getSize();
  MPI_Datatype mpiDatatype = getSdcclToMpiDataType(datatype);

  std::vector<int> mpiSendcounts(size), mpiRecvcounts(size);
  std::vector<int> mpiSdispls(size), mpiRdispls(size);

  for (int i = 0; i < size; i++) {
    mpiSendcounts[i] = static_cast<int>(sendcounts[i]);
    mpiRecvcounts[i] = static_cast<int>(recvcounts[i]);
    mpiSdispls[i] = static_cast<int>(sdispls[i]);
    mpiRdispls[i] = static_cast<int>(rdispls[i]);
  }

  int result =
      MPI_Alltoallv(sendbuff, mpiSendcounts.data(), mpiSdispls.data(),
                    mpiDatatype, recvbuff, mpiRecvcounts.data(),
                    mpiRdispls.data(), mpiDatatype, comm->base->getMpiComm());

  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclInternalError;
}

sdcclResult_t mpiAdaptorSend(const void *sendbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclInnerComm_t comm,
                              sdcclStream_t /*stream*/) {
  SDCCLCHECK(validateComm(comm));

  // validate peer range
  if (peer < 0 || peer >= comm->base->getSize()) {
    printf("Error: Invalid peer %d, must be in range [0, %d)\n", peer,
           comm->base->getSize());
    return sdcclInvalidArgument;
  }

  MPI_Datatype mpiDatatype = getSdcclToMpiDataType(datatype);
  int tag = 0;

  int result = MPI_Send(sendbuff, static_cast<int>(count), mpiDatatype, peer,
                        tag, comm->base->getMpiComm());

  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclSystemError;
}

sdcclResult_t mpiAdaptorRecv(void *recvbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclInnerComm_t comm,
                              sdcclStream_t /*stream*/) {
  SDCCLCHECK(validateComm(comm));

  // validate peer range (allow MPI_ANY_SOURCE)
  if (peer != MPI_ANY_SOURCE && (peer < 0 || peer >= comm->base->getSize())) {
    printf(
        "Error: Invalid peer %d, must be in range [0, %d) or MPI_ANY_SOURCE\n",
        peer, comm->base->getSize());
    return sdcclInvalidArgument;
  }

  MPI_Datatype mpiDatatype = getSdcclToMpiDataType(datatype);
  int tag = 0;
  MPI_Status status;

  int result = MPI_Recv(recvbuff, static_cast<int>(count), mpiDatatype, peer,
                        tag, comm->base->getMpiComm(), &status);

  return (result == MPI_SUCCESS) ? sdcclSuccess : sdcclSystemError;
}

sdcclResult_t mpiAdaptorGroupStart() {
  groupDepth++;
  return sdcclSuccess;
}

sdcclResult_t mpiAdaptorGroupEnd() {
  groupDepth--;
  if (groupDepth == 0) {
    for (size_t i = 0; i < sendStagedBufferList.size(); ++i) {
      sendStagedBufferList[i]->offset = 0;
    }
    for (size_t i = 0; i < recvStagedBufferList.size(); ++i) {
      recvStagedBufferList[i]->offset = 0;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t
mpiAdaptorDevCommCreate(sdcclInnerComm_t /*comm*/,
                        const sdcclDevCommRequirements * /*reqs*/,
                        sdcclInnerDevComm_t * /*devComm*/) {
  return sdcclNotSupported;
}

sdcclResult_t mpiAdaptorDevCommDestroy(sdcclInnerComm_t /*comm*/,
                                        sdcclInnerDevComm_t /*devComm*/) {
  return sdcclNotSupported;
}

struct sdcclCCLAdaptor mpiAdaptor = {
    "MPI",
    // Basic functions
    mpiAdaptorGetVersion, mpiAdaptorGetUniqueId, mpiAdaptorGetErrorString,
    mpiAdaptorGetLastError, mpiAdaptorGetStagedBuffer,
    // Communicator functions
    mpiAdaptorCommInitRank, mpiAdaptorCommFinalize, mpiAdaptorCommDestroy,
    mpiAdaptorCommAbort, mpiAdaptorCommResume, mpiAdaptorCommSuspend,
    mpiAdaptorCommCount, mpiAdaptorCommCuDevice, mpiAdaptorCommUserRank,
    mpiAdaptorCommGetAsyncError, mpiAdaptorMemAlloc, mpiAdaptorMemFree,
    mpiAdaptorCommRegister, mpiAdaptorCommDeregister,
    // Symmetric functions
    mpiAdaptorCommWindowRegister, mpiAdaptorCommWindowDeregister,
    // Communication functions
    mpiAdaptorReduce, mpiAdaptorGather, mpiAdaptorScatter, mpiAdaptorBroadcast,
    mpiAdaptorAllReduce, mpiAdaptorReduceScatter, mpiAdaptorAllGather,
    mpiAdaptorAlltoAll, mpiAdaptorAlltoAllv, mpiAdaptorSend, mpiAdaptorRecv,
    // Group semantics
    mpiAdaptorGroupStart, mpiAdaptorGroupEnd,
    // Device API
    mpiAdaptorDevCommCreate, mpiAdaptorDevCommDestroy};

#endif // USE_MPI_ADAPTOR