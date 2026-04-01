#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

static bool checkIsAllCudaP2p(ncclComm_t comm) {
  int gpuCount;
  if (cudaGetDeviceCount(&gpuCount) != cudaSuccess) {
    return false;
  }

  for (int i = 0; i < gpuCount; ++i) {
    for (int j = i + 1; j < gpuCount; ++j) {
      int canAccess = 0;
      if (cudaDeviceCanAccessPeer(&canAccess, i, j) != cudaSuccess ||
          !canAccess) {
        return false;
      }
    }
  }
  return true;
}
static bool checkNvlsSupport() {
  int driverVersion, currentDevice;
  CUdevice dev;
  int multicastSupported = 0;
  if (cudaDriverGetVersion(&driverVersion) != cudaSuccess ||
      driverVersion < 12010 || cudaGetDevice(&currentDevice) != cudaSuccess ||
      cuDeviceGet(&dev, currentDevice) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&multicastSupported,
                           CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                           dev) != CUDA_SUCCESS) {
    return false;
  }
  return (multicastSupported != 0);
}
sdcclResult_t ncclAdaptorGetVersion(int *version) {
  return (sdcclResult_t)ncclGetVersion(version);
}

sdcclResult_t ncclAdaptorGetUniqueId(sdcclUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    sdcclCalloc(uniqueId, 1);
  }
  return (sdcclResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

sdcclResult_t ncclAdaptorGetStagedBuffer(const sdcclInnerComm_t comm,
                                          void **buff, size_t /*size*/,
                                          int isRecv) {
  sdcclResult_t res = sdcclSuccess;
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (isRecv && comm->recvStagedBuff == NULL) {
    SDCCLCHECK(sdcclCalloc(&comm->recvStagedBuff, 1));
    res = (sdcclResult_t)ncclMemAlloc(&comm->recvStagedBuff->buff,
                                       NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != sdcclSuccess) {
      free(comm->recvStagedBuff);
      comm->recvStagedBuff = NULL;
      return res;
    }
    res = (sdcclResult_t)ncclCommWindowRegister(
        comm->base, comm->recvStagedBuff->buff,
        NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE, &comm->recvStagedBuff->win,
        NCCL_WIN_COLL_SYMMETRIC);
    if (res != sdcclSuccess) {
      (void)ncclMemFree(comm->recvStagedBuff->buff);
      free(comm->recvStagedBuff);
      comm->recvStagedBuff = NULL;
      return res;
    }
  } else if (!isRecv && comm->sendStagedBuff == NULL) {
    SDCCLCHECK(sdcclCalloc(&comm->sendStagedBuff, 1));
    res = (sdcclResult_t)ncclMemAlloc(&comm->sendStagedBuff->buff,
                                       NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != sdcclSuccess) {
      free(comm->sendStagedBuff);
      comm->sendStagedBuff = NULL;
      return res;
    }
    res = (sdcclResult_t)ncclCommWindowRegister(
        comm->base, comm->sendStagedBuff->buff,
        NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE, &comm->sendStagedBuff->win,
        NCCL_WIN_COLL_SYMMETRIC);
    if (res != sdcclSuccess) {
      (void)ncclMemFree(comm->sendStagedBuff->buff);
      free(comm->sendStagedBuff);
      comm->sendStagedBuff = NULL;
      return res;
    }
  }
  if (buff) {
    if (isRecv) {
      *buff = comm->recvStagedBuff->buff;
    } else {
      *buff = comm->sendStagedBuff->buff;
    }
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  return res;
}

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
static sdcclResult_t ncclDevCommCreateHelper(ncclComm_t comm,
                                              ncclDevCommRequirements *reqs,
                                              ncclDevComm *devComm) {
  using pncclDevCommCreate_t =
      sdcclCustomOpFunc_t<ncclResult_t, ncclComm_t, ncclDevCommRequirements *,
                           ncclDevComm *>;
  void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    return sdcclInternalError;
  }
  auto fn = reinterpret_cast<pncclDevCommCreate_t>(
      dlsym(handle, "pncclDevCommCreate"));
  if (!fn) {
    dlclose(handle);
    return sdcclInternalError;
  }
  ncclResult_t ret = fn(comm, reqs, devComm);
  dlclose(handle);
  return (sdcclResult_t)ret;
}

static sdcclResult_t ncclDevCommDestroyHelper(ncclComm_t comm,
                                               const ncclDevComm *devComm) {
  using pncclDevCommDestroy_t =
      sdcclCustomOpFunc_t<ncclResult_t, ncclComm_t, const ncclDevComm *>;
  void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    return sdcclInternalError;
  }
  auto fn = reinterpret_cast<pncclDevCommDestroy_t>(
      dlsym(handle, "pncclDevCommDestroy"));
  if (!fn) {
    dlclose(handle);
    return sdcclInternalError;
  }
  ncclResult_t ret = fn(comm, devComm);
  dlclose(handle);
  return (sdcclResult_t)ret;
}
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

const char *ncclAdaptorGetErrorString(sdcclResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ncclAdaptorGetLastError(sdcclInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

sdcclResult_t ncclAdaptorCommInitRank(sdcclInnerComm_t *comm, int nranks,
                                       sdcclUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    void *p = malloc(sizeof(struct sdcclInnerComm));
    memset(p, 0, sizeof(struct sdcclInnerComm));
    (*comm) = (struct sdcclInnerComm *)p;
  }
  SDCCLCHECK((sdcclResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                               *(ncclUniqueId *)commId, rank));

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if ((*comm)->devBase == NULL) {
    const char *winEnv = sdcclGetEnv("NCCL_WIN_ENABLE");
    const char *cuMemEnv = sdcclGetEnv("NCCL_CUMEM_ENABLE");
    const char *crossNicEnv = sdcclGetEnv("NCCL_CROSS_NIC");
    const char *ibDisableEnv = sdcclGetEnv("NCCL_IB_DISABLE");
    const char *ibMergeNicsEnv = sdcclGetEnv("NCCL_IB_MERGE_NICS");
    int winEnable = winEnv ? atoi(winEnv) : 1;
    int cuMemEnable = cuMemEnv ? atoi(cuMemEnv) : -2;
    int crossNic = crossNicEnv ? atoi(crossNicEnv) : 2;
    int ibDisable = ibDisableEnv ? atoi(ibDisableEnv) : 0;
    int ibMergeNics = ibMergeNicsEnv ? atoi(ibMergeNicsEnv) : 0;
    bool symmetricSupport = (crossNic > 0) && (ibDisable == 0) &&
                            (ibMergeNics == 0) &&
                            checkIsAllCudaP2p((*comm)->base);
    if (winEnable && cuMemEnable != 0 && symmetricSupport) {
      SDCCLCHECK(sdcclCalloc(&(*comm)->devBase, 1));
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      reqs.lsaBarrierCount = NCCL_ADAPTOR_DEVICE_CTA_COUNT;
      reqs.lsaMultimem = checkNvlsSupport();
      // Adaptor DevComm is intra-node only; GIN barriers/signals not needed
      // here. Kernel-level DevComm requests GIN resources via
      // sdcclDevCommCreate. reqs.railGinBarrierCount =
      // NCCL_ADAPTOR_DEVICE_CTA_COUNT; reqs.ginSignalCount = 1;
      sdcclResult_t devCommRes =
          ncclDevCommCreateHelper((*comm)->base, &reqs, (*comm)->devBase);
      if (devCommRes != sdcclSuccess) {
        WARN("ncclDevCommCreate unavailable (res=%d), DevComm disabled",
             devCommRes);
        free((*comm)->devBase);
        (*comm)->devBase = NULL;
      }
    }
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  SDCCLCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 1));
  SDCCLCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 0));
  return sdcclSuccess;
}

sdcclResult_t ncclAdaptorCommFinalize(sdcclInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    SDCCLCHECK(ncclDevCommDestroyHelper(comm->base, comm->devBase));
    free(comm->devBase);
    comm->devBase = NULL;
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  SDCCLCHECK((sdcclResult_t)ncclCommFinalize(comm->base));
  free(comm);
  return sdcclSuccess;
}

sdcclResult_t ncclAdaptorCommDestroy(sdcclInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    SDCCLCHECK(ncclDevCommDestroyHelper(comm->base, comm->devBase));
    free(comm->devBase);
    comm->devBase = NULL;
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  SDCCLCHECK((sdcclResult_t)ncclCommDestroy(comm->base));
  free(comm);
  return sdcclSuccess;
}

sdcclResult_t ncclAdaptorCommAbort(sdcclInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    SDCCLCHECK((sdcclResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    SDCCLCHECK((sdcclResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    SDCCLCHECK(ncclDevCommDestroyHelper(comm->base, comm->devBase));
    free(comm->devBase);
    comm->devBase = NULL;
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  SDCCLCHECK((sdcclResult_t)ncclCommAbort(comm->base));
  free(comm);
  return sdcclSuccess;
}

sdcclResult_t ncclAdaptorCommResume(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t ncclAdaptorCommSuspend(sdcclInnerComm_t comm) {
  return (sdcclResult_t)ncclInvalidUsage;
}

sdcclResult_t ncclAdaptorCommCount(const sdcclInnerComm_t comm, int *count) {
  return (sdcclResult_t)ncclCommCount(comm->base, count);
}

sdcclResult_t ncclAdaptorCommCuDevice(const sdcclInnerComm_t comm,
                                       int *device) {
  return (sdcclResult_t)ncclCommCuDevice(comm->base, device);
}

sdcclResult_t ncclAdaptorCommUserRank(const sdcclInnerComm_t comm,
                                       int *rank) {
  return (sdcclResult_t)ncclCommUserRank(comm->base, rank);
}

sdcclResult_t ncclAdaptorCommGetAsyncError(sdcclInnerComm_t comm,
                                            sdcclResult_t *asyncError) {
  return (sdcclResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

sdcclResult_t ncclAdaptorMemAlloc(void **ptr, size_t size) {
  return (sdcclResult_t)ncclMemAlloc(ptr, size);
}

sdcclResult_t ncclAdaptorMemFree(void *ptr) {
  return (sdcclResult_t)ncclMemFree(ptr);
}

sdcclResult_t ncclAdaptorCommRegister(const sdcclInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return (sdcclResult_t)ncclCommRegister(comm->base, buff, size, handle);
}

sdcclResult_t ncclAdaptorCommDeregister(const sdcclInnerComm_t comm,
                                         void *handle) {
  return (sdcclResult_t)ncclCommDeregister(comm->base, handle);
}

sdcclResult_t ncclAdaptorCommWindowRegister(sdcclInnerComm_t comm, void *buff,
                                             size_t size, sdcclWindow_t *win,
                                             int winFlags) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
  if (*win == NULL) {
    SDCCLCHECK(sdcclCalloc(win, 1));
  }
  sdcclResult_t res = (sdcclResult_t)ncclCommWindowRegister(
      comm->base, buff, size, &(*win)->base, winFlags);
  if (res == sdcclSuccess) {
    (*win)->winFlags = winFlags;
  }
  return res;
#else
  return sdcclNotSupported;
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
}

sdcclResult_t ncclAdaptorCommWindowDeregister(sdcclInnerComm_t comm,
                                               sdcclWindow_t win) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
  sdcclResult_t res = sdcclSuccess;
  res = (sdcclResult_t)ncclCommWindowDeregister(comm->base, win->base);
  free(win);
  return res;
#else
  return sdcclNotSupported;
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
}

sdcclResult_t ncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclRedOp_t op, int root,
                                 sdcclInnerComm_t comm,
                                 sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

sdcclResult_t ncclAdaptorGather(const void *sendbuff, void *recvbuff,
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

sdcclResult_t ncclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

sdcclResult_t ncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    int root, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

sdcclResult_t ncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclRedOp_t op, sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
#if defined(COMPILE_KERNEL_HOST) &&                                            \
    (NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)) &&                            \
    !defined(NVCC_GENCODE_MULTICAST_UNSUPPORTED)
  size_t size = count * getSdcclDataTypeSize(datatype);
  int nranks;
  SDCCLCHECK((sdcclResult_t)ncclCommCount(comm->base, &nranks));
  if (size >= NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE) {
    SDCCLCHECK((sdcclResult_t)ncclAllReduce(
        sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
        comm->base, stream->base));
  } else {
    DEVCHECK(cudaMemcpyAsync(comm->sendStagedBuff->buff, sendbuff, size,
                             cudaMemcpyDeviceToDevice, stream->base));
    if ((nranks <= 4 && size < 512 * 1024) ||
        (nranks <= 8 && size < 256 * 1024)) {
      SDCCLCHECK((sdcclResult_t)ncclAdaptorLocalAllReduce(
          sendbuff, recvbuff, comm->sendStagedBuff->win,
          comm->recvStagedBuff->win, count, (ncclDataType_t)datatype,
          (ncclRedOp_t)op, *comm->devBase, stream->base));
    } else {
      SDCCLCHECK((sdcclResult_t)ncclAdaptorInterleavedAllReduce(
          sendbuff, recvbuff, comm->sendStagedBuff->win,
          comm->recvStagedBuff->win, count, (ncclDataType_t)datatype,
          (ncclRedOp_t)op, *comm->devBase, stream->base));
      DEVCHECK(cudaMemcpyAsync(recvbuff, comm->recvStagedBuff->buff, size,
                               cudaMemcpyDeviceToDevice, stream->base));
    }
  }
#else
  SDCCLCHECK((sdcclResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base));
#endif
  return sdcclSuccess;
}

sdcclResult_t
ncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         sdcclDataType_t datatype, sdcclRedOp_t op,
                         sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

sdcclResult_t ncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, sdcclDataType_t datatype,
                                    sdcclInnerComm_t comm,
                                    sdcclStream_t stream) {
  return (sdcclResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

sdcclResult_t ncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
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

sdcclResult_t ncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
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

sdcclResult_t ncclAdaptorSend(const void *sendbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ncclAdaptorRecv(void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int peer,
                               sdcclInnerComm_t comm, sdcclStream_t stream) {
  return (sdcclResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

sdcclResult_t ncclAdaptorGroupStart() {
  return (sdcclResult_t)ncclGroupStart();
}

sdcclResult_t ncclAdaptorGroupEnd() { return (sdcclResult_t)ncclGroupEnd(); }

sdcclResult_t ncclAdaptorDevCommCreate(sdcclInnerComm_t comm,
                                        const sdcclDevCommRequirements *reqs,
                                        sdcclInnerDevComm_t *devComm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  sdcclInnerDevComm_t inner =
      (sdcclInnerDevComm_t)malloc(sizeof(struct sdcclInnerDevComm));
  if (!inner)
    return sdcclSystemError;

  // Map generic requirements to NCCL-specific requirements
  ncclDevCommRequirements ncclReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclReqs.lsaBarrierCount = reqs->intraBarrierCount;
  ncclReqs.lsaMultimem = reqs->intraMulticast;
  ncclReqs.barrierCount = reqs->barrierCount;
  ncclReqs.lsaLLA2ABlockCount = reqs->intraLLA2ABlockCount;
  ncclReqs.lsaLLA2ASlotCount = reqs->intraLLA2ASlotCount;
  ncclReqs.railGinBarrierCount = reqs->interBarrierCount;
  ncclReqs.ginSignalCount = reqs->interSignalCount;
  ncclReqs.ginForceEnable = reqs->interForceEnable;
  ncclReqs.ginContextCount = reqs->interContextCount;
  ncclReqs.ginCounterCount = reqs->interCounterCount;

  sdcclResult_t ret =
      ncclDevCommCreateHelper(comm->base, &ncclReqs, &inner->base);
  if (ret != sdcclSuccess) {
    free(inner);
    return ret;
  }

  *devComm = inner;
  comm->devBase = &inner->base;
  return sdcclSuccess;
#else
  return sdcclNotSupported;
#endif
}

sdcclResult_t ncclAdaptorDevCommDestroy(sdcclInnerComm_t comm,
                                         sdcclInnerDevComm_t devComm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (!devComm)
    return sdcclSuccess;
  sdcclResult_t ret = ncclDevCommDestroyHelper(comm->base, &devComm->base);
  free(devComm);
  comm->devBase = NULL;
  return ret;
#else
  return sdcclNotSupported;
#endif
}

struct sdcclCCLAdaptor ncclAdaptor = {
    "NCCL",
    // Basic functions
    ncclAdaptorGetVersion, ncclAdaptorGetUniqueId, ncclAdaptorGetErrorString,
    ncclAdaptorGetLastError, ncclAdaptorGetStagedBuffer,
    // Communicator functions
    ncclAdaptorCommInitRank, ncclAdaptorCommFinalize, ncclAdaptorCommDestroy,
    ncclAdaptorCommAbort, ncclAdaptorCommResume, ncclAdaptorCommSuspend,
    ncclAdaptorCommCount, ncclAdaptorCommCuDevice, ncclAdaptorCommUserRank,
    ncclAdaptorCommGetAsyncError, ncclAdaptorMemAlloc, ncclAdaptorMemFree,
    ncclAdaptorCommRegister, ncclAdaptorCommDeregister,
    // Symmetric functions
    ncclAdaptorCommWindowRegister, ncclAdaptorCommWindowDeregister,
    // Communication functions
    ncclAdaptorReduce, ncclAdaptorGather, ncclAdaptorScatter,
    ncclAdaptorBroadcast, ncclAdaptorAllReduce, ncclAdaptorReduceScatter,
    ncclAdaptorAllGather, ncclAdaptorAlltoAll, ncclAdaptorAlltoAllv,
    ncclAdaptorSend, ncclAdaptorRecv,
    // Group semantics
    ncclAdaptorGroupStart, ncclAdaptorGroupEnd,
    // Device API
    ncclAdaptorDevCommCreate, ncclAdaptorDevCommDestroy};

#endif // USE_NVIDIA_ADAPTOR
