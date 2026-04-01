#include "sdccl.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "cluster.h"
#include "comm.h"
#include "cost_model.h"
#include "sdccl_hetero.h"
#include "sdccl_kernel.h"
#include "sdccl_net.h"
#include "ib_common.h"
#include "launch_kernel.h"
#include "net.h"
#include "onesided.h"
#include "param.h"
#include "proxy.h"
#include "reg_pool.h"
#include "runner.h"
#include "timer.h"
#include "transport.h"
#include "utils.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <unordered_map>

sdcclRegPool globalRegPool;
struct sdcclOneSideHandleInfo
    *globalOneSideHandleTable[SDCCL_MAX_ONE_SIDE_HANDLES] = {};
int globalOneSideHandleCount = 0;
struct sdcclOneSideHandleInfo *globalOneSideSignalHandles = NULL;
struct sdcclOneSideHandleInfo *globalOneSideStagingHandles = NULL;

size_t getSdcclDataTypeSize(sdcclDataType_t dtype) {
  switch (dtype) {
    // case sdcclInt8:
    case sdcclChar:
      return sizeof(char); // 1 byte
    case sdcclUint8:
      return sizeof(unsigned char); // 1 byte
    // case sdcclInt32:
    case sdcclInt:
      return sizeof(int); // 4 bytes
    case sdcclUint32:
      return sizeof(unsigned int); // 4 bytes
    case sdcclInt64:
      return sizeof(long long); // 8 bytes
    case sdcclUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case sdcclFloat16:
    case sdcclHalf:
      return 2; // Half precision float is 2 bytes
    // case sdcclFloat32:
    case sdcclFloat:
      return sizeof(float); // 4 bytes
    // case sdcclFloat64:
    case sdcclDouble:
      return sizeof(double); // 8 bytes
    case sdcclBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      fprintf(stderr, "Unknown sdccl data type\n");
      return 0;
  }
}

// Wrapper function for deviceMemcpy without the usage of invalid args
sdcclResult_t wrapper_deviceMemcpy(void *dst, void *src, size_t size,
                                    sdcclMemcpyType_t type,
                                    sdcclStream_t stream) {
  return deviceAdaptor->deviceMemcpy(dst, src, size, type, stream, NULL);
}

static struct sdcclDeviceHandle globalDeviceHandle {
  // Basic functions
  deviceAdaptor->deviceSynchronize, wrapper_deviceMemcpy,
      deviceAdaptor->deviceMemset, deviceAdaptor->deviceMalloc,
      deviceAdaptor->deviceFree, deviceAdaptor->setDevice,
      deviceAdaptor->getDevice, deviceAdaptor->getDeviceCount,
      deviceAdaptor->getVendor, deviceAdaptor->hostGetDevicePointer,
      // Stream functions
      deviceAdaptor->streamCreate, deviceAdaptor->streamDestroy,
      deviceAdaptor->streamCopy, deviceAdaptor->streamFree,
      deviceAdaptor->streamSynchronize, deviceAdaptor->streamQuery,
      deviceAdaptor->streamWaitEvent,
      // Event functions
      deviceAdaptor->eventCreate, deviceAdaptor->eventDestroy,
      deviceAdaptor->eventRecord, deviceAdaptor->eventSynchronize,
      deviceAdaptor->eventQuery,
      // IpcMemHandle functions
      deviceAdaptor->ipcMemHandleCreate, deviceAdaptor->ipcMemHandleGet,
      deviceAdaptor->ipcMemHandleOpen, deviceAdaptor->ipcMemHandleClose,
      deviceAdaptor->ipcMemHandleFree,
};

void sdcclRebuildGlobalDeviceHandle() {
  // Basic functions
  globalDeviceHandle.deviceSynchronize = deviceAdaptor->deviceSynchronize;
  globalDeviceHandle.deviceMemcpy = wrapper_deviceMemcpy;
  globalDeviceHandle.deviceMemset = deviceAdaptor->deviceMemset;
  globalDeviceHandle.deviceMalloc = deviceAdaptor->deviceMalloc;
  globalDeviceHandle.deviceFree = deviceAdaptor->deviceFree;
  globalDeviceHandle.setDevice = deviceAdaptor->setDevice;
  globalDeviceHandle.getDevice = deviceAdaptor->getDevice;
  globalDeviceHandle.getDeviceCount = deviceAdaptor->getDeviceCount;
  globalDeviceHandle.getVendor = deviceAdaptor->getVendor;
  globalDeviceHandle.hostGetDevicePointer = deviceAdaptor->hostGetDevicePointer;
  // Stream functions
  globalDeviceHandle.streamCreate = deviceAdaptor->streamCreate;
  globalDeviceHandle.streamDestroy = deviceAdaptor->streamDestroy;
  globalDeviceHandle.streamCopy = deviceAdaptor->streamCopy;
  globalDeviceHandle.streamFree = deviceAdaptor->streamFree;
  globalDeviceHandle.streamSynchronize = deviceAdaptor->streamSynchronize;
  globalDeviceHandle.streamQuery = deviceAdaptor->streamQuery;
  globalDeviceHandle.streamWaitEvent = deviceAdaptor->streamWaitEvent;
  // Event functions
  globalDeviceHandle.eventCreate = deviceAdaptor->eventCreate;
  globalDeviceHandle.eventDestroy = deviceAdaptor->eventDestroy;
  globalDeviceHandle.eventRecord = deviceAdaptor->eventRecord;
  globalDeviceHandle.eventSynchronize = deviceAdaptor->eventSynchronize;
  globalDeviceHandle.eventQuery = deviceAdaptor->eventQuery;
  // IpcMemHandle functions
  globalDeviceHandle.ipcMemHandleCreate = deviceAdaptor->ipcMemHandleCreate;
  globalDeviceHandle.ipcMemHandleGet = deviceAdaptor->ipcMemHandleGet;
  globalDeviceHandle.ipcMemHandleOpen = deviceAdaptor->ipcMemHandleOpen;
  globalDeviceHandle.ipcMemHandleClose = deviceAdaptor->ipcMemHandleClose;
  globalDeviceHandle.ipcMemHandleFree = deviceAdaptor->ipcMemHandleFree;
}

sdcclResult_t sdcclEnsureCommReady(sdcclComm_t comm) {
  if (comm == NULL) {
    return sdcclInternalError;
  }
  if (comm->commType != sdcclCommunicatorHybrid &&
      comm->commType != sdcclCommunicatorHomo) {
    return sdcclInternalError;
  }
  return sdcclSuccess;
}

bool useHomoComm(sdcclComm_t comm) {
  return comm->commType == sdcclCommunicatorHomo;
}

bool useHostComm() {
  const char *useHostComm = sdcclGetEnv("SDCCL_USE_HOST_COMM");
  if (useHostComm) {
    return std::stoi(useHostComm) == 1;
  }
  return false;
}

bool useHeteroComm() {
  const char *useHeteroComm = sdcclGetEnv("SDCCL_USE_HETERO_COMM");
  if (useHeteroComm) {
    return std::stoi(useHeteroComm) == 1;
  }
  return false;
}

sdcclResult_t sdcclHandleInit(sdcclHandlerGroup_t *handler) {
  sdcclResult_t res = sdcclSuccess;
  sdcclDeviceAdaptorPluginInit();
  sdcclCCLAdaptorPluginInit();
  (*handler) = NULL;
  SDCCLCHECKGOTO(sdcclCalloc(handler, 1), res, fail);
  SDCCLCHECKGOTO(sdcclCalloc(&(*handler)->uniqueId, 1), res, fail);
  SDCCLCHECKGOTO(sdcclCalloc(&(*handler)->comm, 1), res, fail);
  SDCCLCHECKGOTO(sdcclCalloc(&(*handler)->devHandle, 1), res, fail);
  *(*handler)->devHandle = globalDeviceHandle;
  return sdcclSuccess;

fail:
  if (*handler) {
    free((*handler)->uniqueId);
    free((*handler)->comm);
    free((*handler)->devHandle);
    free(*handler);
    *handler = NULL;
  }
  sdcclCCLAdaptorPluginFinalize();
  sdcclDeviceAdaptorPluginFinalize();
  return res;
}

sdcclResult_t sdcclHandleFree(sdcclHandlerGroup_t handler) {
  if (handler != NULL) {
    free(handler->uniqueId);
    free(handler->comm);
    free(handler->devHandle);
    handler->uniqueId = NULL;
    handler->comm = NULL;
    handler->devHandle = NULL;
    free(handler);
    handler = NULL;
  }
  sdcclCCLAdaptorPluginFinalize();
  sdcclDeviceAdaptorPluginFinalize();
  return sdcclSuccess;
}

SDCCL_PARAM(MemEnable, "MEM_ENABLE", 0);

sdcclResult_t sdcclMemAlloc(void **ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    WARN("Invalid ptr(NULL) or size(0) for allocation.");
    return sdcclInvalidArgument;
  }
  if (sdcclParamMemEnable()) {
    SDCCLCHECK(deviceAdaptor->gdrMemAlloc(ptr, size, NULL));
    if (*ptr != NULL) {
      INFO(SDCCL_REG, "sdcclMemAlloc: GDR allocated [%p, %ld]", *ptr, size);
    } else {
      WARN("sdcclMemAlloc: GDR allocation failed");
      return sdcclUnhandledDeviceError;
    }
  } else {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->memAlloc(ptr, size));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclMemFree(void *ptr) {
  if (ptr == NULL) {
    WARN("Invalid pointer(=NULL) for de-allocation.");
    return sdcclSuccess;
  }
  if (sdcclParamMemEnable()) {
    SDCCLCHECK(deviceAdaptor->gdrMemFree(ptr, NULL));
    INFO(SDCCL_REG, "sdcclMemFree: GDR memory deallocated");
  } else {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->memFree(ptr));
  }
  return sdcclSuccess;
}

// Build full-mesh IB connections (including self-loopback) for one-sided ops.
// Called once on the first sdcclOneSideRegister invocation; stored in
// handle[0]. Pattern aligned with NCCL GIN gin.cc:146-158.
static sdcclResult_t
sdcclOneSideBuildFullMesh(const sdcclComm_t comm,
                           struct sdcclOneSideHandleInfo *info) {
  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  struct bootstrapState *state = heteroComm->bootstrap;
  int nranks = state->nranks;
  int rank = state->rank;
  sdcclResult_t res = sdcclSuccess;

  void *listenComm = NULL;
  sdcclNetHandle_t *allHandles = NULL;

  SDCCLCHECKGOTO(sdcclCalloc(&info->fullSendComms, nranks), res, fail);
  SDCCLCHECKGOTO(sdcclCalloc(&info->fullRecvComms, nranks), res, fail);
  info->nRanks = nranks;

  {
    // 1. Create listen comm and allgather listen handles
    sdcclNetHandle_t myListenHandle = {};
    SDCCLCHECKGOTO(heteroComm->netAdaptor->listen(heteroComm->netDev,
                                                   (void *)myListenHandle,
                                                   &listenComm),
                    res, fail);

    // Allgather listen handles from all ranks
    SDCCLCHECKGOTO(sdcclCalloc(&allHandles, nranks), res, fail_listen);
    memcpy(&allHandles[rank], &myListenHandle, sizeof(sdcclNetHandle_t));
    SDCCLCHECKGOTO(bootstrapAllGather(state, (void *)allHandles,
                                       sizeof(sdcclNetHandle_t)),
                    res, fail_handles);

    // 2. Deadlock-free full-mesh connection (NCCL GIN pattern)
    for (int i = 0; i < nranks; i++) {
      int connectPeer = (rank + i) % nranks; // i=0 → self
      int acceptPeer = (rank - i + nranks) % nranks;

      // Connect to connectPeer + accept from acceptPeer in lockstep
      void *sendComm = NULL, *recvComm = NULL;
      while (sendComm == NULL || recvComm == NULL) {
        if (sendComm == NULL) {
          res = heteroComm->netAdaptor->connect(
              heteroComm->netDev, (void *)&allHandles[connectPeer], &sendComm);
          if (res != sdcclSuccess && res != sdcclInProgress) {
            INFO(SDCCL_REG,
                 "sdcclOneSideBuildFullMesh: connect to peer %d failed, "
                 "res=%d",
                 connectPeer, res);
            goto fail_handles;
          }
        }
        if (recvComm == NULL) {
          res = heteroComm->netAdaptor->accept(listenComm, &recvComm);
          if (res != sdcclSuccess && res != sdcclInProgress) {
            INFO(SDCCL_REG,
                 "sdcclOneSideBuildFullMesh: accept from peer %d failed, "
                 "res=%d",
                 acceptPeer, res);
            goto fail_handles;
          }
        }
        if (sendComm == NULL || recvComm == NULL)
          sched_yield();
      }
      info->fullSendComms[connectPeer] = sendComm;
      info->fullRecvComms[acceptPeer] = recvComm;
      INFO(SDCCL_REG,
           "sdcclOneSideBuildFullMesh: rank %d connected peer %d (i=%d)", rank,
           connectPeer, i);
    }

    free(allHandles);
    heteroComm->netAdaptor->closeListen(listenComm);
  }

  INFO(SDCCL_REG,
       "sdcclOneSideBuildFullMesh: rank %d, %d full-mesh connections "
       "(including self-loopback)",
       rank, nranks);
  return sdcclSuccess;

fail_handles:
  // cleanup partial connections on error
  for (int i = 0; i < nranks; i++) {
    if (info->fullSendComms[i])
      heteroComm->netAdaptor->closeSend(info->fullSendComms[i]);
    if (info->fullRecvComms[i])
      heteroComm->netAdaptor->closeRecv(info->fullRecvComms[i]);
  }
  free(allHandles);
fail_listen:
  heteroComm->netAdaptor->closeListen(listenComm);
fail:
  free(info->fullSendComms);
  free(info->fullRecvComms);
  info->fullSendComms = NULL;
  info->fullRecvComms = NULL;
  info->nRanks = 0;
  return res;
}

sdcclResult_t sdcclOneSideRegister(const sdcclComm_t comm, void *buff,
                                     size_t size) {
  // Check if one-sided operations are enabled
  if (useHomoComm(comm) && !useHeteroComm()) {
    return sdcclSuccess;
  }

  // Check for duplicate registration of the same buffer
  for (int i = 0; i < globalOneSideHandleCount; i++) {
    if (globalOneSideHandleTable[i] != NULL &&
        globalOneSideHandleTable[i]->baseVas != NULL &&
        globalOneSideHandleTable[i]->baseVas[comm->rank] == (uintptr_t)buff) {
      INFO(SDCCL_REG,
           "sdcclOneSideRegister: buffer %p already registered at slot %d",
           buff, i);
      return sdcclSuccess;
    }
  }

  if (globalOneSideHandleCount >= SDCCL_MAX_ONE_SIDE_HANDLES) {
    WARN("sdcclOneSideRegister: handle table full (%d/%d)",
         globalOneSideHandleCount, SDCCL_MAX_ONE_SIDE_HANDLES);
    return sdcclNotSupported;
  }

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iput == NULL ||
      heteroComm->netAdaptor->regMr == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideRegister: heteroComm is NULL");
    return sdcclSuccess;
  }

  struct bootstrapState *state = heteroComm->bootstrap;
  if (state == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideRegister: state is NULL");
    return sdcclNotSupported;
  }

  sdcclResult_t res = sdcclSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct sdcclOneSideHandleInfo *info = NULL;
  int slot = globalOneSideHandleCount;
  bool isFirstHandle = (slot == 0);

  SDCCLCHECKGOTO(sdcclCalloc(&info, 1), res, fail);

  // First handle: build full-mesh IB connections (including self-loopback)
  if (isFirstHandle) {
    SDCCLCHECKGOTO(sdcclOneSideBuildFullMesh(comm, info), res, fail_info);
  }

  // Use self recvComm for MR registration (PD match)
  {
    void *selfRecvComm =
        isFirstHandle ? info->fullRecvComms[state->rank]
                      : globalOneSideHandleTable[0]->fullRecvComms[state->rank];
    info->localRecvComm = selfRecvComm;
    regComm = selfRecvComm;
    if (heteroComm->netAdaptor->name &&
        strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
      struct sdcclIbRecvComm *ibRecvComm = (struct sdcclIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
  }

  // Register MR for this buffer
  {
    int type = SDCCL_PTR_CUDA;
    res = heteroComm->netAdaptor->regMr(regComm, buff, size, type,
                                        SDCCL_NET_MR_FLAG_NONE, &mrHandle);
  }
  if (res != sdcclSuccess || mrHandle == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideRegister: regMr failed, res=%d", res);
    res = sdcclNotSupported;
    goto fail_mesh;
  }

  {
    struct sdcclIbMrHandle *localMrHandle =
        (struct sdcclIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  // Allgather MR info
  {
    int nranks = state->nranks;
    SDCCLCHECKGOTO(sdcclCalloc(&info->baseVas, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->rkeys, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[state->rank] = (uintptr_t)buff;
    info->rkeys[state->rank] = mr->rkey;
    info->lkeys[state->rank] = mr->lkey;
    info->localMrHandle = mrHandle;

    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->baseVas, sizeof(uintptr_t)),
        res, fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->rkeys, sizeof(uint32_t)), res,
        fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->lkeys, sizeof(uint32_t)), res,
        fail_mr);

    globalOneSideHandleTable[slot] = info;
    globalOneSideHandleCount = slot + 1;

    INFO(SDCCL_REG,
         "One-sided register slot %d allgather results (rank %d, nranks %d):",
         slot, state->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(SDCCL_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  return sdcclSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
  }
  if (regComm && mrHandle)
    heteroComm->netAdaptor->deregMr(regComm, mrHandle);
fail_mesh:
  if (isFirstHandle) {
    // Clean up full-mesh connections on first-handle failure
    for (int i = 0; i < state->nranks; i++) {
      if (info->fullSendComms && info->fullSendComms[i])
        heteroComm->netAdaptor->closeSend(info->fullSendComms[i]);
      if (info->fullRecvComms && info->fullRecvComms[i])
        heteroComm->netAdaptor->closeRecv(info->fullRecvComms[i]);
    }
    free(info->fullSendComms);
    free(info->fullRecvComms);
  }
fail_info:
  free(info);
fail:
  return res;
}

sdcclResult_t sdcclOneSideDeregister(const sdcclComm_t comm) {
  if (comm == NULL)
    return sdcclInternalError;
  struct sdcclHeteroComm *heteroComm = comm->heteroComm;

  // Deregister all handles in reverse order
  for (int slot = globalOneSideHandleCount - 1; slot >= 0; slot--) {
    struct sdcclOneSideHandleInfo *info = globalOneSideHandleTable[slot];
    if (info == NULL)
      continue;

    if (heteroComm != NULL && heteroComm->netAdaptor != NULL) {
      // Deregister MR
      if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
        void *regComm = info->localRecvComm;
        if (heteroComm->netAdaptor->name &&
            strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
          struct sdcclIbRecvComm *ibRecvComm =
              (struct sdcclIbRecvComm *)regComm;
          regComm = (void *)&ibRecvComm->base;
        }
        heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
      }

      // Close full-mesh connections (only stored in slot 0)
      if (slot == 0 && info->fullSendComms != NULL) {
        for (int i = 0; i < info->nRanks; i++) {
          if (info->fullSendComms[i])
            heteroComm->netAdaptor->closeSend(info->fullSendComms[i]);
          if (info->fullRecvComms[i])
            heteroComm->netAdaptor->closeRecv(info->fullRecvComms[i]);
        }
        free(info->fullSendComms);
        free(info->fullRecvComms);
      }
    }

    free(info->baseVas);
    free(info->rkeys);
    free(info->lkeys);
    free(info);
    globalOneSideHandleTable[slot] = NULL;
  }
  globalOneSideHandleCount = 0;
  return sdcclSuccess;
}

sdcclResult_t sdcclOneSideSignalRegister(const sdcclComm_t comm, void *buff,
                                           size_t size) {
  if (useHomoComm(comm) && !useHeteroComm()) {
    return sdcclSuccess;
  }

  if (globalOneSideSignalHandles != NULL) {
    if (globalOneSideSignalHandles->baseVas != NULL &&
        globalOneSideSignalHandles->baseVas[comm->rank] != (uintptr_t)buff) {
      WARN("sdcclOneSideSignalRegister: already registered with a different "
           "buffer");
    }
    return sdcclSuccess;
  }

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iputSignal == NULL ||
      heteroComm->netAdaptor->regMr == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideSignalRegister: heteroComm is NULL");
    return sdcclSuccess;
  }

  struct bootstrapState *state = heteroComm->bootstrap;
  if (state == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideSignalRegister: state is NULL");
    return sdcclNotSupported;
  }

  // Signal registration reuses full-mesh connections from data handle table.
  // Requires at least one data handle to be registered first.
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullRecvComms == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideSignalRegister: no full-mesh connections, "
                     "register a data buffer first");
    return sdcclNotSupported;
  }

  sdcclResult_t res = sdcclSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct sdcclOneSideHandleInfo *info = NULL;

  // Use self recvComm from data handle[0] for MR registration (PD match)
  void *selfRecvComm = globalOneSideHandleTable[0]->fullRecvComms[state->rank];
  regComm = selfRecvComm;
  if (heteroComm->netAdaptor->name &&
      strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
    struct sdcclIbRecvComm *ibRecvComm = (struct sdcclIbRecvComm *)regComm;
    regComm = (void *)&ibRecvComm->base;
  }

  {
    int type = SDCCL_PTR_CUDA;
    res = heteroComm->netAdaptor->regMr(regComm, buff, size, type,
                                        SDCCL_NET_MR_FLAG_FORCE_SO, &mrHandle);
  }
  if (res != sdcclSuccess || mrHandle == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideSignalRegister: regMr failed, res=%d", res);
    return sdcclNotSupported;
  }

  {
    struct sdcclIbMrHandle *localMrHandle =
        (struct sdcclIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  {
    int nranks = state->nranks;
    SDCCLCHECKGOTO(sdcclCalloc(&info, 1), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->baseVas, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->rkeys, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[state->rank] = (uintptr_t)buff;
    info->rkeys[state->rank] = mr->rkey;
    info->lkeys[state->rank] = mr->lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;

    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->baseVas, sizeof(uintptr_t)),
        res, fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->rkeys, sizeof(uint32_t)), res,
        fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->lkeys, sizeof(uint32_t)), res,
        fail_mr);
    globalOneSideSignalHandles = info;
    INFO(SDCCL_REG,
         "Signal register allgather results (rank %d, nranks %d):", state->rank,
         nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(SDCCL_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  return sdcclSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  heteroComm->netAdaptor->deregMr(regComm, mrHandle);
  return res;
}

sdcclResult_t sdcclOneSideSignalDeregister(const sdcclComm_t comm) {
  struct sdcclOneSideHandleInfo *info = globalOneSideSignalHandles;
  if (info == NULL)
    return sdcclSuccess;
  if (comm == NULL)
    return sdcclInternalError;

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm != NULL && heteroComm->netAdaptor != NULL) {
    // Deregister MR (connections are shared with data handle table, not owned)
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct sdcclIbRecvComm *ibRecvComm =
            (struct sdcclIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
    // No closeSend/closeRecv — connections owned by data handle table[0]
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
  free(info);
  globalOneSideSignalHandles = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclOneSideStagingRegister(const sdcclComm_t comm, void *buff,
                                            size_t size) {
  if (useHomoComm(comm) && !useHeteroComm()) {
    return sdcclSuccess;
  }

  if (globalOneSideStagingHandles != NULL) {
    if (globalOneSideStagingHandles->baseVas != NULL &&
        globalOneSideStagingHandles->baseVas[comm->rank] != (uintptr_t)buff) {
      WARN("sdcclOneSideStagingRegister: already registered with a different "
           "buffer");
    }
    return sdcclSuccess;
  }

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iput == NULL ||
      heteroComm->netAdaptor->regMr == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideStagingRegister: heteroComm is NULL");
    return sdcclSuccess;
  }

  struct bootstrapState *state = heteroComm->bootstrap;
  if (state == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideStagingRegister: state is NULL");
    return sdcclNotSupported;
  }

  // Staging registration reuses full-mesh connections from data handle table.
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullRecvComms == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideStagingRegister: no full-mesh connections, "
                     "register a data buffer first");
    return sdcclNotSupported;
  }

  sdcclResult_t res = sdcclSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct sdcclOneSideHandleInfo *info = NULL;

  // Use self recvComm from data handle[0] for MR registration (PD match)
  void *selfRecvComm = globalOneSideHandleTable[0]->fullRecvComms[state->rank];
  regComm = selfRecvComm;
  if (heteroComm->netAdaptor->name &&
      strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
    struct sdcclIbRecvComm *ibRecvComm = (struct sdcclIbRecvComm *)regComm;
    regComm = (void *)&ibRecvComm->base;
  }

  {
    int type = SDCCL_PTR_HOST;
    res =
        heteroComm->netAdaptor->regMr(regComm, buff, size, type, 0, &mrHandle);
  }
  if (res != sdcclSuccess || mrHandle == NULL) {
    INFO(SDCCL_REG, "sdcclOneSideStagingRegister: regMr failed, res=%d", res);
    return sdcclNotSupported;
  }

  {
    struct sdcclIbMrHandle *localMrHandle =
        (struct sdcclIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  {
    int nranks = state->nranks;
    SDCCLCHECKGOTO(sdcclCalloc(&info, 1), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->baseVas, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->rkeys, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[state->rank] = (uintptr_t)buff;
    info->rkeys[state->rank] = mr->rkey;
    info->lkeys[state->rank] = mr->lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;

    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->baseVas, sizeof(uintptr_t)),
        res, fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->rkeys, sizeof(uint32_t)), res,
        fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->lkeys, sizeof(uint32_t)), res,
        fail_mr);
    globalOneSideStagingHandles = info;
    INFO(SDCCL_REG, "Staging register allgather results (rank %d, nranks %d):",
         state->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(SDCCL_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  return sdcclSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  heteroComm->netAdaptor->deregMr(regComm, mrHandle);
  return res;
}

sdcclResult_t sdcclOneSideStagingDeregister(const sdcclComm_t comm) {
  struct sdcclOneSideHandleInfo *info = globalOneSideStagingHandles;
  if (info == NULL)
    return sdcclSuccess;
  if (comm == NULL)
    return sdcclInternalError;

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm != NULL && heteroComm->netAdaptor != NULL) {
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct sdcclIbRecvComm *ibRecvComm =
            (struct sdcclIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
  free(info);
  globalOneSideStagingHandles = NULL;
  return sdcclSuccess;
}

sdcclResult_t
sdcclOneSideBarrierRegister(const sdcclComm_t comm, void *recvComm,
                             void *buff, size_t size,
                             struct sdcclOneSideHandleInfo **outInfo) {
  if (comm == NULL || outInfo == NULL)
    return sdcclInvalidArgument;
  *outInfo = NULL;

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->regMr == NULL)
    return sdcclNotSupported;

  struct bootstrapState *state = comm->bootstrap;
  if (state == NULL)
    return sdcclNotSupported;

  struct sdcclNetAdaptor *net = heteroComm->netAdaptor;
  sdcclResult_t res = sdcclSuccess;
  void *mrHandle = NULL;
  uint32_t rkey = 0, lkey = 0;
  uintptr_t baseVa = 0;
  struct sdcclOneSideHandleInfo *info = NULL;

  // Leaders (recvComm != NULL): register MR and extract keys
  if (recvComm != NULL && buff != NULL && size > 0) {
    void *regComm = recvComm;
    if (net->name && strcmp(net->name, "IB") == 0) {
      struct sdcclIbRecvComm *ibRecvComm = (struct sdcclIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
    res = net->regMr(regComm, buff, size, SDCCL_PTR_HOST,
                     SDCCL_NET_MR_FLAG_FORCE_SO, &mrHandle);
    if (res != sdcclSuccess || mrHandle == NULL) {
      INFO(SDCCL_REG, "sdcclOneSideBarrierRegister: regMr failed, res=%d",
           res);
      return sdcclNotSupported;
    }
    struct sdcclIbMrHandle *ibMrHandle = (struct sdcclIbMrHandle *)mrHandle;
    struct ibv_mr *mr = ibMrHandle->mrs[0];
    rkey = mr->rkey;
    lkey = mr->lkey;
    baseVa = (uintptr_t)buff;
  }

  // ALL ranks: allocate info, populate own entry, AllGather
  {
    int nranks = state->nranks;
    int myRank = state->rank;
    SDCCLCHECKGOTO(sdcclCalloc(&info, 1), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->baseVas, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->rkeys, nranks), res, fail_mr);
    SDCCLCHECKGOTO(sdcclCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[myRank] = baseVa;
    info->rkeys[myRank] = rkey;
    info->lkeys[myRank] = lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = recvComm;

    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->baseVas, sizeof(uintptr_t)),
        res, fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->rkeys, sizeof(uint32_t)), res,
        fail_mr);
    SDCCLCHECKGOTO(
        bootstrapAllGather(state, (void *)info->lkeys, sizeof(uint32_t)), res,
        fail_mr);

    INFO(SDCCL_REG,
         "Barrier register allgather results (rank %d, nranks %d):", myRank,
         nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(SDCCL_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  *outInfo = info;
  return sdcclSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  if (mrHandle != NULL) {
    void *regComm = recvComm;
    if (net->name && strcmp(net->name, "IB") == 0) {
      struct sdcclIbRecvComm *ibRecvComm = (struct sdcclIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
    net->deregMr(regComm, mrHandle);
  }
  return res;
}

sdcclResult_t
sdcclOneSideBarrierDeregister(const sdcclComm_t comm,
                               struct sdcclOneSideHandleInfo *info) {
  if (info == NULL)
    return sdcclSuccess;
  if (comm == NULL)
    return sdcclInternalError;

  struct sdcclHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm != NULL && heteroComm->netAdaptor != NULL) {
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct sdcclIbRecvComm *ibRecvComm =
            (struct sdcclIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
  free(info);
  return sdcclSuccess;
}

sdcclResult_t sdcclCommRegister(const sdcclComm_t comm, void *buff,
                                  size_t size, void **handle) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));

  if (buff == NULL || size == 0) {
    WARN("Invalid buffer or size for buffer registration.");
    return sdcclInvalidArgument;
  }

  // Step 1: Register in globalRegPool (both paths)
  // Key: heteroComm if available (p2p/net downstream use it), else homoComm
  void *regKey =
      comm->heteroComm ? (void *)comm->heteroComm : (void *)comm->homoComm;
  globalRegPool.registerBuffer(regKey, buff, size);
  sdcclRegItem *regItem = globalRegPool.getItem(regKey, buff);

  *handle = reinterpret_cast<void *>(regItem);

  // Re-registration: backend handle + IPC handle already set up
  if (regItem->refCount > 1) {
    return sdcclSuccess;
  }

  sdcclResult_t res = sdcclSuccess;

  // Step 2a: Homo path — backend CCL registration
  // NCCL handles IPC/VMM internally via ncclCommRegister, so skip Step 2b
  // (cudaIpcGetMemHandle is incompatible with ncclMemAlloc VMM buffers)
  // and Step 3 (one-sided MR registration, hetero-only).
  if (useHomoComm(comm) && !useHeteroComm()) {
    void *homoHandle = nullptr;
    res = cclAdaptors[sdcclCCLAdaptorDevice]->commRegister(
        comm->homoComm, buff, size, &homoHandle);
    if (res != sdcclSuccess)
      goto fail;
    regItem->homoRegHandle = homoHandle;
    return sdcclSuccess;
  }

  // Step 2b: Create IPC handle for the buffer (hetero path only)
  {
    sdcclIpcMemHandle_t handlePtr = nullptr;
    size_t ipcSize = 0;
    res = deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize);
    if (res != sdcclSuccess)
      goto fail;
    res = deviceAdaptor->ipcMemHandleGet(handlePtr, buff);
    if (res != sdcclSuccess) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      goto fail;
    }
    if (ipcSize > sizeof(sdcclIpcHandleData)) {
      deviceAdaptor->ipcMemHandleFree(handlePtr);
      res = sdcclInternalError;
      goto fail;
    }
    memcpy(&regItem->ipcHandleData, handlePtr, ipcSize);
    deviceAdaptor->ipcMemHandleFree(handlePtr);
  }

  // Step 3: One-sided MR registration (hetero path only)
  {
    sdcclResult_t regRes = sdcclOneSideRegister(comm, buff, size);
    if (regRes != sdcclSuccess) {
      INFO(SDCCL_REG, "sdcclCommRegister: one-sided register skipped (%d)",
           regRes);
    }
  }

  return sdcclSuccess;

fail:
  // Undo Step 2a
  if (useHomoComm(comm) && !useHeteroComm() && regItem->homoRegHandle) {
    cclAdaptors[sdcclCCLAdaptorDevice]->commDeregister(comm->homoComm,
                                                        regItem->homoRegHandle);
    regItem->homoRegHandle = nullptr;
  }
  // Undo Step 1
  globalRegPool.deregisterBuffer(regKey, regItem);
  *handle = nullptr;
  return res;
}

sdcclResult_t sdcclCommDeregister(const sdcclComm_t comm, void *handle) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (handle == nullptr)
    return sdcclSuccess;
  sdcclRegItem *regItem = reinterpret_cast<sdcclRegItem *>(handle);

  // Backend-specific deregistration (homo path only, last ref only)
  if (regItem->refCount == 1) {
    if (useHomoComm(comm) && !useHeteroComm() && regItem->homoRegHandle) {
      cclAdaptors[sdcclCCLAdaptorDevice]->commDeregister(
          comm->homoComm, regItem->homoRegHandle);
    }
  }

  // Clean up globalRegPool (both paths)
  void *regKey =
      comm->heteroComm ? (void *)comm->heteroComm : (void *)comm->homoComm;
  globalRegPool.deregisterBuffer(regKey, handle);
  return sdcclSuccess;
}

sdcclResult_t sdcclCommWindowRegister(sdcclComm_t comm, void *buff,
                                        size_t size, sdcclWindow_t *win,
                                        int winFlags) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm) && !useHeteroComm()) {
    sdcclResult_t res =
        cclAdaptors[sdcclCCLAdaptorDevice]->commWindowRegister(
            comm->homoComm, buff, size, win, winFlags);
    if (res == sdcclSuccess) {
      return sdcclSuccess;
    }
    if (res != sdcclNotSupported) {
      return res;
    }
    WARN("sdcclCommWindowRegister: backend returned %d, window not available, "
         "falling back",
         res);
  }
  *win = nullptr;
  return sdcclSuccess;
}

sdcclResult_t sdcclCommWindowDeregister(sdcclComm_t comm,
                                          sdcclWindow_t win) {
  if (win == nullptr) {
    return sdcclSuccess;
  }
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm) && !useHeteroComm()) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commWindowDeregister(
        comm->homoComm, win));
    return sdcclSuccess;
  }
  return sdcclNotSupported;
}

sdcclResult_t sdcclIsHomoComm(sdcclComm_t comm, int *isHomo) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    *isHomo = 1;
  } else {
    *isHomo = 0;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclGetVersion(int *version) {
  // TODO: implement a method to retrieve global verison
  return sdcclHeteroGetVersion(version);
}

sdcclResult_t sdcclGetUniqueId(sdcclUniqueId_t *uniqueId) {
  (*uniqueId) = NULL;
  sdcclCalloc(uniqueId, 1);

  // Init bootstrap net
  SDCCLCHECK(bootstrapNetInit());

  // Init uniqueId using bootstrap
  struct sdcclBootstrapHandle handle;
  SDCCLCHECK(bootstrapGetUniqueId(&handle));
  // sdcclUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset((void *)*uniqueId, 0, sizeof(**uniqueId));
  // copy to avoid alignment mismatch
  memcpy((void *)*uniqueId, &handle, sizeof(handle));
  return sdcclSuccess;
}

const char *sdcclGetErrorString(sdcclResult_t result) {
  // TODO: implement a method to retrieve error string
  return "Not implemented.";
}

const char *sdcclGetLastError(sdcclComm_t comm) {
  // TODO: implement a method to retrieve last error string
  if (comm == NULL) {
    return "Undefined: sdcclComm is not fully initialized.";
  }
  if (useHomoComm(comm)) {
    return cclAdaptors[sdcclCCLAdaptorDevice]->getLastError(comm->homoComm);
  }
  return "Not implemented.";
}

sdcclResult_t sdcclCommInitRank(sdcclComm_t *comm, int nranks,
                                  sdcclUniqueId_t commId, int rank) {
  if (nranks < 1 || rank < 0 || rank >= nranks) {
    WARN("Invalid rank requested : %d/%d", rank, nranks);
    return sdcclInvalidArgument;
  }

  (*comm) = NULL;
  sdcclCalloc(comm, 1);
  (*comm)->rank = rank;
  (*comm)->nranks = nranks;
  (*comm)->nclusters = -1;
  (*comm)->homoRank = -1;
  (*comm)->homoRootRank = -1;
  (*comm)->homoRanks = -1;
  (*comm)->hasSingleRankHomoComm = -1;
  (*comm)->magic = 0;
  (*comm)->abortFlag = 0;
  (*comm)->bootstrap = NULL;
  (*comm)->localRank = 0;
  (*comm)->localRanks = 1;
  (*comm)->localRankToRank = NULL;
  (*comm)->hostComm = NULL;
  (*comm)->homoComm = NULL;
  (*comm)->heteroComm = NULL;
  (*comm)->clusterIds = NULL;
  (*comm)->clusterSizes = NULL;
  (*comm)->clusterInterRanks = NULL;
  (*comm)->globalRank2HomoRank = NULL;
  (*comm)->commType = sdcclCommunicatorUnknown;
  (*comm)->homoInterRootRank = -1;
  (*comm)->homoInterMyRank = -1;
  (*comm)->homoInterRanks = -1;
  (*comm)->homoInterComm = NULL;
  (*comm)->c2cSchedule = NULL;

  struct bootstrapState *state = NULL;
  SDCCLCHECK(sdcclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = (*comm)->abortFlag;
  (*comm)->bootstrap = state;
  state->magic = ((struct sdcclBootstrapHandle *)commId)->magic;
  (*comm)->magic = ((struct sdcclBootstrapHandle *)commId)->magic;

  // Init bootstrap net
  SDCCLCHECK(bootstrapNetInit());

  // Init bootstrap state
  SDCCLCHECK(bootstrapInit((struct sdcclBootstrapHandle *)commId, state));

  // Ready to detect heterogeneous/homogeneous communicator
  // Use bootstrap allgather to exchange Device info
  sdcclVendor *vendorData =
      NULL; // temp data used for device vendor gather operation.

  // Get current gpu vendor
  sdcclVendor vendor;
  deviceAdaptor->getVendor(vendor.internal);
  SDCCLCHECK(sdcclCalloc(&vendorData, nranks));
  memcpy(vendorData + rank, &vendor, sizeof(sdcclVendor));
  SDCCLCHECK(
      bootstrapAllGather(state, (void *)vendorData, sizeof(sdcclVendor)));
  SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));

  // Compute intra-node topology using hostHash
  {
    uint64_t myHash = getHostHash();
    uint64_t *hostHashes = nullptr;
    SDCCLCHECK(sdcclCalloc(&hostHashes, nranks));
    hostHashes[rank] = myHash;
    SDCCLCHECK(bootstrapAllGather(state, hostHashes, sizeof(uint64_t)));
    SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));

    int localCount = 0;
    for (int r = 0; r < nranks; r++) {
      if (hostHashes[r] == myHash)
        localCount++;
    }
    (*comm)->localRanks = localCount;

    SDCCLCHECK(sdcclCalloc(&(*comm)->localRankToRank, localCount));
    int lr = 0;
    for (int r = 0; r < nranks; r++) {
      if (hostHashes[r] == myHash) {
        (*comm)->localRankToRank[lr] = r;
        if (r == rank)
          (*comm)->localRank = lr;
        lr++;
      }
    }
    free(hostHashes);
    INFO(SDCCL_INIT, "Intra-node topology: localRank=%d localRanks=%d",
         (*comm)->localRank, (*comm)->localRanks);
  }

  // Init cluster info
  int *globalRankToHomoRankData;
  int *clusterIdData;
  int *clusterInterRankData;
  SDCCLCHECK(sdcclCalloc(&globalRankToHomoRankData, nranks));
  SDCCLCHECK(sdcclCalloc(&clusterIdData, nranks));
  SDCCLCHECK(sdcclCalloc(&clusterInterRankData, nranks));
  SDCCLCHECK(sdcclCollectClusterInfos(
      vendorData, &(*comm)->commType, globalRankToHomoRankData + rank,
      &(*comm)->homoRootRank, &(*comm)->homoRanks, clusterIdData + rank,
      clusterInterRankData + rank, &(*comm)->nclusters, rank, nranks));
  SDCCLCHECK(
      bootstrapAllGather(state, (void *)globalRankToHomoRankData, sizeof(int)));
  SDCCLCHECK(bootstrapAllGather(state, (void *)clusterIdData, sizeof(int)));
  SDCCLCHECK(
      bootstrapAllGather(state, (void *)clusterInterRankData, sizeof(int)));
  SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));
  (*comm)->homoRank = globalRankToHomoRankData[rank];
  (*comm)->clusterIds = clusterIdData;
  (*comm)->globalRank2HomoRank = globalRankToHomoRankData;

  // fill clusterVendorMap
  SDCCLCHECK(sdcclFillClusterVendorInfo(vendorData, (*comm), clusterIdData,
                                          nranks, (*comm)->nclusters));

  int *clusterSizes;
  int *clusterInterRanks;
  SDCCLCHECK(sdcclCalloc(&clusterSizes, (*comm)->nclusters));
  SDCCLCHECK(sdcclCalloc(&clusterInterRanks, (*comm)->nclusters));
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    clusterInterRanks[i] = -1;
  }

  int cid = 0;
  int sum = 0;
  for (int i = 0; i < nranks; ++i) {
    if (clusterIdData[i] == cid + 1) {
      clusterSizes[cid] = i - sum;
      cid += 1;
      sum = i;
    }
  }
  clusterSizes[cid] = nranks - sum;
  (*comm)->clusterSizes = clusterSizes;

  for (int i = 0; i < nranks; ++i) {
    if (clusterInterRankData[i] != -1) {
      clusterInterRanks[clusterIdData[i]] = clusterInterRankData[i];
    }
  }
  (*comm)->clusterInterRanks = clusterInterRanks;

  int start = 0;
  if (clusterIdData[rank] >= 1) {
    for (int i = 0; i < clusterIdData[rank]; ++i) {
      start += clusterSizes[i];
    }
  }

  // Build c2cSchedule
  SDCCLCHECK(sdcclCalloc(&(*comm)->c2cSchedule, (*comm)->nclusters));
  int nLocals = (*comm)->nclusters;
  int local = (*comm)->clusterIds[rank];

  int nLocalsPow2 = pow2Up(nLocals);
  uint32_t localRound = 0;
  uint32_t localDelta = 0;
  int round = 0;
  do {
    if ((int)localDelta < nLocals) { // Filter nonsensical local deltas
      int sendLocal = (local + localDelta) % nLocals;
      int recvLocal = (local - localDelta + nLocals) % nLocals;
      (*comm)->c2cSchedule[round].sendCluster = sendLocal;
      (*comm)->c2cSchedule[round].recvCluster = recvLocal;
      round += 1;
    }
    localRound += 1;
    // Quadratic update
    localDelta = (localDelta + localRound) & (nLocalsPow2 - 1);
  } while (localRound != (uint32_t)nLocalsPow2);
  for (int i = 0; i < round; ++i) {
    INFO(SDCCL_INIT,
         "cluster %d c2cSchedule[%d] sendCluster %d recvCluster %d", local, i,
         (*comm)->c2cSchedule[i].sendCluster,
         (*comm)->c2cSchedule[i].recvCluster);
  }

  // Update comm hasSingleRankHomoComm
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    if ((*comm)->clusterSizes[i] == 1) {
      (*comm)->hasSingleRankHomoComm = 1;
    }
  }
  if ((*comm)->hasSingleRankHomoComm == -1) {
    (*comm)->hasSingleRankHomoComm = 0;
  }
  if ((*comm)->hasSingleRankHomoComm == 1 && useHomoComm(*comm)) {
    // no need to record it for homo comm
    (*comm)->hasSingleRankHomoComm = 0;
  }

  sdcclUniqueId *uniqueIdData;
  SDCCLCHECK(sdcclCalloc(&uniqueIdData, nranks));

  // Tuner init
  bool useTuner = false;
  const char *useTunerEnv = sdcclGetEnv("SDCCL_USE_TUNER");
  if (useTunerEnv) {
    useTuner = (std::stoi(useTunerEnv) == 1) ? true : false;
  }
  INFO(SDCCL_INIT, "Sdccl USE_TUNER flag set to %d", useTuner);
  if (useTuner) {
    (*comm)->tuner = &internalTuner;
    (*comm)->commId = commId;
    (*comm)->uniqueIdData = uniqueIdData;
    (*comm)->tunerInnerComm = NULL;
    (*comm)->isTunningComm = false;
    (*comm)->isTuningWithFlagscale = false;
    (*comm)->isUseSingleTunerComm = false;
    bool isTuningWithFlagscale = false;
    const char *isTuningWithFlagscaleEnv =
        sdcclGetEnv("SDCCL_TUNING_WITH_FLAGSCALE");
    if (isTuningWithFlagscaleEnv) {
      isTuningWithFlagscale =
          (std::stoi(isTuningWithFlagscaleEnv) == 1) ? true : false;
    }
    (*comm)->isTuningWithFlagscale = isTuningWithFlagscale;

    bool isUseSingleTunerComm = false;
    const char *isUseSingleTunerCommEnv =
        sdcclGetEnv("TUNNING_WITH_SINGLE_COMM");

    if (isUseSingleTunerCommEnv) {
      isUseSingleTunerComm =
          (std::stoi(isUseSingleTunerCommEnv) == 1) ? true : false;
    }
    (*comm)->isUseSingleTunerComm = isUseSingleTunerComm;

    SDCCLCHECK((*comm)->tuner->init((*comm)->nranks, (*comm)->rank,
                                     sdcclDebugLog, &((*comm)->tunerContext),
                                     state));
    uint32_t nConfigs = 0;
    SDCCLCHECK(
        (*comm)->tuner->getCandidateNumber((*comm)->tunerContext, &nConfigs));
    if (nConfigs < 1) {
      WARN("Tuner returned 0 candidates, at least 1 is required.");
      return sdcclInternalError;
    }
    (*comm)->homoCommMap.clear();
    (*comm)->homoBestCommMap.clear();
    (*comm)->commMap.clear();

    if (!isUseSingleTunerComm) {
      // Note: The tuner only support homo comm optimization for now
      for (uint32_t i = 0; i < nConfigs; ++i) {
        struct sdcclCommTag tag = {""};
        SDCCLCHECK(
            (*comm)->tuner->setCandidate((*comm)->tunerContext, i, &tag));
        INFO(SDCCL_INIT | SDCCL_TUNING,
             "start to prepare communicator tag=%s(%u/%u)", tag.tag, i,
             nConfigs);

        sdcclInnerComm_t innerComm = NULL;
        SDCCLCHECK(
            sdcclHomoCommInit(commId, uniqueIdData, state, *comm, &innerComm));
        // Insert item into commMap
        (*comm)->commMap[tag] = innerComm;
        // For backward compatible, also assign homo_comm field.
        (*comm)->homoComm = innerComm;
      }
    }

    if (isTuningWithFlagscale) {
      // Create a default communicator based on the default config
      sdcclInnerComm_t innerComm = NULL;
      SDCCLCHECK(
          sdcclHomoCommInit(commId, uniqueIdData, state, *comm, &innerComm));
      // Insert item into homoCommMap
      (*comm)->tunerInnerComm = innerComm;
      // For backward compatible, also assign homoComm field.
      (*comm)->homoComm = innerComm;
    }
  } else {
    (*comm)->tuner = NULL;
    SDCCLCHECK(sdcclHomoCommInit(commId, uniqueIdData, state, *comm,
                                   &((*comm)->homoComm)));
  }

  if (!useHomoComm(*comm) || useHeteroComm()) {
    // Reset commId and hetero root rank calls sdcclHeteroGetUniqueId
    memset((void *)commId, 0, sizeof(sdcclUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(sdcclUniqueId));
    if (rank == 0) {
      sdcclHeteroGetUniqueId(commId);
      memcpy((void *)&uniqueIdData[0], (void *)commId, sizeof(sdcclUniqueId));
    }
    SDCCLCHECK(bootstrapAllGather(state, (void *)uniqueIdData,
                                   sizeof(sdcclUniqueId)));
    SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));

    memcpy((void *)commId, (void *)&uniqueIdData[0], sizeof(sdcclUniqueId));
    // call sdcclHeteroCommInitRank
    SDCCLCHECK(
        sdcclHeteroCommInitRank(&(*comm)->heteroComm, nranks, *commId, rank));

    // Init host cclAdaptor
    if (useHostComm() || (*comm)->hasSingleRankHomoComm) {
      if (!sdcclParamTopoDetectionDisable()) {
        SDCCLCHECK((*comm)->heteroComm->netAdaptor->getProperties(
            (*comm)->heteroComm->netDev, state->properties));
      }
      SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->commInitRank(
          &(*comm)->hostComm, nranks, commId, rank, state));
    }
  }

  if ((!useHomoComm(*comm) || useHeteroComm()) && !useHostComm()) {
    // Experimental for multi-nic support
    // Collect nic distance to ranks
    (*comm)->clusterInterRankList.resize((*comm)->nclusters);
    struct sdcclNicDistance *nicDistanceData;
    SDCCLCHECK(sdcclCalloc(&nicDistanceData, nranks));
    SDCCLCHECK(sdcclGetNicDistance((*comm)->heteroComm->topoServer, rank,
                                     nicDistanceData + rank));
    SDCCLCHECK(bootstrapAllGather(state, (void *)nicDistanceData,
                                   sizeof(sdcclNicDistance)));
    SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));
    for (int i = 0; i < (*comm)->nclusters; ++i) {
      int minDistance = INT_MAX;
      std::unordered_map<int, std::vector<int>> nicDistanceToRanks;
      std::unordered_map<int, std::unordered_set<uint64_t>> nicDistanceToNic;
      for (int j = 0; j < nranks; ++j) {
        if (clusterIdData[j] != i) {
          continue;
        }
        int val = nicDistanceData[j].distance;
        uint64_t netGuid = nicDistanceData[j].netGuid;
        if (nicDistanceToNic[val].find(netGuid) ==
            nicDistanceToNic[val].end()) {
          nicDistanceToRanks[val].push_back(j);
          nicDistanceToNic[val].insert(netGuid);
        }
        minDistance = std::min(minDistance, val);
      }
      (*comm)->clusterInterRankList[i] =
          std::move(nicDistanceToRanks[minDistance]);
    }
    // Set homoInterMyRank, homoInterRootRank and homoInterRanks
    auto &myClusterInterRanks =
        (*comm)->clusterInterRankList[clusterIdData[rank]];
    for (size_t i = 0; i < myClusterInterRanks.size(); ++i) {
      if (rank == myClusterInterRanks[i]) {
        (*comm)->homoInterMyRank = i;
      }
    }
    if ((*comm)->homoInterMyRank != -1) {
      (*comm)->homoInterRootRank = myClusterInterRanks[0];
      (*comm)->homoInterRanks = myClusterInterRanks.size();
    }

    INFO(SDCCL_INIT,
         "rank = %d, nranks = %d, nclusters = %d, "
         "clusterId = %d, clusterSize = %d, "
         "clusterInterRank = %d, homoRank = %d, "
         "homoRootRank = %d, homoRanks = %d, "
         "homoInterRootRank = %d, homoInterMyRank = %d, "
         "homoInterRanks = %d, hasSingleRankHomoComm = %d, ",
         rank, nranks, (*comm)->nclusters, (*comm)->clusterIds[rank],
         (*comm)->clusterSizes[(*comm)->clusterIds[rank]],
         (*comm)->clusterInterRanks[(*comm)->clusterIds[rank]],
         (*comm)->homoRank, (*comm)->homoRootRank, (*comm)->homoRanks,
         (*comm)->homoInterRootRank, (*comm)->homoInterMyRank,
         (*comm)->homoInterRanks, (*comm)->hasSingleRankHomoComm);

    // Experimental for multi-nic support
    // Reset commId and homo inter root rank calls underlying GetUniqueId
    // function for initialization of homo inter communicator
    memset((void *)commId, 0, sizeof(sdcclUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(sdcclUniqueId));
    // Let homoInterRootRank call underlying GetUniqueId function
    // for initialization of homo inter communicator
    if (rank == (*comm)->homoInterRootRank) {
      cclAdaptors[sdcclCCLAdaptorDevice]->getUniqueId(&commId);
      memcpy((void *)&uniqueIdData[rank], (void *)commId,
             sizeof(sdcclUniqueId));
    }
    // Collect uniqueIdData globally
    SDCCLCHECK(bootstrapAllGather(state, (void *)uniqueIdData,
                                   sizeof(sdcclUniqueId)));
    SDCCLCHECK(bootstrapBarrier(state, rank, nranks, 0));
    // Call cclAdaptor->commInitRank
    if ((*comm)->homoInterRootRank != -1) {
      memcpy((void *)commId, (void *)&uniqueIdData[(*comm)->homoInterRootRank],
             sizeof(sdcclUniqueId));
      SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commInitRank(
          &(*comm)->homoInterComm, (*comm)->homoInterRanks, commId,
          (*comm)->homoInterMyRank, NULL));
    }
    free(nicDistanceData);
    const char *deviceFuncPathEnv = sdcclGetEnv("SDCCL_DEVICE_FUNC_PATH");
    if (deviceFuncPathEnv) {
      SDCCLCHECK(loadKernelSymbol(deviceFuncPathEnv, "deviceAsyncKernel",
                                   &deviceAsyncKernel));
      if (deviceAsyncKernel == NULL) {
        WARN("Failed to load async kernel from %s", deviceFuncPathEnv);
        return sdcclInvalidArgument;
      }
    }
  }

  free(clusterInterRankData);
  free(vendorData);
  if (!useTuner) {
    free(uniqueIdData);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCommFinalize(sdcclComm_t comm) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  SDCCLCHECK(
      cclAdaptors[sdcclCCLAdaptorDevice]->commFinalize(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented
    return sdcclNotSupported;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCommDestroy(sdcclComm_t comm) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));

  // Destroy cluster info
  free(comm->clusterIds);
  free(comm->clusterSizes);
  free(comm->globalRank2HomoRank);
  free(comm->localRankToRank);
  free(comm->c2cSchedule);

  // Destroy homo comms
  if (comm->tuner) {
    for (const auto &item : comm->homoCommMap) {
      if (item.second != nullptr) {
        SDCCLCHECK(
            cclAdaptors[sdcclCCLAdaptorDevice]->commDestroy(item.second));
      }
    }
  } else {
    SDCCLCHECK(
        cclAdaptors[sdcclCCLAdaptorDevice]->commDestroy(comm->homoComm));
  }

  if (!useHomoComm(comm)) {
    // Destroy hetero comm
    SDCCLCHECK(sdcclHeteroCommDestroy(comm->heteroComm));
    // Destroy host comm
    if (useHostComm()) {
      SDCCLCHECK(
          cclAdaptors[sdcclCCLAdaptorHost]->commDestroy(comm->hostComm));
    }
  }

  // Clean up IPC peer pointer table — deferred to here.
  SDCCLCHECK(sdcclCommCleanupIpcTable(comm));

  // Drain deferred device/host-pinned memory frees,
  // collected during DevComm/DevMem cleanup.
  SDCCLCHECK(sdcclCommDrainDeferredFrees(comm));

  // Destroy bootstrap state and net
  bootstrapClose(comm->bootstrap);

  // Destroy tuner
  if (comm->tuner) {
    comm->tuner->destroy(comm->tunerContext);
    // Free uniqueIdData
    free(comm->uniqueIdData);
  }

  // Finalize net adaptor plugin (dlclose)
  SDCCLCHECK(sdcclNetAdaptorPluginFinalize());

  return sdcclSuccess;
}

sdcclResult_t sdcclCommAbort(sdcclComm_t comm) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commAbort(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return sdcclNotSupported;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCommResume(sdcclComm_t comm) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commResume(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return sdcclNotSupported;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCommSuspend(sdcclComm_t comm) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commSuspend(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return sdcclNotSupported;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCommCount(const sdcclComm_t comm, int *count) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[sdcclCCLAdaptorDevice]->commCount(comm->homoComm,
                                                          count);
  }
  return sdcclHeteroCommCount(comm->heteroComm, count);
}

sdcclResult_t sdcclCommGetDeviceNumber(const sdcclComm_t comm, int *device) {
  return cclAdaptors[sdcclCCLAdaptorDevice]->commGetDeviceNumber(
      comm->homoComm, device);
}

sdcclResult_t sdcclCommUserRank(const sdcclComm_t comm, int *rank) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[sdcclCCLAdaptorDevice]->commUserRank(comm->homoComm,
                                                             rank);
  }
  return sdcclHeteroCommUserRank(comm->heteroComm, rank);
}

sdcclResult_t sdcclCommFifoBuffer(const sdcclComm_t comm, void **buffer) {
  if (comm->heteroComm->fifoBuffer == NULL) {
    return sdcclInvalidUsage;
  }
  *buffer = comm->heteroComm->fifoBuffer;
  return sdcclSuccess;
}

sdcclResult_t sdcclCommGetAsyncError(sdcclComm_t comm,
                                       sdcclResult_t *asyncError) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[sdcclCCLAdaptorDevice]->commGetAsyncError(
        comm->homoComm, asyncError);
  }
  // TODO: to be implemented.
  return sdcclNotSupported;
}

sdcclResult_t sdcclBarrier(sdcclComm_t comm, sdcclStream_t stream) {
  void *barrierBuff;
  deviceAdaptor->deviceMalloc(&barrierBuff, comm->nranks, sdcclMemDevice,
                              stream);
  deviceAdaptor->deviceMemset(barrierBuff, 0, comm->nranks, sdcclMemDevice,
                              stream);
  sdcclAllReduce(barrierBuff, barrierBuff, comm->nranks, sdcclChar, sdcclMax,
                  comm, stream);
  deviceAdaptor->deviceFree(barrierBuff, sdcclMemDevice, stream);
  deviceAdaptor->streamSynchronize(stream);
  return sdcclSuccess;
}

sdcclResult_t sdcclReduce(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, sdcclRedOp_t op,
                            int root, sdcclComm_t comm,
                            sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C reduce op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclGather(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, int root,
                            sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C gather op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclScatter(const void *sendbuff, void *recvbuff, size_t count,
                             sdcclDataType_t datatype, int root,
                             sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C scatter op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               int root, sdcclComm_t comm,
                               sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C broadcast op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               sdcclRedOp_t op, sdcclComm_t comm,
                               sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C allreduce op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C reducescatter op when "
           "comm->hasSingleRankHomoComm is True");
    }
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, sdcclDataType_t datatype,
                               sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclAlltoAll(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype,
                              sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclAlltoAllv(const void *sendbuff, size_t *sendcounts,
                               size_t *sdispls, void *recvbuff,
                               size_t *recvcounts, size_t *rdispls,
                               sdcclDataType_t datatype, sdcclComm_t comm,
                               sdcclStream_t stream) {

  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSend(const void *sendbuff, size_t count,
                          sdcclDataType_t datatype, int peer,
                          sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->send(sendbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->send(
        sendbuff, count, datatype, peer, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclRecv(void *recvbuff, size_t count,
                          sdcclDataType_t datatype, int peer,
                          sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclEnsureCommReady(comm));
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->recv(recvbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->recv(
        recvbuff, count, datatype, peer, comm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclGet(sdcclComm_t comm, int peer, size_t srcOffset,
                         size_t dstOffset, size_t size, int srcMrIdx,
                         int dstMrIdx) {
  if (comm == NULL || comm->heteroComm == NULL)
    return sdcclInvalidArgument;
  return sdcclHeteroGet(comm->heteroComm, peer, srcOffset, dstOffset, size,
                         srcMrIdx, dstMrIdx);
}

sdcclResult_t sdcclPutSignal(sdcclComm_t comm, int peer, size_t srcOffset,
                               size_t dstOffset, size_t size,
                               size_t signalOffset, int srcMrIdx, int dstMrIdx,
                               uint64_t signalValue) {
  if (comm == NULL || comm->heteroComm == NULL)
    return sdcclInvalidArgument;
  return sdcclHeteroPutSignal(comm->heteroComm, peer, srcOffset, dstOffset,
                               size, signalOffset, srcMrIdx, dstMrIdx,
                               signalValue);
}

sdcclResult_t sdcclSignal(sdcclComm_t comm, int peer, size_t signalOffset,
                            uint64_t signalValue) {
  if (comm == NULL || comm->heteroComm == NULL)
    return sdcclInvalidArgument;
  // Signal-only: size == 0, srcMrIdx/dstMrIdx unused
  return sdcclHeteroPutSignal(comm->heteroComm, peer, 0, 0, 0, signalOffset, 0,
                               0, signalValue);
}

sdcclResult_t sdcclWaitSignal(sdcclComm_t comm, int peer,
                                size_t signalOffset, uint64_t expected,
                                sdcclStream_t stream) {
  if (comm == NULL || comm->heteroComm == NULL)
    return sdcclInvalidArgument;
  if (stream == NULL)
    return sdcclInvalidArgument;
  return sdcclHeteroWaitSignal(comm->heteroComm, peer, signalOffset, expected,
                                stream);
}

sdcclResult_t sdcclGroupStart(sdcclComm_t comm) {
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->groupStart());
  } else if (comm == NULL || useHomoComm(comm)) {
    if (comm == NULL) {
      INFO(
          SDCCL_COLL,
          "sdcclGroupStart: comm is NULL, delegating to homo runner directly");
    }
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->groupStart());
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->groupStart());
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->groupStart());
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclGroupEnd(sdcclComm_t comm) {
  if (useHeteroComm()) {
    SDCCLCHECK(sdcclRunners[sdcclUniRunner]->groupEnd());
  } else if (comm == NULL || useHomoComm(comm)) {
    if (comm == NULL) {
      INFO(SDCCL_COLL,
           "sdcclGroupEnd: comm is NULL, delegating to homo runner directly");
    }
    SDCCLCHECK(sdcclRunners[sdcclHomoRunner]->groupEnd());
  } else if (useHostComm()) {
    SDCCLCHECK(sdcclRunners[sdcclHostRunner]->groupEnd());
  } else {
    SDCCLCHECK(sdcclRunners[sdcclHybridRunner]->groupEnd());
  }
  return sdcclSuccess;
}
