#include "net.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "device.h"
#include "proxy.h"
#include "reg_pool.h"

#include <errno.h>
#include <string.h>
#include <string>

int64_t sdcclNetBufferSize;
int64_t sdcclNetChunkSize;
int64_t sdcclNetChunks;

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
// Use adaptor system for all network types
struct sdcclNetAdaptor *sdcclNetAdaptors[3] = {
    nullptr, getUnifiedNetAdaptor(IBRC), getUnifiedNetAdaptor(SOCKET)};
enum sdcclNetState sdcclNetStates[3] = {
    sdcclNetStateInit, sdcclNetStateInit, sdcclNetStateInit};

sdcclResult_t sdcclNetCheckDeviceVersion(struct sdcclHeteroComm *comm,
                                           struct sdcclNetAdaptor *net,
                                           int dev) {
  sdcclNetProperties_v1_t props;

  SDCCLCHECK(net->getProperties(dev, (void *)&props));
  sdcclNetDeviceType type = props.netDeviceType;
  if (type)
    switch (type) {
      case SDCCL_NET_DEVICE_UNPACK:
        if (props.netDeviceVersion == SDCCL_NET_DEVICE_UNPACK_VERSION) {
          INFO(SDCCL_INIT,
               "Using SDCCL_NET_DEVICE_UNPACK net plugin version %d",
               props.netDeviceVersion);
          return sdcclSuccess;
        } else {
          WARN("SDCCL_DEVICE_UNPACK plugin has incompatible version %d, this "
               "sdccl build is compatible with %d, not using it",
               props.netDeviceVersion, SDCCL_NET_DEVICE_UNPACK_VERSION);
          return sdcclInternalError;
        }
      default:
        WARN("Unknown device code index");
        return sdcclInternalError;
    }

  INFO(SDCCL_INIT, "Using non-device net plugin version %d",
       props.netDeviceVersion);
  return sdcclSuccess;
}

static sdcclResult_t netGetState(int i, enum sdcclNetState *state) {
  pthread_mutex_lock(&netLock);
  if (sdcclNetStates[i] == sdcclNetStateInit) {
    int ndev;
    if (sdcclNetAdaptors[i] == nullptr) {
      sdcclNetStates[i] = sdcclNetStateDisabled;
    } else if (sdcclNetAdaptors[i]->init() != sdcclSuccess) {
      sdcclNetStates[i] = sdcclNetStateDisabled;
    } else if (sdcclNetAdaptors[i]->devices(&ndev) != sdcclSuccess ||
               ndev <= 0) {
      sdcclNetStates[i] = sdcclNetStateDisabled;
    } else {
      sdcclNetStates[i] = sdcclNetStateEnabled;
    }
  }
  *state = sdcclNetStates[i];
  pthread_mutex_unlock(&netLock);
  return sdcclSuccess;
}

sdcclResult_t sdcclNetInit(struct sdcclHeteroComm *comm) {
  // Initialize main communication network
  const char *netName;
  bool ok = false;

  const char *forceSocketEnv = getenv("SDCCL_FORCE_NET_SOCKET");
  bool forceSocket = (forceSocketEnv && atoi(forceSocketEnv) == 1);

  netName = comm->config.netName;

  if (!forceSocket) {
    // Load net plugin if SDCCL_NET_ADAPTOR_PLUGIN is set.
    // This populates sdcclNetAdaptors[0] with the plugin.
    // Must be called before the selection loop below.
    SDCCLCHECK(sdcclNetAdaptorPluginInit());
  }

  if (forceSocket) {
    // Force socket network usage
    for (int i = 2; i >= 0; i--) {
      if (sdcclNetAdaptors[i] == nullptr)
        continue;
      if (sdcclNetAdaptors[i] != getUnifiedNetAdaptor(SOCKET))
        continue;
      enum sdcclNetState state;
      SDCCLCHECK(netGetState(i, &state));
      if (state != sdcclNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, sdcclNetAdaptors[i]->name) != 0)
        continue;
      if (sdcclSuccess !=
          sdcclNetCheckDeviceVersion(comm, sdcclNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = sdcclNetAdaptors[i];
      ok = true;

      break;
    }
  } else {
    // Normal network selection order (IBUC first when enabled, then IBRC, then
    // socket)
    for (int i = 0; i < 3; i++) {
      if (sdcclNetAdaptors[i] == nullptr)
        continue;
      enum sdcclNetState state;
      SDCCLCHECK(netGetState(i, &state));
      if (state != sdcclNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, sdcclNetAdaptors[i]->name) != 0)
        continue;
      if (sdcclSuccess !=
          sdcclNetCheckDeviceVersion(comm, sdcclNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = sdcclNetAdaptors[i];
      ok = true;

      break;
    }
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return sdcclInvalidUsage;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclProxySend(sendNetResources *resources, void *data,
                               size_t size, sdcclProxyArgs *args) {
  if (args->done) {
    return sdcclSuccess;
  }
  if (!args->semaphore->pollStart(args->opId, args->step)) {
    return sdcclSuccess;
  }
  if (args->transmitted < args->chunkSteps) {
    int stepMask = args->sendStepMask;

    if (args->waitCopy < args->chunkSteps &&
        args->waitCopy - args->transmitted < sdcclNetChunks) {
      int step = args->waitCopy & stepMask;
      args->subs[step].stepSize =
          std::min(args->chunkSize, size - args->totalCopySize);
      if (!args->regBufFlag) {
        args->subs[step].stepBuff =
            resources->buffers[0] + (sdcclNetChunkSize * step);
        if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
          SDCCLCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, sdcclMemcpyDeviceToDevice,
              resources->cpStream, args->subs[step].copyArgs));
        } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
          SDCCLCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, sdcclMemcpyDeviceToHost,
              resources->cpStream, args->subs[step].copyArgs));
        } else {
          SDCCLCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize,
              (resources->ptrSupport & SDCCL_PTR_CUDA)
                  ? sdcclMemcpyDeviceToDevice
                  : sdcclMemcpyDeviceToHost,
              resources->cpStream, args->subs[step].copyArgs));
        }
        SDCCLCHECK(deviceAdaptor->eventRecord(resources->cpEvents[step],
                                               resources->cpStream));
      } else {
        args->subs[step].stepBuff =
            (void *)((char *)data + (sdcclNetChunkSize * args->waitCopy));
      }
      args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->posted < args->waitCopy) {
      int step = args->posted & stepMask;
      int done = 0;
      if (!args->regBufFlag) {
        if (deviceAdaptor->eventQuery(resources->cpEvents[step]) ==
            sdcclSuccess) {
          args->copied++;
          done = 1;
        }
      } else {
        done = 1;
      }
      if (done) {
        void *req = NULL;
        resources->netAdaptor->isend(
            resources->netSendComm,
            args->subs[args->posted & stepMask].stepBuff,
            args->subs[args->posted & stepMask].stepSize, 0,
            args->regBufFlag ? args->regHandle : resources->mhandles[0], NULL,
            &req);
        if (req) {
          args->subs[args->posted++ & stepMask].requests[0] = req;
        }
      }
    }

    if (args->transmitted < args->posted) {
      void *req = args->subs[args->transmitted & stepMask].requests[0];
      int done = 0, sizes;
      resources->netAdaptor->test(req, &done, &sizes);
      if (done) {
        args->transmitted++;
      }
    }
  } else {
    if (args->done != 1) {
      args->semaphore->subCounter(args->opId);
      args->done = 1;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclProxyRecv(recvNetResources *resources, void *data,
                               size_t size, sdcclProxyArgs *args) {
  if (args->done) {
    return sdcclSuccess;
  }
  if (!args->semaphore->pollStart(args->opId, args->step)) {
    return sdcclSuccess;
  }
  if (args->copied < args->chunkSteps) {
    int stepMask = args->sendStepMask;
    if (args->posted < args->chunkSteps &&
        args->posted - args->copied < sdcclNetChunks) {
      int tags[8] = {0};
      void *req = NULL;
      args->subs[args->posted & stepMask].stepSize =
          std::min(args->chunkSize, size - args->totalPostSize);
      if (!args->regBufFlag) {
        args->subs[args->posted & stepMask].stepBuff =
            resources->buffers[0] +
            sdcclNetChunkSize * (args->posted & stepMask);
      } else {
        args->subs[args->posted & stepMask].stepBuff =
            (void *)((char *)data + sdcclNetChunkSize * args->posted);
      }
      resources->netAdaptor->irecv(
          resources->netRecvComm, 1,
          &args->subs[args->posted & stepMask].stepBuff,
          (size_t *)&args->subs[args->posted & stepMask].stepSize, tags,
          args->regBufFlag ? &args->regHandle : resources->mhandles, NULL,
          &req);
      if (req) {
        args->subs[args->posted & stepMask].requests[0] = req;
        args->totalPostSize += args->subs[args->posted++ & stepMask].stepSize;
        return sdcclSuccess;
      }
    }

    if (args->postFlush < args->posted) {
      void *req = args->subs[args->postFlush & stepMask].requests[0];
      int done = 0, sizes;
      resources->netAdaptor->test(req, &done, &sizes);
      if (done) {
        if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
          void *req = NULL;
          resources->netAdaptor->iflush(
              resources->netRecvComm, 1,
              &args->subs[args->postFlush & stepMask].stepBuff,
              &args->subs[args->postFlush & stepMask].stepSize,
              args->regBufFlag ? &args->regHandle : resources->mhandles, &req);
          if (req) {
            args->subs[args->postFlush++ & stepMask].requests[0] = req;
          }
        } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
          args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
        } else {
          if (resources->ptrSupport & SDCCL_PTR_CUDA) {
            // RDMA-style: flush
            void *req = NULL;
            resources->netAdaptor->iflush(
                resources->netRecvComm, 1,
                &args->subs[args->postFlush & stepMask].stepBuff,
                &args->subs[args->postFlush & stepMask].stepSize,
                args->regBufFlag ? &args->regHandle : resources->mhandles,
                &req);
            if (req) {
              args->subs[args->postFlush++ & stepMask].requests[0] = req;
            }
          } else {
            // Host-only: skip flush
            args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
          }
        }
        return sdcclSuccess;
      }
    }

    if (args->waitCopy < args->postFlush) {
      int step = args->waitCopy & stepMask;
      void *req = args->subs[step].requests[0];
      int done = 0, sizes;
      if (req == (void *)0x1) {
        done = 1;
        sizes = 0;
      } else {
        resources->netAdaptor->test(req, &done, &sizes);
      }
      if (done) {
        if (!args->regBufFlag) {
          if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            SDCCLCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize, sdcclMemcpyDeviceToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
            SDCCLCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize, sdcclMemcpyHostToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          } else {
            SDCCLCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize,
                (resources->ptrSupport & SDCCL_PTR_CUDA)
                    ? sdcclMemcpyDeviceToDevice
                    : sdcclMemcpyHostToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          }
          SDCCLCHECK(deviceAdaptor->eventRecord(resources->cpEvents[step],
                                                 resources->cpStream));
        }
        args->totalCopySize += args->subs[step].stepSize;
        args->waitCopy++;
        return sdcclSuccess;
      }
    }

    if (args->copied < args->waitCopy) {
      int step = args->copied & stepMask;
      if (!args->regBufFlag) {
        if (deviceAdaptor->eventQuery(resources->cpEvents[step]) ==
            sdcclSuccess) {
          args->copied++;
        }
      } else {
        args->copied++;
      }
    }
  } else {
    if (args->done != 1) {
      args->semaphore->subCounter(args->opId);
      args->done = 1;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSendProxyFree(sendNetResources *resources) {
  for (int s = 0; s < sdcclNetChunks; s++) {
    SDCCLCHECK(deviceAdaptor->eventDestroy(resources->cpEvents[s]));
  }
  SDCCLCHECK(deviceAdaptor->streamDestroy(resources->cpStream));
  resources->netAdaptor->deregMr(resources->netSendComm,
                                 resources->mhandles[0]);
  resources->netAdaptor->closeSend(resources->netSendComm);
  if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
    free(resources->buffers[0]);
  } else if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
    SDCCLCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
  } else {
    if (resources->ptrSupport & SDCCL_PTR_CUDA) {
      SDCCLCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
    } else {
      free(resources->buffers[0]);
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclRecvProxyFree(recvNetResources *resources) {
  for (int s = 0; s < sdcclNetChunks; s++) {
    SDCCLCHECK(deviceAdaptor->eventDestroy(resources->cpEvents[s]));
  }
  SDCCLCHECK(deviceAdaptor->streamDestroy(resources->cpStream));
  resources->netAdaptor->deregMr(resources->netRecvComm,
                                 resources->mhandles[0]);
  resources->netAdaptor->closeRecv(resources->netRecvComm);
  resources->netAdaptor->closeListen(resources->netListenComm);
  if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
    free(resources->buffers[0]);
  } else if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
    SDCCLCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
  } else {
    if (resources->ptrSupport & SDCCL_PTR_CUDA) {
      SDCCLCHECK(deviceAdaptor->gdrMemFree(resources->buffers[0], NULL));
    } else {
      free(resources->buffers[0]);
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t netRegisterBuffer(sdcclHeteroComm *comm,
                                        const void *userbuff, size_t buffSize,
                                        struct sdcclConnector **peerConns,
                                        int nPeers, sdcclRegItem *regRecord,
                                        int *outRegBufFlag, void **outHandle) {
  *outRegBufFlag = 0;
  if (regRecord) {
    for (int p = 0; p < nPeers; ++p) {
      struct sdcclConnector *peerConn = peerConns[p];
      struct sdcclProxyConnector *peerProxyConn = NULL;
      bool found = false;
      if (peerConn == NULL)
        continue;
      peerProxyConn = &peerConn->proxyConn;
      for (auto it = regRecord->handles.begin(); it != regRecord->handles.end();
           it++) {
        if (it->first.proxyConn == peerProxyConn && it->first.handle) {
          found = true;
          outHandle[p] = it->first.handle;
          *outRegBufFlag = 1;
          INFO(SDCCL_REG,
               "rank %d - NET reuse buffer %p size %ld (baseAddr %p size %ld) "
               "handle %p",
               comm->rank, userbuff, buffSize, (void *)regRecord->beginAddr,
               regRecord->endAddr - regRecord->beginAddr, it->first.handle);
          break;
        }
      }
      if (!found) {
        struct netRegInfo info = {regRecord->beginAddr,
                                  regRecord->endAddr - regRecord->beginAddr};
        void *handle = NULL;
        SDCCLCHECK(sdcclProxyCallBlocking(
            (sdcclHeteroComm *)comm, peerProxyConn, sdcclProxyMsgRegister,
            &info, sizeof(struct netRegInfo), &handle, sizeof(void *)));
        if (handle) {
          SDCCLCHECK(globalRegPool.addNetHandle(comm, regRecord, handle,
                                                 peerProxyConn));
          outHandle[p] = handle;
          *outRegBufFlag = 1;
          INFO(SDCCL_REG,
               "rank %d - NET register userbuff %p (handle %p), buffSize %ld",
               comm->rank, userbuff, handle, buffSize);
        } else {
          INFO(SDCCL_REG,
               "rank %d failed to NET register userbuff %p buffSize %ld",
               comm->rank, userbuff, buffSize);
        }
      }
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetRegisterBuffer(sdcclHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct sdcclConnector **peerConns,
                                       int nPeers, int *outRegBufFlag,
                                       void **outHandle) {
  INFO(SDCCL_REG, "comm = %p, userbuff = %p, buffSize = %ld, nPeers = %d",
       comm, userbuff, buffSize, nPeers);
  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    sdcclRegItem *reg = globalRegPool.getItem(reinterpret_cast<void *>(comm),
                                               const_cast<void *>(userbuff));
    if (reg != NULL && reg->refCount > 0) {
      SDCCLCHECK(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers,
                                    reg, outRegBufFlag, outHandle));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetDeregisterBuffer(void *comm,
                                         struct sdcclProxyConnector *proxyConn,
                                         void *handle) {
  INFO(SDCCL_REG, "rank %d - deregister net buffer handle %p",
       reinterpret_cast<sdcclHeteroComm *>(comm)->rank, handle);
  SDCCLCHECK(sdcclProxyCallBlocking(
      reinterpret_cast<sdcclHeteroComm *>(comm), proxyConn,
      sdcclProxyMsgDeregister, &handle, sizeof(void *), NULL, 0));
  return sdcclSuccess;
}