#include "cambricon_adaptor.h"

#ifdef USE_CAMBRICON_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, cnrtMemTransDir_t> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, cnrtMemcpyHostToDev},
    {sdcclMemcpyDeviceToHost, cnrtMemcpyDevToHost},
    {sdcclMemcpyDeviceToDevice, cnrtMemcpyDevToDev},
};

sdcclResult_t mluAdaptorDeviceSynchronize() {
  DEVCHECK(cnrtSyncDevice());
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                      sdcclMemcpyType_t type,
                                      sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(cnrtMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(cnrtMemcpyAsync_V2(dst, src, size, stream->base,
                                memcpy_type_map[type]));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                      sdcclMemType_t type,
                                      sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cnrtMemset(ptr, value, size));
    } else {
      DEVCHECK(cnrtMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorDeviceMalloc(void **ptr, size_t size,
                                      sdcclMemType_t type,
                                      sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cnrtHostMalloc(ptr, size));
  } else {
    DEVCHECK(cnrtMalloc(ptr, size));
    // DEVCHECK(cnrtMallocAsync(ptr, size, stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                    sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cnrtFreeHost(ptr));
  } else {
    DEVCHECK(cnrtFree(ptr));
    // DEVCHECK(cnrtFreeAsync(ptr, stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorSetDevice(int dev) {
  DEVCHECK(cnrtSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetDevice(int *dev) {
  DEVCHECK(cnrtGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cnrtGetDeviceCount(reinterpret_cast<unsigned int *>(count)));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "MLU");
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGdrMemAlloc(void **ptr, size_t size, void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cnrtMalloc(ptr, size));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(cnrtFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(cnrtQueueCreate((cnrtQueue_t *)(*stream)));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cnrtQueueDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamCopy(sdcclStream_t *newStream,
                                    void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (cnrtQueue_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cnrtQueueSync(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    cnrtRet_t error = cnrtQueueQuery(stream->base);
    if (error == cnrtSuccess) {
      res = sdcclSuccess;
    } else if (error == cnrtErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t mluAdaptorLaunchHostFunc(sdcclStream_t stream,
                                        void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cnrtInvokeHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                             int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }
  cnrtDeviceProp_t devProp;
  DEVCHECK(cnrtGetDeviceProperties(&devProp, dev));

  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;

  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cnrtDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cnrtDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorStreamWaitEvent(sdcclStream_t stream,
                                         sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(cnrtQueueWaitNotifier(event->base, stream->base,
                                   CNRT_NOTIFIER_WAIT_DEFAULT);)
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorEventCreate(sdcclEvent_t *event,
                                     sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? CNRT_NOTIFIER_DEFAULT
                                 : CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  DEVCHECK(cnrtNotifierCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cnrtNotifierDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorEventRecord(sdcclEvent_t event,
                                     sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cnrtPlaceNotifierWithFlags(event->base, stream->base,
                                          CNRT_NOTIFIER_PLACE_DEFAULT));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cnrtWaitNotifier(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t mluAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    cnrtRet_t error = cnrtQueryNotifier(event->base);
    if (error == cnrtSuccess) {
      res = sdcclSuccess;
    } else if (error == cnrtErrorBusy) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t mluAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                            size_t *size) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t mluAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                         void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t mluAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                          void **devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t mluAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t mluAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t mluAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                           int) {
  return sdcclNotSupported;
}
sdcclResult_t mluAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t mluAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                          sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor mluAdaptor {
  "MLU",
      // Basic functions
      mluAdaptorDeviceSynchronize, mluAdaptorDeviceMemcpy,
      mluAdaptorDeviceMemset, mluAdaptorDeviceMalloc, mluAdaptorDeviceFree,
      mluAdaptorSetDevice, mluAdaptorGetDevice, mluAdaptorGetDeviceCount,
      mluAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      mluAdaptorGdrMemAlloc, mluAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      mluAdaptorStreamCreate, mluAdaptorStreamDestroy, mluAdaptorStreamCopy,
      mluAdaptorStreamFree, mluAdaptorStreamSynchronize, mluAdaptorStreamQuery,
      mluAdaptorStreamWaitEvent, mluAdaptorStreamWaitValue64,
      mluAdaptorStreamWriteValue64,
      // Event functions
      mluAdaptorEventCreate, mluAdaptorEventDestroy, mluAdaptorEventRecord,
      mluAdaptorEventSynchronize, mluAdaptorEventQuery,
      mluAdaptorEventElapsedTime,
      // IpcMemHandle functions
      mluAdaptorIpcMemHandleCreate, mluAdaptorIpcMemHandleGet,
      mluAdaptorIpcMemHandleOpen, mluAdaptorIpcMemHandleClose,
      mluAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      NULL, // sdcclResult_t (*launchDeviceFunc)(sdcclStream_t stream,
            // void *args);
      // Others
      mluAdaptorGetDeviceProperties, // sdcclResult_t
                                     // (*getDeviceProperties)(struct
                                     // sdcclDevProps *props, int dev);
      mluAdaptorGetDevicePciBusId,   // sdcclResult_t (*getDevicePciBusId)(char
                                     // *pciBusId, int len, int dev);
      mluAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                     // (*getDeviceByPciBusId)(int
                                     // *dev, const char *pciBusId);
      mluAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_CAMBRICON_ADAPTOR
