#include "tsmicro_adaptor.h"

#ifdef USE_TSM_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, txMemcpyKind> memcpyTypeMap = {
    {sdcclMemcpyHostToDevice, txMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, txMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, txMemcpyDeviceToDevice},
};

sdcclResult_t tsmicroAdaptorDeviceSynchronize() {
  DEVCHECK(txDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                          sdcclMemcpyType_t type,
                                          sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(txMemcpy(dst, src, size, memcpyTypeMap[type]));
  } else {
    DEVCHECK(txMemcpyAsync(dst, src, size, memcpyTypeMap[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                          sdcclMemType_t type,
                                          sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      // TODO: supported later
      // DEVCHECK(txMemset(ptr, value, size));
    } else {
      // TODO: supported later
    }
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorDeviceMalloc(void **ptr, size_t size,
                                          sdcclMemType_t type,
                                          sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(txMallocHost(ptr, size));
  } else if (type == sdcclMemDevice) {
    DEVCHECK(txMalloc(ptr, size));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                        sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(txFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(txFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(txFree(ptr));
    } else {
      DEVCHECK(txFree(ptr));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorSetDevice(int dev) {
  DEVCHECK(txSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetDevice(int *dev) {
  DEVCHECK(txGetDevice((uint32_t *)dev));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetDeviceCount(int *count) {
  DEVCHECK(txGetDeviceCount((uint32_t *)count));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetVendor(char *vendor) {
  if (vendor != NULL) {
    strncpy(vendor, "TSMICRO", MAX_VENDOR_LEN - 1);
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  return sdcclNotSupported;
}

sdcclResult_t tsmicroAdaptorGdrMemAlloc(void **ptr, size_t size,
                                         void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(txMalloc(ptr, size));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr != NULL) {
    DEVCHECK(txFree(ptr));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(txStreamCreate((txStream_t *)(*stream)));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(txStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamCopy(sdcclStream_t *newStream,
                                        void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (txStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(txStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    txError_t error = txStreamQuery(stream->base);
    if (error == TX_SUCCESS) {
      res = sdcclSuccess;
    } else if (error == TX_ERROR_NOT_READY) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t tsmicroAdaptorStreamWaitEvent(sdcclStream_t stream,
                                             sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(txStreamWaitEvent(stream->base, event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventCreate(sdcclEvent_t *event,
                                         sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  DEVCHECK(txEventCreate(&(*event)->base));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(txEventDestroy(event->base));
    free(event);
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventRecord(sdcclEvent_t event,
                                         sdcclStream_t stream) {
  if (event != NULL) {
    DEVCHECK(txEventRecord(event->base, stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(txEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    txError_t error = txEventQuery(event->base);
    if (error == TX_SUCCESS) {
      res = sdcclSuccess;
    } else if (error == TX_ERROR_NOT_READY) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  } else {
    return sdcclInvalidArgument;
  }
  return res;
}

sdcclResult_t tsmicroAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                                size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(txIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                             void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(txIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                              void **devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(txIpcOpenMemHandle(devPtr, handle->base));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(txIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorLaunchHostFunc(sdcclStream_t stream,
                                            void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(txLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorLaunchDeviceFunc(sdcclStream_t stream,
                                              sdcclLaunchFunc_t fn,
                                              void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                                 int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }
  txDeviceProperty devProp;
  DEVCHECK(txGetDeviceProperty(dev, &devProp));

  strncpy(props->name, devProp.devProp.devName, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.devProp.busId;
  props->pciDeviceId = devProp.devProp.deviceId;
  props->pciDomainId = devProp.devProp.domainId;
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                               int dev) {
  if (pciBusId == NULL || len < 12) {
    return sdcclInvalidArgument;
  }

  txDeviceProperty devProp;
  DEVCHECK(txGetDeviceProperty(dev, &devProp));

  // Format PCI Bus ID as "domain:bus:device.function"
  snprintf(pciBusId, len, "%04x:%02x:%02x.0", devProp.devProp.domainId,
           devProp.devProp.busId, devProp.devProp.deviceId);
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorGetDeviceByPciBusId(int *dev,
                                                 const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(txGetDeviceByPCIBusId((uint32_t *)dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return sdcclInvalidArgument;

  *dmaBufferSupport = true;
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorMemGetHandleForAddressRange(
    void *handleOut, void *buffer, size_t size, unsigned long long flags) {
  DEVCHECK(txMemGetHandleForAddressRange(
      handleOut, buffer, size, TX_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags));
  return sdcclSuccess;
}

sdcclResult_t tsmicroAdaptorEventElapsedTime(float *ms, sdcclEvent_t start,
                                              sdcclEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return sdcclInvalidArgument;
  }
  txError_t error = txEventElapsedTime(ms, start->base, end->base);
  if (error == TX_SUCCESS) {
    return sdcclSuccess;
  } else if (error == TX_ERROR_INVALID_HANDLE) {
    return sdcclInvalidArgument;
  } else if (error == TX_ERROR_NOT_READY) {
    return sdcclInProgress;
  } else {
    return sdcclUnhandledDeviceError;
  }
}
sdcclResult_t tsmicroAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                               int) {
  return sdcclNotSupported;
}
sdcclResult_t tsmicroAdaptorStreamWriteValue64(sdcclStream_t, void *,
                                                uint64_t, int) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor tsmicroAdaptor {
  "TSM",
      // Basic functions
      tsmicroAdaptorDeviceSynchronize, tsmicroAdaptorDeviceMemcpy,
      tsmicroAdaptorDeviceMemset, tsmicroAdaptorDeviceMalloc,
      tsmicroAdaptorDeviceFree, tsmicroAdaptorSetDevice,
      tsmicroAdaptorGetDevice, tsmicroAdaptorGetDeviceCount,
      tsmicroAdaptorGetVendor, tsmicroAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      tsmicroAdaptorGdrMemAlloc, tsmicroAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      tsmicroAdaptorStreamCreate, tsmicroAdaptorStreamDestroy,
      tsmicroAdaptorStreamCopy, tsmicroAdaptorStreamFree,
      tsmicroAdaptorStreamSynchronize, tsmicroAdaptorStreamQuery,
      tsmicroAdaptorStreamWaitEvent, tsmicroAdaptorStreamWaitValue64,
      tsmicroAdaptorStreamWriteValue64,
      // Event functions
      tsmicroAdaptorEventCreate, tsmicroAdaptorEventDestroy,
      tsmicroAdaptorEventRecord, tsmicroAdaptorEventSynchronize,
      tsmicroAdaptorEventQuery, tsmicroAdaptorEventElapsedTime,
      // IpcMemHandle functions
      tsmicroAdaptorIpcMemHandleCreate, tsmicroAdaptorIpcMemHandleGet,
      tsmicroAdaptorIpcMemHandleOpen, tsmicroAdaptorIpcMemHandleClose,
      tsmicroAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      tsmicroAdaptorLaunchDeviceFunc, // sdcclResult_t
                                      // (*launchDeviceFunc)(sdcclStream_t
                                      // stream, void *args);
      // Others
      tsmicroAdaptorGetDeviceProperties, // sdcclResult_t
                                         // (*getDeviceProperties)(struct
                                         // sdcclDevProps *props, int dev);
      tsmicroAdaptorGetDevicePciBusId,   // sdcclResult_t
                                         // (*getDevicePciBusId)(char *pciBusId,
                                         // int len, int dev);
      tsmicroAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                         // (*getDeviceByPciBusId)(int
                                         // *dev, const char *pciBusId);
      tsmicroAdaptorLaunchHostFunc,
      // DMA buffer
      tsmicroAdaptorDmaSupport, // sdcclResult_t (*dmaSupport)(bool
                                // *dmaBufferSupport);
      tsmicroAdaptorMemGetHandleForAddressRange, // sdcclResult_t
                                                 // (*memGetHandleForAddressRange)(void
                                                 // *handleOut, void *buffer,
                                                 // size_t size, unsigned long
                                                 // long flags);
};

#endif // USE_TSM_ADAPTOR
