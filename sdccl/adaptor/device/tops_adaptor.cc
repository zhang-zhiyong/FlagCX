/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#include "enflame_adaptor.h"

#ifdef USE_ENFLAME_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, topsMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, topsMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, topsMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, topsMemcpyDeviceToDevice},
};

sdcclResult_t topsAdaptorDeviceSynchronize() {
  DEVCHECK(topsDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       sdcclMemcpyType_t type,
                                       sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(topsMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        topsMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(topsMemset(ptr, value, size));
    } else {
      DEVCHECK(topsMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorDeviceMalloc(void **ptr, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(topsHostMalloc(ptr, size, topsHostMallocMapped));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(topsMallocManaged(ptr, size, topsMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(topsMalloc(ptr, size));
    } else {
      DEVCHECK(topsMallocAsync(ptr, size, stream->base, 0));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                     sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(topsHostFree(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(topsFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(topsFree(ptr));
    } else {
      DEVCHECK(topsFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorSetDevice(int dev) {
  DEVCHECK(topsSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetDevice(int *dev) {
  DEVCHECK(topsGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetDeviceCount(int *count) {
  DEVCHECK(topsGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ENFLAME");
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(topsHostGetDevicePointer(pDevice, pHost, 0));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(topsMalloc(ptr, size));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(topsFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(topsStreamCreateWithFlags((topsStream_t *)(*stream),
                                     topsStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(topsStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamCopy(sdcclStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (topsStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(topsStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    topsError_t error = topsStreamQuery(stream->base);
    if (error == topsSuccess) {
      res = sdcclSuccess;
    } else if (error == topsErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t topsAdaptorStreamWaitEvent(sdcclStream_t stream,
                                          sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(topsStreamWaitEvent(stream->base, event->base, 0));
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorEventCreate(sdcclEvent_t *event,
                                      sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? topsEventDefault
                                 : topsEventDisableTiming;
  DEVCHECK(topsEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(topsEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorEventRecord(sdcclEvent_t event,
                                      sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(topsEventRecord(event->base, stream->base));
    } else {
      DEVCHECK(topsEventRecord(event->base, NULL));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(topsEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    topsError_t error = topsEventQuery(event->base);
    if (error == topsSuccess) {
      res = sdcclSuccess;
    } else if (error == topsErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t topsAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                             size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(topsIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(topsIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(topsIpcOpenMemHandle(devPtr, handle->base,
                                topsIpcMemLazyEnablePeerAccess));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(topsIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorLaunchHostFunc(sdcclStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(topsLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorLaunchDeviceFunc(sdcclStream_t stream,
                                           sdcclLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }

  topsDeviceProp_t devProp;
  DEVCHECK(topsGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;

  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  // TOPS uses topsGetDeviceProperties to get PCI bus ID
  topsDeviceProp_t devProp;
  DEVCHECK(topsGetDeviceProperties(&devProp, dev));
  snprintf(pciBusId, len, "%04x:%02x:%02x.%01x", devProp.pciDomainID,
           devProp.pciBusID, devProp.pciDeviceID, devProp.pciFunctionID);
  return sdcclSuccess;
}

sdcclResult_t topsAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  // Search for device by PCI bus ID
  int count;
  DEVCHECK(topsGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    char busId[64];
    topsAdaptorGetDevicePciBusId(busId, sizeof(busId), i);
    if (strcasecmp(busId, pciBusId) == 0) {
      *dev = i;
      return sdcclSuccess;
    }
  }
  return sdcclInvalidArgument;
}

sdcclResult_t topsAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return sdcclInvalidArgument;

  // TOPS/GCU may not support DMA buffer in the same way as CUDA
  *dmaBufferSupport = false;
  return sdcclSuccess;
}

sdcclResult_t
topsAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  // Not supported for TOPS
  return sdcclNotSupported;
}

sdcclResult_t topsAdaptorEventElapsedTime(float *ms, sdcclEvent_t start,
                                           sdcclEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return sdcclInvalidArgument;
  }
  topsError_t error = topsEventElapsedTime(ms, start->base, end->base);
  if (error == topsSuccess) {
    return sdcclSuccess;
  } else if (error == topsErrorNotReady) {
    return sdcclInProgress;
  } else {
    return sdcclUnhandledDeviceError;
  }
}

sdcclResult_t topsAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t topsAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                             int) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor topsAdaptor {
  "TOPS",
      // Basic functions
      topsAdaptorDeviceSynchronize, topsAdaptorDeviceMemcpy,
      topsAdaptorDeviceMemset, topsAdaptorDeviceMalloc, topsAdaptorDeviceFree,
      topsAdaptorSetDevice, topsAdaptorGetDevice, topsAdaptorGetDeviceCount,
      topsAdaptorGetVendor, topsAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      topsAdaptorGdrMemAlloc, topsAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      topsAdaptorStreamCreate, topsAdaptorStreamDestroy, topsAdaptorStreamCopy,
      topsAdaptorStreamFree, topsAdaptorStreamSynchronize,
      topsAdaptorStreamQuery, topsAdaptorStreamWaitEvent,
      topsAdaptorStreamWaitValue64, topsAdaptorStreamWriteValue64,
      // Event functions
      topsAdaptorEventCreate, topsAdaptorEventDestroy, topsAdaptorEventRecord,
      topsAdaptorEventSynchronize, topsAdaptorEventQuery,
      topsAdaptorEventElapsedTime,
      // IpcMemHandle functions
      topsAdaptorIpcMemHandleCreate, topsAdaptorIpcMemHandleGet,
      topsAdaptorIpcMemHandleOpen, topsAdaptorIpcMemHandleClose,
      topsAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      topsAdaptorLaunchDeviceFunc, // sdcclResult_t
                                   // (*launchDeviceFunc)(sdcclStream_t stream,
                                   // void *args);
      // Others
      topsAdaptorGetDeviceProperties, // sdcclResult_t
                                      // (*getDeviceProperties)(struct
                                      // sdcclDevProps *props, int dev);
      topsAdaptorGetDevicePciBusId, // sdcclResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      topsAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      topsAdaptorLaunchHostFunc,
      // DMA buffer
      topsAdaptorDmaSupport, // sdcclResult_t (*dmaSupport)(bool
                             // *dmaBufferSupport);
      topsAdaptorMemGetHandleForAddressRange, // sdcclResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
};

#endif // USE_ENFLAME_ADAPTOR
