/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, mcMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, mcMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, mcMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, mcMemcpyDeviceToDevice},
};

sdcclResult_t macaAdaptorDeviceSynchronize() {
  DEVCHECK(mcDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       sdcclMemcpyType_t type,
                                       sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(mcMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        mcMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(mcMemset(ptr, value, size));
    } else {
      DEVCHECK(mcMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorDeviceMalloc(void **ptr, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(mcMallocHost(ptr, size));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(mcMallocManaged(ptr, size, mcMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(mcMalloc(ptr, size));
    } else {
      DEVCHECK(mcMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                     sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(mcFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(mcFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(mcFree(ptr));
    } else {
      DEVCHECK(mcFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorSetDevice(int dev) {
  DEVCHECK(mcSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetDevice(int *dev) {
  DEVCHECK(mcGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(mcGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "METAX");
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(mcMalloc(ptr, size));
  mcPointerAttribute_t attrs;
  DEVCHECK(mcPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(mcPointerSetAttribute(&flags, mcPointerAttributeSyncMemops,
                                 (mcDeviceptr_t)attrs.devicePointer));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(mcFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(
      mcStreamCreateWithFlags((mcStream_t *)(*stream), mcStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(mcStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamCopy(sdcclStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (mcStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(mcStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    mcError_t error = mcStreamQuery(stream->base);
    if (error == mcSuccess) {
      res = sdcclSuccess;
    } else if (error == mcErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t macaAdaptorStreamWaitEvent(sdcclStream_t stream,
                                          sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(mcStreamWaitEvent(stream->base, event->base, mcEventWaitDefault));
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventCreate(sdcclEvent_t *event,
                                      sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags =
      (eventType == sdcclEventDefault) ? mcEventDefault : mcEventDisableTiming;
  DEVCHECK(mcEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(mcEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventRecord(sdcclEvent_t event,
                                      sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(mcEventRecordWithFlags(event->base, stream->base,
                                      mcEventRecordDefault));
    } else {
      DEVCHECK(mcEventRecordWithFlags(event->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(mcEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    mcError_t error = mcEventQuery(event->base);
    if (error == mcSuccess) {
      res = sdcclSuccess;
    } else if (error == mcErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t macaAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                             size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(mcIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(mcIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(
      mcIpcOpenMemHandle(devPtr, handle->base, mcIpcMemLazyEnablePeerAccess));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(mcIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorLaunchHostFunc(sdcclStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(mcLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }

  mcDeviceProp_t devProp;
  DEVCHECK(mcGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some mc versions,
  // mcDeviceProp_t does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(mcDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(mcDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t macaAdaptorEventElapsedTime(float *ms, sdcclEvent_t start,
                                           sdcclEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return sdcclInvalidArgument;
  }
  mcError_t error = mcEventElapsedTime(ms, start->base, end->base);
  if (error == mcSuccess) {
    return sdcclSuccess;
  } else if (error == mcErrorNotReady) {
    return sdcclInProgress;
  } else {
    return sdcclUnhandledDeviceError;
  }
}

sdcclResult_t macaAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t macaAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                             int) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor macaAdaptor {
  "MACA",
      // Basic functions
      macaAdaptorDeviceSynchronize, macaAdaptorDeviceMemcpy,
      macaAdaptorDeviceMemset, macaAdaptorDeviceMalloc, macaAdaptorDeviceFree,
      macaAdaptorSetDevice, macaAdaptorGetDevice, macaAdaptorGetDeviceCount,
      macaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      macaAdaptorGdrMemAlloc, macaAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      macaAdaptorStreamCreate, macaAdaptorStreamDestroy, macaAdaptorStreamCopy,
      macaAdaptorStreamFree, macaAdaptorStreamSynchronize,
      macaAdaptorStreamQuery, macaAdaptorStreamWaitEvent,
      macaAdaptorStreamWaitValue64, macaAdaptorStreamWriteValue64,
      // Event functions
      macaAdaptorEventCreate, macaAdaptorEventDestroy, macaAdaptorEventRecord,
      macaAdaptorEventSynchronize, macaAdaptorEventQuery,
      macaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      macaAdaptorIpcMemHandleCreate, macaAdaptorIpcMemHandleGet,
      macaAdaptorIpcMemHandleOpen, macaAdaptorIpcMemHandleClose,
      macaAdaptorIpcMemHandleFree,
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
      macaAdaptorGetDeviceProperties, // sdcclResult_t
                                      // (*getDeviceProperties)(struct
                                      // sdcclDevProps *props, int dev);
      macaAdaptorGetDevicePciBusId, // sdcclResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      macaAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      macaAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_METAX_ADAPTOR
