#include "musa_adaptor.h"

#ifdef USE_MUSA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, musaMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, musaMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, musaMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, musaMemcpyDeviceToDevice},
};

sdcclResult_t musaAdaptorDeviceSynchronize() {
  DEVCHECK(musaDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       sdcclMemcpyType_t type,
                                       sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(musaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        musaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(musaMemset(ptr, value, size));
    } else {
      DEVCHECK(musaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorDeviceMalloc(void **ptr, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(musaMallocHost(ptr, size));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(musaMallocManaged(ptr, size, musaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(musaMalloc(ptr, size));
    } else {
      DEVCHECK(musaMalloc(ptr, size));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                     sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(musaFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(musaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(musaFree(ptr));
    } else {
      DEVCHECK(musaFree(ptr));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorSetDevice(int dev) {
  DEVCHECK(musaSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetDevice(int *dev) {
  DEVCHECK(musaGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(musaGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "MUSA");
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(musaMalloc(ptr, size));
  musaPointerAttributes attrs;
  DEVCHECK(musaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(muPointerSetAttribute(&flags, MU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (MUdeviceptr)attrs.devicePointer));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(musaFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(musaStreamCreateWithFlags((musaStream_t *)(*stream),
                                     musaStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(musaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamCopy(sdcclStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (musaStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(musaStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    musaError error = musaStreamQuery(stream->base);
    if (error == musaSuccess) {
      res = sdcclSuccess;
    } else if (error == musaErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t musaAdaptorStreamWaitEvent(sdcclStream_t stream,
                                          sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        musaStreamWaitEvent(stream->base, event->base, musaEventWaitDefault));
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorEventCreate(sdcclEvent_t *event,
                                      sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? musaEventDefault
                                 : musaEventDisableTiming;
  DEVCHECK(musaEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorEventRecord(sdcclEvent_t event,
                                      sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(musaEventRecordWithFlags(event->base, stream->base,
                                        musaEventRecordDefault));
    } else {
      DEVCHECK(musaEventRecordWithFlags(event->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    musaError error = musaEventQuery(event->base);
    if (error == musaSuccess) {
      res = sdcclSuccess;
    } else if (error == musaErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t musaAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                             size_t *size) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t musaAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                          void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t musaAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                           void **devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t musaAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t musaAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t musaAdaptorLaunchHostFunc(sdcclStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(musaLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }

  musaDeviceProp devProp;
  DEVCHECK(musaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some musa versions,
  // musaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(musaDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(musaDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t musaAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t musaAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                             int) {
  return sdcclNotSupported;
}
sdcclResult_t musaAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                           sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor musaAdaptor {
  "MUSA",
      // Basic functions
      musaAdaptorDeviceSynchronize, musaAdaptorDeviceMemcpy,
      musaAdaptorDeviceMemset, musaAdaptorDeviceMalloc, musaAdaptorDeviceFree,
      musaAdaptorSetDevice, musaAdaptorGetDevice, musaAdaptorGetDeviceCount,
      musaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      musaAdaptorGdrMemAlloc, musaAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      musaAdaptorStreamCreate, musaAdaptorStreamDestroy, musaAdaptorStreamCopy,
      musaAdaptorStreamFree, musaAdaptorStreamSynchronize,
      musaAdaptorStreamQuery, musaAdaptorStreamWaitEvent,
      musaAdaptorStreamWaitValue64, musaAdaptorStreamWriteValue64,
      // Event functions
      musaAdaptorEventCreate, musaAdaptorEventDestroy, musaAdaptorEventRecord,
      musaAdaptorEventSynchronize, musaAdaptorEventQuery,
      musaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      musaAdaptorIpcMemHandleCreate, musaAdaptorIpcMemHandleGet,
      musaAdaptorIpcMemHandleOpen, musaAdaptorIpcMemHandleClose,
      musaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      NULL, // sdcclResult_t (*launchDeviceFunc)(sdcclStream_t stream, void
            // *args);
      // Others
      musaAdaptorGetDeviceProperties, // sdcclResult_t
                                      // (*getDeviceProperties)(struct
                                      // sdcclDevProps *props, int dev);
      musaAdaptorGetDevicePciBusId, // sdcclResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      musaAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      musaAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_MUSA_ADAPTOR
