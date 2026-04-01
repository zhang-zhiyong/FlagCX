#include "amd_adaptor.h"

#ifdef USE_AMD_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, hipMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, hipMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, hipMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, hipMemcpyDeviceToDevice},
};

sdcclResult_t hipAdaptorDeviceSynchronize() {
  DEVCHECK(hipDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                      sdcclMemcpyType_t type,
                                      sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(hipMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        hipMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                      sdcclMemType_t type,
                                      sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(hipMemset(ptr, value, size));
    } else {
      DEVCHECK(hipMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorDeviceMalloc(void **ptr, size_t size,
                                      sdcclMemType_t type,
                                      sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(hipHostMalloc(ptr, size));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(hipMallocManaged(ptr, size, hipMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(hipMalloc(ptr, size));
    } else {
      DEVCHECK(hipMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                    sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(hipFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(hipFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(hipFree(ptr));
    } else {
      DEVCHECK(hipFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorSetDevice(int dev) {
  DEVCHECK(hipSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetDevice(int *dev) {
  DEVCHECK(hipGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetDeviceCount(int *count) {
  DEVCHECK(hipGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetVendor(char *vendor) {
  strncpy(vendor, "AMD", MAX_VENDOR_LEN - 1);
  vendor[MAX_VENDOR_LEN - 1] = '\0';
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGdrMemAlloc(void **ptr, size_t size, void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(hipMalloc(ptr, size));
  hipPointerAttribute_t attrs;
  DEVCHECK(hipPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(hipPointerSetAttribute(&flags, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (hipDeviceptr_t)attrs.devicePointer));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(hipFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(
      hipStreamCreateWithFlags((hipStream_t *)(*stream), hipStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamDestroy(stream->base));
    free(stream);
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamCopy(sdcclStream_t *newStream,
                                    void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (hipStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    hipError_t error = hipStreamQuery(stream->base);
    if (error == hipSuccess) {
      res = sdcclSuccess;
    } else if (error == hipErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t hipAdaptorStreamWaitEvent(sdcclStream_t stream,
                                         sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(hipStreamWaitEvent(stream->base, event->base, 0));
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorEventCreate(sdcclEvent_t *event,
                                     sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? hipEventDefault
                                 : hipEventDisableTiming;
  DEVCHECK(hipEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventDestroy(event->base));
    free(event);
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorEventRecord(sdcclEvent_t event,
                                     sdcclStream_t stream) {
  if (event != NULL && stream != NULL) {
    DEVCHECK(hipEventRecord(event->base, stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    hipError_t error = hipEventQuery(event->base);
    if (error == hipSuccess) {
      res = sdcclSuccess;
    } else if (error == hipErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t hipAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                            size_t *size) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                         void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                          void **devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorLaunchHostFunc(sdcclStream_t stream,
                                        void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(hipLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}
sdcclResult_t hipAdaptorLaunchDeviceFunc(sdcclStream_t stream,
                                          sdcclLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                             int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }

  hipDeviceProp_t devProp;
  DEVCHECK(hipGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(hipDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(hipDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorDmaSupport(bool *dmaBufferSupport) {
  *dmaBufferSupport = false;
  return sdcclSuccess;
}

sdcclResult_t hipAdaptorMemGetHandleForAddressRange(void *handleOut,
                                                     void *buffer, size_t size,
                                                     unsigned long long flags) {
  return sdcclNotSupported;
}

sdcclResult_t hipAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                           int) {
  return sdcclNotSupported;
}
sdcclResult_t hipAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t hipAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                          sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor hipAdaptor {
  "HIP",
      // Basic functions
      hipAdaptorDeviceSynchronize, hipAdaptorDeviceMemcpy,
      hipAdaptorDeviceMemset, hipAdaptorDeviceMalloc, hipAdaptorDeviceFree,
      hipAdaptorSetDevice, hipAdaptorGetDevice, hipAdaptorGetDeviceCount,
      hipAdaptorGetVendor,
      NULL, // sdcclResult_t (*hostGetDevicePointer)(void **pDevice, void
            // *pHost);
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      hipAdaptorGdrMemAlloc, hipAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      hipAdaptorStreamCreate, hipAdaptorStreamDestroy, hipAdaptorStreamCopy,
      hipAdaptorStreamFree, hipAdaptorStreamSynchronize, hipAdaptorStreamQuery,
      hipAdaptorStreamWaitEvent, hipAdaptorStreamWaitValue64,
      hipAdaptorStreamWriteValue64,
      // Event functions
      hipAdaptorEventCreate, hipAdaptorEventDestroy, hipAdaptorEventRecord,
      hipAdaptorEventSynchronize, hipAdaptorEventQuery,
      hipAdaptorEventElapsedTime,
      // IpcMemHandle functions
      hipAdaptorIpcMemHandleCreate, hipAdaptorIpcMemHandleGet,
      hipAdaptorIpcMemHandleOpen, hipAdaptorIpcMemHandleClose,
      hipAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      hipAdaptorLaunchDeviceFunc, // sdcclResult_t
                                  // (*launchDeviceFunc)(sdcclStream_t stream,
                                  // void *args);
      // Others
      hipAdaptorGetDeviceProperties, // sdcclResult_t
                                     // (*getDeviceProperties)(struct
                                     // sdcclDevProps *props, int dev);
      hipAdaptorGetDevicePciBusId,   // sdcclResult_t (*getDevicePciBusId)(char
                                     // *pciBusId, int len, int dev);
      hipAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                     // (*getDeviceByPciBusId)(int
                                     // *dev, const char *pciBusId);
      hipAdaptorLaunchHostFunc,
      // DMA buffer
      hipAdaptorDmaSupport, // sdcclResult_t (*dmaSupport)(bool
                            // *dmaBufferSupport);
      hipAdaptorMemGetHandleForAddressRange, // sdcclResult_t
                                             // (*memGetHandleForAddressRange)(void
                                             // *handleOut, void *buffer,
                                             // size_t size, unsigned long long
                                             // flags);
};

#endif // USE_AMD_ADAPTOR
