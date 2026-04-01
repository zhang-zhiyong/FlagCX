#include "iluvatar_corex_adaptor.h"

#ifdef USE_ILUVATAR_COREX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

sdcclResult_t ixcudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                         sdcclMemcpyType_t type,
                                         sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(cudaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        cudaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorDeviceMalloc(void **ptr, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cudaMallocHost(ptr, size));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMalloc(ptr, size));
    } else {
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ILUVATAR_COREX");
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaMalloc(ptr, size));
  cudaPointerAttributes attrs;
  DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)attrs.devicePointer));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(cudaFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamCopy(sdcclStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    cudaError error = cudaStreamQuery(stream->base);
    if (error == cudaSuccess) {
      res = sdcclSuccess;
    } else if (error == cudaErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t ixcudaAdaptorStreamWaitEvent(sdcclStream_t stream,
                                            sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(cudaStreamWaitEvent(stream->base, event->base, 0));
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorEventCreate(sdcclEvent_t *event,
                                        sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));

  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorEventRecord(sdcclEvent_t event,
                                        sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecord(event->base, stream->base));
    } else {
      DEVCHECK(cudaEventRecord(event->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    cudaError error = cudaEventQuery(event->base);
    if (error == cudaSuccess) {
      res = sdcclSuccess;
    } else if (error == cudaErrorNotReady) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t ixcudaAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                               size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorLaunchHostFunc(sdcclStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetDeviceProperties(struct sdcclDevProps *props,
                                                int dev) {
  if (props == NULL) {
    return sdcclInvalidArgument;
  }

  cudaDeviceProp devProp;
  DEVCHECK(cudaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some cuda versions,
  // cudaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t ixcudaAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                              int) {
  return sdcclNotSupported;
}
sdcclResult_t ixcudaAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                               int) {
  return sdcclNotSupported;
}
sdcclResult_t ixcudaAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                             sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor ixcudaAdaptor {
  "IXCUDA",
      // Basic functions
      ixcudaAdaptorDeviceSynchronize, ixcudaAdaptorDeviceMemcpy,
      ixcudaAdaptorDeviceMemset, ixcudaAdaptorDeviceMalloc,
      ixcudaAdaptorDeviceFree, ixcudaAdaptorSetDevice, ixcudaAdaptorGetDevice,
      ixcudaAdaptorGetDeviceCount, ixcudaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ixcudaAdaptorGdrMemAlloc, ixcudaAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ixcudaAdaptorStreamCreate, ixcudaAdaptorStreamDestroy,
      ixcudaAdaptorStreamCopy, ixcudaAdaptorStreamFree,
      ixcudaAdaptorStreamSynchronize, ixcudaAdaptorStreamQuery,
      ixcudaAdaptorStreamWaitEvent, ixcudaAdaptorStreamWaitValue64,
      ixcudaAdaptorStreamWriteValue64,
      // Event functions
      ixcudaAdaptorEventCreate, ixcudaAdaptorEventDestroy,
      ixcudaAdaptorEventRecord, ixcudaAdaptorEventSynchronize,
      ixcudaAdaptorEventQuery, ixcudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ixcudaAdaptorIpcMemHandleCreate, ixcudaAdaptorIpcMemHandleGet,
      ixcudaAdaptorIpcMemHandleOpen, ixcudaAdaptorIpcMemHandleClose,
      ixcudaAdaptorIpcMemHandleFree,
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
      ixcudaAdaptorGetDeviceProperties, // sdcclResult_t
                                        // (*getDeviceProperties)(struct
                                        // sdcclDeviceProps *props, int dev);
      ixcudaAdaptorGetDevicePciBusId,   // sdcclResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      ixcudaAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                        // (*getDeviceByPciBusId)(int *dev,
                                        // const char *pciBusId);
      ixcudaAdaptorLaunchHostFunc,

      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_ILUVATAR_COREX_ADAPTOR