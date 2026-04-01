#ifdef USE_KUNLUNXIN_ADAPTOR

#include "kunlunxin_adaptor.h"

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

sdcclResult_t kunlunAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

sdcclResult_t kunlunAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorDeviceMalloc(void **ptr, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMalloc(ptr, size));
    } else {
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == sdcclMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "KUNLUNXIN");
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGdrMemAlloc(void **ptr, size_t size,
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

sdcclResult_t kunlunAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(cudaFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGdrPtrMmap(void **pcpuptr, void *devptr,
                                       size_t sz) {
  if (pcpuptr == NULL || devptr == NULL || sz == 0) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(baidu::xpu::bkcl::xccl_mmap(pcpuptr, devptr, sz));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGdrPtrMunmap(void *cpuptr, size_t sz) {
  if (cpuptr == NULL || sz == 0) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(baidu::xpu::bkcl::xccl_munmap(cpuptr, sz));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamCopy(sdcclStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamQuery(sdcclStream_t stream) {
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

sdcclResult_t kunlunAdaptorStreamWaitEvent(sdcclStream_t stream,
                                            sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorEventCreate(sdcclEvent_t *event,
                                        sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorEventRecord(sdcclEvent_t event,
                                        sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecordWithFlags(event->base, stream->base,
                                        cudaEventRecordDefault));
    } else {
      DEVCHECK(cudaEventRecordWithFlags(event->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorEventQuery(sdcclEvent_t event) {
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

sdcclResult_t kunlunAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                               size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorLaunchHostFunc(sdcclStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorLaunchDeviceFunc(sdcclStream_t stream,
                                             sdcclLaunchFunc_t fn,
                                             void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGetDeviceProperties(struct sdcclDevProps *props,
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

sdcclResult_t kunlunAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t kunlunAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                              int) {
  return sdcclNotSupported;
}
sdcclResult_t kunlunAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                               int) {
  return sdcclNotSupported;
}
sdcclResult_t kunlunAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                             sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor kunlunAdaptor {
  "KUNLUN",
      // Basic functions
      kunlunAdaptorDeviceSynchronize, kunlunAdaptorDeviceMemcpy,
      kunlunAdaptorDeviceMemset, kunlunAdaptorDeviceMalloc,
      kunlunAdaptorDeviceFree, kunlunAdaptorSetDevice, kunlunAdaptorGetDevice,
      kunlunAdaptorGetDeviceCount, kunlunAdaptorGetVendor,
      kunlunAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      kunlunAdaptorGdrMemAlloc, kunlunAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      kunlunAdaptorGdrPtrMmap,   // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr,
                                 // void *devptr, size_t sz);
      kunlunAdaptorGdrPtrMunmap, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr,
                                 // size_t sz);
      // Stream functions
      kunlunAdaptorStreamCreate, kunlunAdaptorStreamDestroy,
      kunlunAdaptorStreamCopy, kunlunAdaptorStreamFree,
      kunlunAdaptorStreamSynchronize, kunlunAdaptorStreamQuery,
      kunlunAdaptorStreamWaitEvent, kunlunAdaptorStreamWaitValue64,
      kunlunAdaptorStreamWriteValue64,
      // Event functions
      kunlunAdaptorEventCreate, kunlunAdaptorEventDestroy,
      kunlunAdaptorEventRecord, kunlunAdaptorEventSynchronize,
      kunlunAdaptorEventQuery, kunlunAdaptorEventElapsedTime,
      // IpcMemHandle functions
      kunlunAdaptorIpcMemHandleCreate, kunlunAdaptorIpcMemHandleGet,
      kunlunAdaptorIpcMemHandleOpen, kunlunAdaptorIpcMemHandleClose,
      kunlunAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      kunlunAdaptorLaunchDeviceFunc,
      // Others
      kunlunAdaptorGetDeviceProperties, // sdcclResult_t
                                        // (*getDeviceProperties)(struct
                                        // sdcclDevProps *props, int dev);
      kunlunAdaptorGetDevicePciBusId,   // sdcclResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      kunlunAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                        // (*getDeviceByPciBusId)(int
                                        // *dev, const char *pciBusId);
      kunlunAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_KUNLUNXIN_ADAPTOR