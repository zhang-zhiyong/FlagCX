#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {sdcclMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {sdcclMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

sdcclResult_t cudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

sdcclResult_t cudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
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

sdcclResult_t cudaAdaptorDeviceMalloc(void **ptr, size_t size,
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
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
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

sdcclResult_t cudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "NVIDIA");
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
#if 0
#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;
  int cudaDev;
  int flag;

  DEVCHECK(cudaGetDevice(&cudaDev));
  DEVCHECK(cuDeviceGet(&currentDev, cudaDev));

  size_t handleSize = size;
  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  // Query device to see if FABRIC handle support is available
  flag = 0;
  DEVCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
  memprop.location.id = currentDev;
  // Query device to see if RDMA support is available
  flag = 0;
  DEVCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
  DEVCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  ALIGN_SIZE(handleSize, memGran);
  /* Allocate the physical memory on the device */
  DEVCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
  /* Reserve a virtual address range */
  DEVCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, handleSize, memGran, 0, 0));
  /* Map the virtual address range to the physical allocation */
  DEVCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
#endif
#endif
  DEVCHECK(cudaMalloc(ptr, size));
  cudaPointerAttributes attrs;
  DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)attrs.devicePointer));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
#if 0
#if CUDART_VERSION >= 12010
  CUdevice ptrDev = 0;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  DEVCHECK(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr));
  DEVCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  DEVCHECK(cuMemRelease(handle));
  DEVCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  DEVCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  DEVCHECK(cuMemRelease(handle));
  DEVCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
#endif
#endif
  DEVCHECK(cudaFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamCopy(sdcclStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamQuery(sdcclStream_t stream) {
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

sdcclResult_t cudaAdaptorStreamWaitEvent(sdcclStream_t stream,
                                          sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorStreamWaitValue64(sdcclStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return sdcclInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value,
                                     CU_STREAM_WAIT_VALUE_GEQ);
  return (err == CUDA_SUCCESS) ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t cudaAdaptorStreamWriteValue64(sdcclStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return sdcclInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWriteValue64(cuStream, (CUdeviceptr)addr, value,
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  return (err == CUDA_SUCCESS) ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t cudaAdaptorEventCreate(sdcclEvent_t *event,
                                      sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags = (eventType == sdcclEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorEventRecord(sdcclEvent_t event,
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

sdcclResult_t cudaAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorEventQuery(sdcclEvent_t event) {
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

sdcclResult_t cudaAdaptorEventElapsedTime(float *ms, sdcclEvent_t start,
                                           sdcclEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return sdcclInvalidArgument;
  }
  cudaError_t error = cudaEventElapsedTime(ms, start->base, end->base);
  if (error == cudaSuccess) {
    return sdcclSuccess;
  } else if (error == cudaErrorNotReady) {
    return sdcclInProgress;
  } else {
    return sdcclUnhandledDeviceError;
  }
}

sdcclResult_t cudaAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                             size_t *size) {
  sdcclCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorLaunchHostFunc(sdcclStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return sdcclSuccess;
}
sdcclResult_t cudaAdaptorLaunchDeviceFunc(sdcclStream_t stream,
                                           sdcclLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGetDeviceProperties(struct sdcclDevProps *props,
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

sdcclResult_t cudaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return sdcclSuccess;
}

sdcclResult_t cudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return sdcclInvalidArgument;

#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion = 0;

  CUresult cuRes = cuDriverGetVersion(&cudaDriverVersion);
  if (cuRes != CUDA_SUCCESS || cudaDriverVersion < 11070) {
    *dmaBufferSupport = false;
    return sdcclSuccess;
  }

  int deviceId = 0;
  if (cudaGetDevice(&deviceId) != cudaSuccess) {
    *dmaBufferSupport = false;
    return sdcclSuccess;
  }

  CUresult devRes = cuDeviceGet(&dev, deviceId);
  if (devRes != CUDA_SUCCESS) {
    *dmaBufferSupport = false;
    return sdcclSuccess;
  }

  CUresult attrRes =
      cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
  if (attrRes != CUDA_SUCCESS || flag == 0) {
    *dmaBufferSupport = false;
    return sdcclSuccess;
  }

  *dmaBufferSupport = true;
  return sdcclSuccess;

#else
  *dmaBufferSupport = false;
  return sdcclSuccess;
#endif
}

sdcclResult_t
cudaAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  CUdeviceptr dptr = (CUdeviceptr)buffer;
  DEVCHECK(cuMemGetHandleForAddressRange(
      handleOut, dptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags));
  return sdcclSuccess;
}

struct sdcclDeviceAdaptor cudaAdaptor {
  "CUDA",
      // Basic functions
      cudaAdaptorDeviceSynchronize, cudaAdaptorDeviceMemcpy,
      cudaAdaptorDeviceMemset, cudaAdaptorDeviceMalloc, cudaAdaptorDeviceFree,
      cudaAdaptorSetDevice, cudaAdaptorGetDevice, cudaAdaptorGetDeviceCount,
      cudaAdaptorGetVendor, cudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cudaAdaptorGdrMemAlloc, cudaAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      cudaAdaptorStreamCreate, cudaAdaptorStreamDestroy, cudaAdaptorStreamCopy,
      cudaAdaptorStreamFree, cudaAdaptorStreamSynchronize,
      cudaAdaptorStreamQuery, cudaAdaptorStreamWaitEvent,
      cudaAdaptorStreamWaitValue64, cudaAdaptorStreamWriteValue64,
      // Event functions
      cudaAdaptorEventCreate, cudaAdaptorEventDestroy, cudaAdaptorEventRecord,
      cudaAdaptorEventSynchronize, cudaAdaptorEventQuery,
      cudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cudaAdaptorIpcMemHandleCreate, cudaAdaptorIpcMemHandleGet,
      cudaAdaptorIpcMemHandleOpen, cudaAdaptorIpcMemHandleClose,
      cudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // sdcclResult_t (*copyArgsInit)(void **args);
      NULL, // sdcclResult_t (*copyArgsFree)(void *args);
      cudaAdaptorLaunchDeviceFunc, // sdcclResult_t
                                   // (*launchDeviceFunc)(sdcclStream_t stream,
                                   // void *args);
      // Others
      cudaAdaptorGetDeviceProperties, // sdcclResult_t
                                      // (*getDeviceProperties)(struct
                                      // sdcclDevProps *props, int dev);
      cudaAdaptorGetDevicePciBusId, // sdcclResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      cudaAdaptorGetDeviceByPciBusId, // sdcclResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      cudaAdaptorLaunchHostFunc,
      // DMA buffer
      cudaAdaptorDmaSupport, // sdcclResult_t (*dmaSupport)(bool
                             // *dmaBufferSupport);
      cudaAdaptorMemGetHandleForAddressRange, // sdcclResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
};

#endif // USE_NVIDIA_ADAPTOR
