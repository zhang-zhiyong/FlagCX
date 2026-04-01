#include "ascend_adaptor.h"

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<sdcclMemcpyType_t, aclrtMemcpyKind> memcpy_type_map = {
    {sdcclMemcpyHostToDevice, ACL_MEMCPY_HOST_TO_DEVICE},
    {sdcclMemcpyDeviceToHost, ACL_MEMCPY_DEVICE_TO_HOST},
    {sdcclMemcpyDeviceToDevice, ACL_MEMCPY_DEVICE_TO_DEVICE},
};

sdcclResult_t cannAdaptorDeviceSynchronize() {
  DEVCHECK(aclrtSynchronizeDevice());
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       sdcclMemcpyType_t type,
                                       sdcclStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(aclrtMemcpy(dst, size, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(aclrtMemcpyAsync(dst, size, src, size, memcpy_type_map[type],
                              stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(aclrtMemset(ptr, size, value, size));
    } else {
      DEVCHECK(aclrtMemsetAsync(ptr, size, value, size, stream->base));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorDeviceMalloc(void **ptr, size_t size,
                                       sdcclMemType_t type,
                                       sdcclStream_t stream) {

  if (type == sdcclMemHost) {
    DEVCHECK(aclrtMallocHost(ptr, size));
  } else {
    DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorDeviceFree(void *ptr, sdcclMemType_t type,
                                     sdcclStream_t stream) {
  if (type == sdcclMemHost) {
    DEVCHECK(aclrtFreeHost(ptr));
  } else {
    DEVCHECK(aclrtFree(ptr));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorSetDevice(int dev) {
  DEVCHECK(aclrtSetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorGetDevice(int *dev) {
  DEVCHECK(aclrtGetDevice(dev));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorGetDeviceCount(int *count) {
  DEVCHECK(aclrtGetDeviceCount((uint32_t *)count));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return sdcclSuccess;
}
// TODO:unsupport
sdcclResult_t cannAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return sdcclInvalidArgument;
  }
  DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  return sdcclSuccess;
}

// TODO:unsupported
sdcclResult_t cannAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return sdcclSuccess;
  }
  DEVCHECK(aclrtFree(ptr));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamCreate(sdcclStream_t *stream) {
  (*stream) = NULL;
  sdcclCalloc(stream, 1);
  DEVCHECK(aclrtCreateStream((aclrtStream *)(*stream)));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamDestroy(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtDestroyStream(stream->base));
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamCopy(sdcclStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  sdcclCalloc(newStream, 1);
  (*newStream)->base = (aclrtStream)oldStream;
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamFree(sdcclStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamSynchronize(sdcclStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtSynchronizeStream(stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamQuery(sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  if (stream != NULL) {
    aclrtStreamStatus status;
    DEVCHECK(aclrtStreamQuery(stream->base, &status));
    if (status == ACL_STREAM_STATUS_COMPLETE) {
      res = sdcclSuccess;
    } else if (status == ACL_STREAM_STATUS_NOT_READY) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t cannAdaptorStreamWaitEvent(sdcclStream_t stream,
                                          sdcclEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(aclrtStreamWaitEvent(stream->base, event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorEventCreate(sdcclEvent_t *event,
                                      sdcclEventType_t eventType) {
  (*event) = NULL;
  sdcclCalloc(event, 1);
  const unsigned int flags =
      (eventType == sdcclEventDefault) ? ACL_EVENT_TIME_LINE : ACL_EVENT_SYNC;
  DEVCHECK(aclrtCreateEventWithFlag(&((*event)->base), flags));
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorEventDestroy(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtDestroyEvent(event->base));
    free(event);
    event = NULL;
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorEventRecord(sdcclEvent_t event,
                                      sdcclStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(aclrtRecordEvent(event->base, stream->base));
    } else {
      return sdcclUnhandledDeviceError;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorEventSynchronize(sdcclEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtSynchronizeEvent(event->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorEventQuery(sdcclEvent_t event) {
  sdcclResult_t res = sdcclSuccess;
  if (event != NULL) {
    aclrtEventWaitStatus status;
    DEVCHECK(aclrtQueryEventWaitStatus(event->base, &status));
    if (status == ACL_EVENT_WAIT_STATUS_COMPLETE) {
      res = sdcclSuccess;
    } else if (status == ACL_EVENT_WAIT_STATUS_NOT_READY) {
      res = sdcclInProgress;
    } else {
      res = sdcclUnhandledDeviceError;
    }
  }
  return res;
}

sdcclResult_t cannAdaptorIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                             size_t *size) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t cannAdaptorIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                          void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t cannAdaptorIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                           void **devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t cannAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t cannAdaptorIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  // to be implemented
  return sdcclNotSupported;
}

sdcclResult_t cannAdaptorLaunchHostFunc(sdcclStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(
        aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base));
  }
  return sdcclSuccess;
}

sdcclResult_t cannAdaptorStreamWaitValue64(sdcclStream_t, void *, uint64_t,
                                            int) {
  return sdcclNotSupported;
}
sdcclResult_t cannAdaptorStreamWriteValue64(sdcclStream_t, void *, uint64_t,
                                             int) {
  return sdcclNotSupported;
}
sdcclResult_t cannAdaptorEventElapsedTime(float *, sdcclEvent_t,
                                           sdcclEvent_t) {
  return sdcclNotSupported;
}

struct sdcclDeviceAdaptor cannAdaptor {
  "CANN",
      // Basic functions
      cannAdaptorDeviceSynchronize, cannAdaptorDeviceMemcpy,
      cannAdaptorDeviceMemset, cannAdaptorDeviceMalloc, cannAdaptorDeviceFree,
      cannAdaptorSetDevice, cannAdaptorGetDevice, cannAdaptorGetDeviceCount,
      cannAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cannAdaptorGdrMemAlloc, cannAdaptorGdrMemFree,
      NULL, // sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      cannAdaptorStreamCreate, cannAdaptorStreamDestroy, cannAdaptorStreamCopy,
      cannAdaptorStreamFree, cannAdaptorStreamSynchronize,
      cannAdaptorStreamQuery, cannAdaptorStreamWaitEvent,
      cannAdaptorStreamWaitValue64, cannAdaptorStreamWriteValue64,
      // Event functions
      cannAdaptorEventCreate, cannAdaptorEventDestroy, cannAdaptorEventRecord,
      cannAdaptorEventSynchronize, cannAdaptorEventQuery,
      cannAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cannAdaptorIpcMemHandleCreate, cannAdaptorIpcMemHandleGet,
      cannAdaptorIpcMemHandleOpen, cannAdaptorIpcMemHandleClose,
      cannAdaptorIpcMemHandleFree,
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
      NULL, // sdcclResult_t (*getDeviceProperties)(struct sdcclDevProps
            // *props, int dev);
      NULL, // sdcclResult_t (*getDevicePciBusId)(char
            // *pciBusId, int len, int dev);
      NULL, // sdcclResult_t
            // (*getDeviceByPciBusId)(int
            // *dev, const char *pciBusId);
      cannAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // sdcclResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_ASCEND_ADAPTOR
