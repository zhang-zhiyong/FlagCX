/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_DEVICE_ADAPTOR_H_
#define SDCCL_DEVICE_ADAPTOR_H_

#include "sdccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device properties — defined here so plugin authors have the full layout.
struct sdcclDevProps {
  char name[256];
  int pciBusId;
  int pciDeviceId;
  int pciDomainId;
};

// C-compatible typedef matching the C++ using alias in dlsymbols.h.
typedef void (*sdcclLaunchFunc_t)(sdcclStream_t, void *);

// Version history:
//   v1 — Initial version with basic device functions, GDR functions,
//         stream/event/IPC functions, kernel launch, device properties,
//         host func launch, DMA buffer, event elapsed time, and
//         stream memory operations.
struct sdcclDeviceAdaptor_v1 {
  char name[32];
  // Basic functions
  sdcclResult_t (*deviceSynchronize)();
  sdcclResult_t (*deviceMemcpy)(void *dst, void *src, size_t size,
                                 sdcclMemcpyType_t type, sdcclStream_t stream,
                                 void *args);
  sdcclResult_t (*deviceMemset)(void *ptr, int value, size_t size,
                                 sdcclMemType_t type, sdcclStream_t stream);
  sdcclResult_t (*deviceMalloc)(void **ptr, size_t size, sdcclMemType_t type,
                                 sdcclStream_t stream);
  sdcclResult_t (*deviceFree)(void *ptr, sdcclMemType_t type,
                               sdcclStream_t stream);
  sdcclResult_t (*setDevice)(int dev);
  sdcclResult_t (*getDevice)(int *dev);
  sdcclResult_t (*getDeviceCount)(int *count);
  sdcclResult_t (*getVendor)(char *vendor);
  sdcclResult_t (*hostGetDevicePointer)(void **pDevice, void *pHost);

  // GDR functions
  sdcclResult_t (*memHandleInit)(int dev_id, void **memHandle);
  sdcclResult_t (*memHandleDestroy)(int dev, void *memHandle);
  sdcclResult_t (*gdrMemAlloc)(void **ptr, size_t size, void *memHandle);
  sdcclResult_t (*gdrMemFree)(void *ptr, void *memHandle);
  sdcclResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void *memHandle);
  sdcclResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
  sdcclResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t sz);
  sdcclResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);

  // Stream functions
  sdcclResult_t (*streamCreate)(sdcclStream_t *stream);
  sdcclResult_t (*streamDestroy)(sdcclStream_t stream);
  sdcclResult_t (*streamCopy)(sdcclStream_t *newStream, void *oldStream);
  sdcclResult_t (*streamFree)(sdcclStream_t stream);
  sdcclResult_t (*streamSynchronize)(sdcclStream_t stream);
  sdcclResult_t (*streamQuery)(sdcclStream_t stream);
  sdcclResult_t (*streamWaitEvent)(sdcclStream_t stream, sdcclEvent_t event);
  sdcclResult_t (*streamWaitValue64)(sdcclStream_t stream, void *addr,
                                      uint64_t value, int flags);
  sdcclResult_t (*streamWriteValue64)(sdcclStream_t stream, void *addr,
                                       uint64_t value, int flags);

  // Event functions
  sdcclResult_t (*eventCreate)(sdcclEvent_t *event,
                                sdcclEventType_t eventType);
  sdcclResult_t (*eventDestroy)(sdcclEvent_t event);
  sdcclResult_t (*eventRecord)(sdcclEvent_t event, sdcclStream_t stream);
  sdcclResult_t (*eventSynchronize)(sdcclEvent_t event);
  sdcclResult_t (*eventQuery)(sdcclEvent_t event);
  sdcclResult_t (*eventElapsedTime)(float *ms, sdcclEvent_t start,
                                     sdcclEvent_t end);

  // IpcMemHandle functions
  sdcclResult_t (*ipcMemHandleCreate)(sdcclIpcMemHandle_t *handle,
                                       size_t *size);
  sdcclResult_t (*ipcMemHandleGet)(sdcclIpcMemHandle_t handle, void *devPtr);
  sdcclResult_t (*ipcMemHandleOpen)(sdcclIpcMemHandle_t handle,
                                     void **devPtr);
  sdcclResult_t (*ipcMemHandleClose)(void *devPtr);
  sdcclResult_t (*ipcMemHandleFree)(sdcclIpcMemHandle_t handle);

  // Kernel launch
  sdcclResult_t (*launchKernel)(void *func, unsigned int block_x,
                                 unsigned int block_y, unsigned int block_z,
                                 unsigned int grid_x, unsigned int grid_y,
                                 unsigned int grid_z, void **args,
                                 size_t share_mem, void *stream,
                                 void *memHandle);
  sdcclResult_t (*copyArgsInit)(void **args);
  sdcclResult_t (*copyArgsFree)(void *args);
  sdcclResult_t (*launchDeviceFunc)(sdcclStream_t stream,
                                     sdcclLaunchFunc_t fn, void *args);

  // Others
  sdcclResult_t (*getDeviceProperties)(struct sdcclDevProps *props, int dev);
  sdcclResult_t (*getDevicePciBusId)(char *pciBusId, int len, int dev);
  sdcclResult_t (*getDeviceByPciBusId)(int *dev, const char *pciBusId);

  // HostFunc launch
  sdcclResult_t (*launchHostFunc)(sdcclStream_t stream, void (*fn)(void *),
                                   void *args);
  // DMA buffer
  sdcclResult_t (*dmaSupport)(bool *dmaBufferSupport);
  sdcclResult_t (*getHandleForAddressRange)(void *handleOut, void *buffer,
                                             size_t size,
                                             unsigned long long flags);
};
#define sdcclDeviceAdaptor sdcclDeviceAdaptor_v1

// Device adaptor plugin API version (independent of CCL/Net versions)
#define SDCCL_DEVICE_ADAPTOR_PLUGIN_VERSION 1

// Versioned export symbol name
#define SDCCL_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 sdcclDeviceAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // SDCCL_DEVICE_ADAPTOR_H_
