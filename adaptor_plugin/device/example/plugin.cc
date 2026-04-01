/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example device adaptor plugin for SDCCL.
 * This is a minimal skeleton: all operations return sdcclInternalError,
 * so this plugin is only useful for verifying that the loading mechanism
 * works. A real plugin would wrap a device runtime (e.g. CUDA, CANN).
 ************************************************************************/

#include "sdccl/sdccl_device_adaptor.h"
#include "sdccl/nvidia_adaptor.h"

static sdcclResult_t pluginDeviceSynchronize() { return sdcclInternalError; }

static sdcclResult_t pluginDeviceMemcpy(void *dst, void *src, size_t size,
                                         sdcclMemcpyType_t type,
                                         sdcclStream_t stream, void *args) {
  return sdcclInternalError;
}

static sdcclResult_t pluginDeviceMemset(void *ptr, int value, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginDeviceMalloc(void **ptr, size_t size,
                                         sdcclMemType_t type,
                                         sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginDeviceFree(void *ptr, sdcclMemType_t type,
                                       sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginSetDevice(int dev) { return sdcclInternalError; }

static sdcclResult_t pluginGetDevice(int *dev) { return sdcclInternalError; }

static sdcclResult_t pluginGetDeviceCount(int *count) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetVendor(char *vendor) {
  return sdcclInternalError;
}

static sdcclResult_t pluginHostGetDevicePointer(void **pDevice, void *pHost) {
  return sdcclInternalError;
}

// GDR functions
static sdcclResult_t pluginMemHandleInit(int dev_id, void **memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginMemHandleDestroy(int dev, void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGdrMemFree(void *ptr, void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginHostShareMemAlloc(void **ptr, size_t size,
                                              void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginHostShareMemFree(void *ptr, void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGdrPtrMmap(void **pcpuptr, void *devptr,
                                       size_t sz) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGdrPtrMunmap(void *cpuptr, size_t sz) {
  return sdcclInternalError;
}

// Stream functions
static sdcclResult_t pluginStreamCreate(sdcclStream_t *stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamDestroy(sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamCopy(sdcclStream_t *newStream,
                                       void *oldStream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamFree(sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamSynchronize(sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamQuery(sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamWaitEvent(sdcclStream_t stream,
                                            sdcclEvent_t event) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamWaitValue64(sdcclStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  return sdcclInternalError;
}

static sdcclResult_t pluginStreamWriteValue64(sdcclStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  return sdcclInternalError;
}

// Event functions
static sdcclResult_t pluginEventCreate(sdcclEvent_t *event,
                                        sdcclEventType_t eventType) {
  return sdcclInternalError;
}

static sdcclResult_t pluginEventDestroy(sdcclEvent_t event) {
  return sdcclInternalError;
}

static sdcclResult_t pluginEventRecord(sdcclEvent_t event,
                                        sdcclStream_t stream) {
  return sdcclInternalError;
}

static sdcclResult_t pluginEventSynchronize(sdcclEvent_t event) {
  return sdcclInternalError;
}

static sdcclResult_t pluginEventQuery(sdcclEvent_t event) {
  return sdcclInternalError;
}

static sdcclResult_t pluginEventElapsedTime(float *ms, sdcclEvent_t start,
                                             sdcclEvent_t end) {
  return sdcclInternalError;
}

// IpcMemHandle functions
static sdcclResult_t pluginIpcMemHandleCreate(sdcclIpcMemHandle_t *handle,
                                               size_t *size) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIpcMemHandleGet(sdcclIpcMemHandle_t handle,
                                            void *devPtr) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIpcMemHandleOpen(sdcclIpcMemHandle_t handle,
                                             void **devPtr) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIpcMemHandleClose(void *devPtr) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIpcMemHandleFree(sdcclIpcMemHandle_t handle) {
  return sdcclInternalError;
}

// Kernel launch
static sdcclResult_t
pluginLaunchKernel(void *func, unsigned int block_x, unsigned int block_y,
                   unsigned int block_z, unsigned int grid_x,
                   unsigned int grid_y, unsigned int grid_z, void **args,
                   size_t share_mem, void *stream, void *memHandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCopyArgsInit(void **args) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCopyArgsFree(void *args) {
  return sdcclInternalError;
}

static sdcclResult_t pluginLaunchDeviceFunc(sdcclStream_t stream,
                                             sdcclLaunchFunc_t fn,
                                             void *args) {
  return sdcclInternalError;
}

// Others
static sdcclResult_t pluginGetDeviceProperties(struct sdcclDevProps *props,
                                                int dev) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  return sdcclInternalError;
}

// HostFunc launch
static sdcclResult_t pluginLaunchHostFunc(sdcclStream_t stream,
                                           void (*fn)(void *), void *args) {
  return sdcclInternalError;
}

// DMA buffer
static sdcclResult_t pluginDmaSupport(bool *dmaBufferSupport) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetHandleForAddressRange(void *handleOut,
                                                     void *buffer, size_t size,
                                                     unsigned long long flags) {
  return sdcclInternalError;
}

__attribute__((visibility("default"))) struct sdcclDeviceAdaptor
    SDCCL_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",
        // Basic functions
        pluginDeviceSynchronize,
        pluginDeviceMemcpy,
        pluginDeviceMemset,
        pluginDeviceMalloc,
        pluginDeviceFree,
        pluginSetDevice,
        pluginGetDevice,
        pluginGetDeviceCount,
        pluginGetVendor,
        pluginHostGetDevicePointer,
        // GDR functions
        pluginMemHandleInit,
        pluginMemHandleDestroy,
        pluginGdrMemAlloc,
        pluginGdrMemFree,
        pluginHostShareMemAlloc,
        pluginHostShareMemFree,
        pluginGdrPtrMmap,
        pluginGdrPtrMunmap,
        // Stream functions
        pluginStreamCreate,
        pluginStreamDestroy,
        pluginStreamCopy,
        pluginStreamFree,
        pluginStreamSynchronize,
        pluginStreamQuery,
        pluginStreamWaitEvent,
        pluginStreamWaitValue64,
        pluginStreamWriteValue64,
        // Event functions
        pluginEventCreate,
        pluginEventDestroy,
        pluginEventRecord,
        pluginEventSynchronize,
        pluginEventQuery,
        pluginEventElapsedTime,
        // IpcMemHandle functions
        pluginIpcMemHandleCreate,
        pluginIpcMemHandleGet,
        pluginIpcMemHandleOpen,
        pluginIpcMemHandleClose,
        pluginIpcMemHandleFree,
        // Kernel launch
        pluginLaunchKernel,
        pluginCopyArgsInit,
        pluginCopyArgsFree,
        pluginLaunchDeviceFunc,
        // Others
        pluginGetDeviceProperties,
        pluginGetDevicePciBusId,
        pluginGetDeviceByPciBusId,
        // HostFunc launch
        pluginLaunchHostFunc,
        // DMA buffer
        pluginDmaSupport,
        pluginGetHandleForAddressRange,
};
