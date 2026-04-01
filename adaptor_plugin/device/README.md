# SDCCL Device Adaptor Plugin Documentation

This page describes the SDCCL Device Adaptor plugin API and how to implement a device adaptor plugin for SDCCL.

## Overview

SDCCL supports external device adaptor plugins to allow custom device runtime implementations without modifying the SDCCL source tree. Plugins implement the SDCCL device adaptor API as a shared library (`.so`), which SDCCL loads at runtime via `dlopen`.

When a plugin is loaded, it replaces the built-in device adaptor (`deviceAdaptor`) and rebuilds the cached `globalDeviceHandle` so that all subsequent device operations use the plugin's implementations.

## Plugin Architecture

### Loading

SDCCL looks for a plugin when the `SDCCL_DEVICE_ADAPTOR_PLUGIN` environment variable is set. The value can be:

- An absolute or relative path to a `.so` file (e.g. `./libsdccl-device-myplugin.so`)
- `none` to explicitly disable plugin loading

If the variable is unset, no plugin is loaded and the built-in adaptor is used.

### Symbol Versioning

Once the library is loaded, SDCCL looks for a symbol named `sdcclDeviceAdaptorPlugin_v1`. This versioned naming allows future API changes while maintaining backwards compatibility.

The symbol must be a `struct sdcclDeviceAdaptor` instance with `visibility("default")` so that `dlsym` can find it.

### Lifecycle

The device adaptor plugin is initialized during `sdcclHandleInit()` (before the device handle is copied) and finalized during `sdcclHandleFree()` (after freeing the device handle). A reference count ensures the plugin stays loaded when multiple handles exist.

## Building a Plugin

### Headers

Plugins should copy the required SDCCL headers into their own source tree to avoid build-time dependency on the full SDCCL source. The example plugin demonstrates this pattern with a local `sdccl/` directory containing:

- `sdccl.h` — Core types and error codes
- `sdccl_device_adaptor.h` — The `sdcclDeviceAdaptor` struct and plugin symbol macro
- **Platform adaptor header** — Copy the vendor adaptor header corresponding to your target platform from `sdccl/adaptor/include/`. For example, `nvidia_adaptor.h` for NVIDIA/CUDA. This header provides struct definitions for `sdcclStream`, `sdcclEvent`, `sdcclIpcMemHandle`, `sdcclWindow`, etc. Note: `sdcclDevProps` is defined in `sdccl_device_adaptor.h`, not in the vendor platform header — do not rely on the vendor header for `sdcclDevProps`.

When copying the vendor adaptor header, **remove the `#ifdef USE_XXX_ADAPTOR` / `#endif` guard**. Since your plugin targets a specific platform, the platform choice is implicit — adding the guard would require an unnecessary `-DUSE_XXX_ADAPTOR` flag in your Makefile. See `example/sdccl/nvidia_adaptor.h` and `cuda/sdccl/nvidia_adaptor.h` for reference.

### Compilation

Plugins must be compiled as shared libraries with `-fPIC`. Using `-fvisibility=hidden` is recommended to avoid exporting internal symbols, with only the plugin symbol marked visible:

```c
__attribute__((visibility("default")))
struct sdcclDeviceAdaptor SDCCL_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 = {
    "MyPlugin",
    myDeviceSynchronize, myDeviceMemcpy, myDeviceMemset,
    ...
};
```

A minimal Makefile:

```makefile
build: libsdccl-device-myplugin.so

libsdccl-device-myplugin.so: plugin.cc
	g++ -Isdccl -fPIC -shared -o $@ $^

clean:
	rm -f libsdccl-device-myplugin.so
```

## API (v1)

Below is the `sdcclDeviceAdaptor` struct with all members (1 name + function pointers).

```c
struct sdcclDeviceAdaptor {
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
```

### Validation

When loading a plugin, SDCCL validates that `name` is non-empty and the function pointers that all built-in adaptors implement are non-NULL:
- `name[0] != '\0'`
- Basic: `deviceSynchronize`, `deviceMemcpy`, `deviceMemset`, `deviceMalloc`, `deviceFree`, `setDevice`, `getDevice`, `getDeviceCount`, `getVendor`
- GDR: `gdrMemAlloc`, `gdrMemFree`
- Stream: `streamCreate`, `streamDestroy`, `streamCopy`, `streamFree`, `streamSynchronize`, `streamQuery`, `streamWaitEvent`, `streamWaitValue64`, `streamWriteValue64`
- Event: `eventCreate`, `eventDestroy`, `eventRecord`, `eventSynchronize`, `eventQuery`, `eventElapsedTime`
- IPC: `ipcMemHandleCreate`, `ipcMemHandleGet`, `ipcMemHandleOpen`, `ipcMemHandleClose`, `ipcMemHandleFree`
- Other: `launchHostFunc`

The following fields are **not** validated because some built-in adaptors leave them NULL: `hostGetDevicePointer`, `memHandleInit`, `memHandleDestroy`, `hostShareMemAlloc`, `hostShareMemFree`, `gdrPtrMmap`, `gdrPtrMunmap`, `launchKernel`, `copyArgsInit`, `copyArgsFree`, `launchDeviceFunc`, `getDeviceProperties`, `getDevicePciBusId`, `getDeviceByPciBusId`, `dmaSupport`, `getHandleForAddressRange`.

If any required field is missing, the plugin is not loaded and SDCCL falls back to the built-in adaptor.

### Error Codes

All plugin functions return `sdcclResult_t`. Return `sdcclSuccess` on success.

- `sdcclSuccess` — Operation completed successfully.
- `sdcclUnhandledDeviceError` — A device runtime call failed.
- `sdcclSystemError` — A system call failed.
- `sdcclInternalError` — An internal logic error or unsupported operation.

## Examples

### Example Plugin (Skeleton)

The `example/` directory contains a minimal skeleton plugin where all operations return `sdcclInternalError`. It demonstrates the required file structure, headers, and export symbol.

### CUDA Plugin

The `cuda/` directory contains a real plugin wrapping CUDA runtime APIs. It can be used as a reference for implementing device plugins for other platforms.

### Build and Test

```bash
# Build the example plugin (no dependencies)
cd adaptor_plugin/device/example
make

# Run with the plugin (plugin loads but operations will fail)
SDCCL_DEVICE_ADAPTOR_PLUGIN=./adaptor_plugin/device/example/libsdccl-device-example.so \
  SDCCL_DEBUG=INFO <your_app>

# Expect log output:
#   ADAPTOR/Plugin: Loaded device adaptor plugin 'Example'

# Build the CUDA plugin
cd adaptor_plugin/device/cuda
CUDA_HOME=/usr/local/cuda make

# Run with CUDA plugin
SDCCL_DEVICE_ADAPTOR_PLUGIN=./adaptor_plugin/device/cuda/libsdccl-device-cuda.so \
  SDCCL_DEBUG=INFO <your_app>

# Disable plugin
SDCCL_DEVICE_ADAPTOR_PLUGIN=none <your_app>

# Test with bad path (warning logged, fallback to default)
SDCCL_DEVICE_ADAPTOR_PLUGIN=/nonexistent.so SDCCL_DEBUG=INFO <your_app>
```
