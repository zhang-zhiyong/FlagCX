# SDCCL CCL Adaptor Plugin Documentation

This page describes the SDCCL CCL Adaptor plugin API and how to implement a device-side CCL plugin for SDCCL.

## Overview

SDCCL supports external CCL (Collective Communication Library) plugins to allow custom device-side collective implementations without modifying the SDCCL source tree. Plugins implement the SDCCL CCL adaptor API as a shared library (`.so`), which SDCCL loads at runtime via `dlopen`.

When a plugin is loaded, it replaces the built-in device-side CCL adaptor (index 1 in the `cclAdaptors` array). The host-side adaptor (bootstrap/gloo/MPI at index 0) is never affected by the plugin.

## Plugin Architecture

### Loading

SDCCL looks for a plugin when the `SDCCL_CCL_ADAPTOR_PLUGIN` environment variable is set. The value can be:

- An absolute or relative path to a `.so` file (e.g. `./libsdccl-ccl-myplugin.so`)
- `none` to explicitly disable plugin loading

If the variable is unset, no plugin is loaded and the built-in adaptor is used.

### Symbol Versioning

Once the library is loaded, SDCCL looks for a symbol named `sdcclCCLAdaptorPlugin_v1`. This versioned naming allows future API changes while maintaining backwards compatibility.

The symbol must be a `struct sdcclCCLAdaptor_v1` instance with `visibility("default")` so that `dlsym` can find it.

### Lifecycle

The CCL adaptor plugin is initialized during `sdcclCommInitRank()` (before any device-side CCL calls) and finalized during `sdcclCommDestroy()` (after all device-side communicators are destroyed). A reference count ensures the plugin stays loaded when multiple communicators exist.

## Building a Plugin

### Headers

Plugins should copy the required SDCCL headers into their own source tree to avoid build-time dependency on the full SDCCL source. The example plugin demonstrates this pattern with a local `sdccl/` directory containing:

- `sdccl.h` — Core types and error codes
- `sdccl_ccl_adaptor.h` — The `sdcclCCLAdaptor_v1` struct and plugin symbol macro
- **Platform adaptor header** — Copy the vendor adaptor header corresponding to your target platform from `sdccl/adaptor/include/`. For example, `nvidia_adaptor.h` for NVIDIA/NCCL. This header provides struct definitions for `sdcclInnerComm`, `sdcclStream`, `sdcclEvent`, `sdcclIpcMemHandle`, `sdcclWindow`, etc.

When copying the vendor adaptor header, **remove the `#ifdef USE_XXX_ADAPTOR` / `#endif` guard**. Since your plugin targets a specific platform, the platform choice is implicit — adding the guard would require an unnecessary `-DUSE_XXX_ADAPTOR` flag in your Makefile. See `example/sdccl/nvidia_adaptor.h` and `nccl/sdccl/nvidia_adaptor.h` for reference.

### Compilation

Plugins must be compiled as shared libraries with `-fPIC`. Using `-fvisibility=hidden` is recommended to avoid exporting internal symbols, with only the plugin symbol marked visible:

```c
__attribute__((visibility("default")))
struct sdcclCCLAdaptor_v1 SDCCL_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 = {
    "MyPlugin",
    myGetVersion, myGetUniqueId, myGetErrorString,
    ...
};
```

A minimal Makefile:

```makefile
build: libsdccl-ccl-myplugin.so

libsdccl-ccl-myplugin.so: plugin.cc
	g++ -Isdccl -fPIC -shared -o $@ $^

clean:
	rm -f libsdccl-ccl-myplugin.so
```

## API (v1)

Below is the `sdcclCCLAdaptor_v1` struct with all 35 members (1 name + 34 function pointers).

```c
struct sdcclCCLAdaptor_v1 {
  const char *name;

  // Basic functions
  sdcclResult_t (*getVersion)(int *version);
  sdcclResult_t (*getUniqueId)(sdcclUniqueId_t *uniqueId);
  const char *(*getErrorString)(sdcclResult_t result);
  const char *(*getLastError)(sdcclInnerComm_t comm);
  sdcclResult_t (*getStagedBuffer)(const sdcclInnerComm_t comm, void **buff,
                                    size_t size, int isRecv);

  // Communicator functions
  sdcclResult_t (*commInitRank)(sdcclInnerComm_t *comm, int nranks,
                                 sdcclUniqueId *commId, int rank,
                                 bootstrapState *bootstrap);
  sdcclResult_t (*commFinalize)(sdcclInnerComm_t comm);
  sdcclResult_t (*commDestroy)(sdcclInnerComm_t comm);
  sdcclResult_t (*commAbort)(sdcclInnerComm_t comm);
  sdcclResult_t (*commResume)(sdcclInnerComm_t comm);
  sdcclResult_t (*commSuspend)(sdcclInnerComm_t comm);
  sdcclResult_t (*commCount)(const sdcclInnerComm_t comm, int *count);
  sdcclResult_t (*commGetDeviceNumber)(const sdcclInnerComm_t comm,
                                        int *device);
  sdcclResult_t (*commUserRank)(const sdcclInnerComm_t comm, int *rank);
  sdcclResult_t (*commGetAsyncError)(sdcclInnerComm_t comm,
                                      sdcclResult_t *asyncError);
  sdcclResult_t (*memAlloc)(void **ptr, size_t size);
  sdcclResult_t (*memFree)(void *ptr);
  sdcclResult_t (*commRegister)(const sdcclInnerComm_t comm, void *buff,
                                 size_t size, void **handle);
  sdcclResult_t (*commDeregister)(const sdcclInnerComm_t comm, void *handle);

  // Symmetric functions
  sdcclResult_t (*commWindowRegister)(sdcclInnerComm_t comm, void *buff,
                                       size_t size, sdcclWindow_t *win,
                                       int winFlags);
  sdcclResult_t (*commWindowDeregister)(sdcclInnerComm_t comm,
                                         sdcclWindow_t win);

  // Communication functions
  sdcclResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, sdcclRedOp_t op,
                           int root, sdcclInnerComm_t comm,
                           sdcclStream_t stream);
  sdcclResult_t (*gather)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, int root,
                           sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*scatter)(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, int root,
                            sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype, int root,
                              sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*allReduce)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype,
                              sdcclRedOp_t op, sdcclInnerComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*reduceScatter)(const void *sendbuff, void *recvbuff,
                                  size_t recvcount, sdcclDataType_t datatype,
                                  sdcclRedOp_t op, sdcclInnerComm_t comm,
                                  sdcclStream_t stream);
  sdcclResult_t (*allGather)(const void *sendbuff, void *recvbuff,
                              size_t sendcount, sdcclDataType_t datatype,
                              sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*alltoAll)(const void *sendbuff, void *recvbuff, size_t count,
                             sdcclDataType_t datatype, sdcclInnerComm_t comm,
                             sdcclStream_t stream);
  sdcclResult_t (*alltoAllv)(const void *sendbuff, size_t *sendcounts,
                              size_t *sdispls, void *recvbuff,
                              size_t *recvcounts, size_t *rdispls,
                              sdcclDataType_t datatype, sdcclInnerComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*send)(const void *sendbuff, size_t count,
                         sdcclDataType_t datatype, int peer,
                         sdcclInnerComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*recv)(void *recvbuff, size_t count,
                         sdcclDataType_t datatype, int peer,
                         sdcclInnerComm_t comm, sdcclStream_t stream);

  // Group semantics
  sdcclResult_t (*groupStart)();
  sdcclResult_t (*groupEnd)();
};
```

### Validation

When loading a plugin, SDCCL validates that all 34 function pointers (and `name`) are non-NULL:
- `name`
- `getVersion`, `getUniqueId`, `getErrorString`, `getLastError`, `getStagedBuffer`
- `commInitRank`, `commFinalize`, `commDestroy`, `commAbort`, `commResume`, `commSuspend`
- `commCount`, `commGetDeviceNumber`, `commUserRank`, `commGetAsyncError`
- `memAlloc`, `memFree`, `commRegister`, `commDeregister`
- `commWindowRegister`, `commWindowDeregister`
- `reduce`, `gather`, `scatter`, `broadcast`, `allReduce`, `reduceScatter`, `allGather`
- `alltoAll`, `alltoAllv`, `send`, `recv`
- `groupStart`, `groupEnd`

If any field is NULL, the plugin is not loaded and SDCCL falls back to the built-in adaptor. Functions that your platform does not support should be implemented as stubs returning `sdcclInternalError` or `sdcclNotSupported`.

### Error Codes

All plugin functions return `sdcclResult_t`. Return `sdcclSuccess` on success.

- `sdcclSuccess` — Operation completed successfully.
- `sdcclSystemError` — A system or hardware call failed.
- `sdcclInternalError` — An internal logic error or unsupported operation.

## Example

The `example/` directory contains a minimal skeleton plugin where all operations return `sdcclInternalError`. It demonstrates the required file structure, headers, and export symbol.

### Build and Test

```bash
# Build the example plugin
cd adaptor_plugin/ccl/example
make

# Run with the plugin (plugin loads but operations will fail)
SDCCL_CCL_ADAPTOR_PLUGIN=./adaptor_plugin/ccl/example/libsdccl-ccl-example.so \
  SDCCL_DEBUG=INFO <your_app>

# Expect log output:
#   ADAPTOR/Plugin: Loaded CCL adaptor plugin 'Example'

# Disable plugin
SDCCL_CCL_ADAPTOR_PLUGIN=none <your_app>
```
