/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_ADAPTOR_PLUGIN_LOAD_H_
#define SDCCL_ADAPTOR_PLUGIN_LOAD_H_

#include "sdccl.h"

// ---- Shared utility functions (used by per-type plugin loaders) ----

// Open a plugin library by path (e.g., the value of an env var).
// Calls dlopen on the given path. Returns handle or NULL.
void *sdcclAdaptorOpenPluginLib(const char *path);

// Close a previously opened plugin library.
sdcclResult_t sdcclAdaptorClosePluginLib(void *handle);

// ---- Per-type plugin load/unload (implemented in ccl/, device/, net/) ----

// CCL adaptor plugin loading (ccl/ccl_plugin_load.cc)
// Reads SDCCL_CCL_ADAPTOR_PLUGIN, overrides
// cclAdaptors[sdcclCCLAdaptorDevice].
sdcclResult_t sdcclCCLAdaptorPluginLoad();
sdcclResult_t sdcclCCLAdaptorPluginUnload();

// Device adaptor plugin loading (device/device_plugin_load.cc)
// Reads SDCCL_DEVICE_ADAPTOR_PLUGIN, overrides deviceAdaptor.
sdcclResult_t sdcclDeviceAdaptorPluginLoad();
sdcclResult_t sdcclDeviceAdaptorPluginUnload();

// Net adaptor plugin loading (net/net_plugin_load.cc)
// Reads SDCCL_NET_ADAPTOR_PLUGIN, populates sdcclNetAdaptors[0].

// ---- Per-type plugin init/finalize (wrap Load/Unload with fallback) ----

// CCL adaptor plugin init/finalize
// Init calls Load, with fallback logic on failure.
// Finalize calls Unload, with best-effort cleanup on failure.
sdcclResult_t sdcclCCLAdaptorPluginInit();
sdcclResult_t sdcclCCLAdaptorPluginFinalize();

// Device adaptor plugin init/finalize
sdcclResult_t sdcclDeviceAdaptorPluginInit();
sdcclResult_t sdcclDeviceAdaptorPluginFinalize();

// Net adaptor plugin init/finalize
sdcclResult_t sdcclNetAdaptorPluginInit();
sdcclResult_t sdcclNetAdaptorPluginFinalize();

// Top-level orchestrators removed: each plugin type (device, ccl, net)
// has different lifecycle requirements and will be initialized/finalized
// at the appropriate stage in later phases.

#endif // SDCCL_ADAPTOR_PLUGIN_LOAD_H_
