/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "core.h"
#include "sdccl_ccl_adaptor.h"

#include <dlfcn.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

static void *cclPluginDlHandle = NULL;
static int cclPluginRefCount = 0;
static std::mutex cclPluginMutex;
static struct sdcclCCLAdaptor *cclDefaultDeviceAdaptor = NULL;
extern struct sdcclCCLAdaptor *cclAdaptors[];

sdcclResult_t sdcclCCLAdaptorPluginLoad() {
  // Already loaded — nothing to do.
  if (cclPluginDlHandle != NULL) {
    return sdcclSuccess;
  }

  const char *envValue = getenv("SDCCL_CCL_ADAPTOR_PLUGIN");
  if (envValue == NULL || strcmp(envValue, "none") == 0) {
    return sdcclSuccess;
  }

  cclPluginDlHandle = sdcclAdaptorOpenPluginLib(envValue);
  if (cclPluginDlHandle == NULL) {
    WARN("ADAPTOR/Plugin: Failed to open CCL adaptor plugin '%s'", envValue);
    return sdcclSuccess;
  }

  // Future: When v2 is introduced, try dlsym("sdcclCCLAdaptorPlugin_v2")
  // first, then fall back to "sdcclCCLAdaptorPlugin_v1" and wrap in a v1→v2
  // shim.
  struct sdcclCCLAdaptor *plugin = (struct sdcclCCLAdaptor *)dlsym(
      cclPluginDlHandle, "sdcclCCLAdaptorPlugin_v1");
  if (plugin == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'sdcclCCLAdaptorPlugin_v1' in "
         "'%s': %s",
         envValue, dlerror());
    sdcclAdaptorClosePluginLib(cclPluginDlHandle);
    cclPluginDlHandle = NULL;
    return sdcclSuccess;
  }

  // Validate all 34 function pointers
  if (plugin->name == NULL || plugin->getVersion == NULL ||
      plugin->getUniqueId == NULL || plugin->getErrorString == NULL ||
      plugin->getLastError == NULL || plugin->getStagedBuffer == NULL ||
      plugin->commInitRank == NULL || plugin->commFinalize == NULL ||
      plugin->commDestroy == NULL || plugin->commAbort == NULL ||
      plugin->commResume == NULL || plugin->commSuspend == NULL ||
      plugin->commCount == NULL || plugin->commGetDeviceNumber == NULL ||
      plugin->commUserRank == NULL || plugin->commGetAsyncError == NULL ||
      plugin->memAlloc == NULL || plugin->memFree == NULL ||
      plugin->commRegister == NULL || plugin->commDeregister == NULL ||
      plugin->commWindowRegister == NULL ||
      plugin->commWindowDeregister == NULL || plugin->reduce == NULL ||
      plugin->gather == NULL || plugin->scatter == NULL ||
      plugin->broadcast == NULL || plugin->allReduce == NULL ||
      plugin->reduceScatter == NULL || plugin->allGather == NULL ||
      plugin->alltoAll == NULL || plugin->alltoAllv == NULL ||
      plugin->send == NULL || plugin->recv == NULL ||
      plugin->groupStart == NULL || plugin->groupEnd == NULL) {
    WARN("ADAPTOR/Plugin: CCL adaptor plugin '%s' is missing required function "
         "pointers",
         envValue);
    sdcclAdaptorClosePluginLib(cclPluginDlHandle);
    cclPluginDlHandle = NULL;
    return sdcclSuccess;
  }

  cclDefaultDeviceAdaptor = cclAdaptors[sdcclCCLAdaptorDevice];
  cclAdaptors[sdcclCCLAdaptorDevice] = plugin;
  INFO(SDCCL_INIT, "ADAPTOR/Plugin: Loaded CCL adaptor plugin '%s'",
       plugin->name);
  return sdcclSuccess;
}

sdcclResult_t sdcclCCLAdaptorPluginUnload() {
  if (cclDefaultDeviceAdaptor != NULL) {
    cclAdaptors[sdcclCCLAdaptorDevice] = cclDefaultDeviceAdaptor;
    cclDefaultDeviceAdaptor = NULL;
  }
  sdcclAdaptorClosePluginLib(cclPluginDlHandle);
  cclPluginDlHandle = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclCCLAdaptorPluginInit() {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  sdcclCCLAdaptorPluginLoad();
  if (cclPluginDlHandle != NULL) {
    cclPluginRefCount++;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclCCLAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  if (cclPluginRefCount > 0 && --cclPluginRefCount == 0) {
    INFO(SDCCL_INIT, "Unloading CCL adaptor plugin");
    sdcclCCLAdaptorPluginUnload();
  }
  return sdcclSuccess;
}
