/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "core.h"
#include "sdccl_device_adaptor.h"

#include <dlfcn.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

static void *devicePluginDlHandle = NULL;
static int devicePluginRefCount = 0;
static std::mutex devicePluginMutex;
static struct sdcclDeviceAdaptor *defaultDeviceAdaptor = NULL;
extern struct sdcclDeviceAdaptor *deviceAdaptor;

// Defined in sdccl.cc — rebuilds globalDeviceHandle from current deviceAdaptor
extern void sdcclRebuildGlobalDeviceHandle();

sdcclResult_t sdcclDeviceAdaptorPluginLoad() {
  // Already loaded — nothing to do.
  if (devicePluginDlHandle != NULL) {
    return sdcclSuccess;
  }

  const char *envValue = getenv("SDCCL_DEVICE_ADAPTOR_PLUGIN");
  if (envValue == NULL || strcmp(envValue, "none") == 0) {
    return sdcclSuccess;
  }

  devicePluginDlHandle = sdcclAdaptorOpenPluginLib(envValue);
  if (devicePluginDlHandle == NULL) {
    WARN("ADAPTOR/Plugin: Failed to open device adaptor plugin '%s'", envValue);
    return sdcclSuccess;
  }

  struct sdcclDeviceAdaptor *plugin = (struct sdcclDeviceAdaptor *)dlsym(
      devicePluginDlHandle, "sdcclDeviceAdaptorPlugin_v1");
  if (plugin == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'sdcclDeviceAdaptorPlugin_v1' "
         "in '%s': %s",
         envValue, dlerror());
    sdcclAdaptorClosePluginLib(devicePluginDlHandle);
    devicePluginDlHandle = NULL;
    return sdcclSuccess;
  }

  // Validate function pointers that all built-in adaptors implement.
  // Fields that some adaptors leave NULL (hostGetDevicePointer, memHandleInit,
  // memHandleDestroy, hostShareMemAlloc, hostShareMemFree, gdrPtrMmap,
  // gdrPtrMunmap, launchKernel, copyArgsInit, copyArgsFree, launchDeviceFunc,
  // getDeviceProperties, getDevicePciBusId, getDeviceByPciBusId, dmaSupport,
  // getHandleForAddressRange) are intentionally not checked here.
  if (plugin->name[0] == '\0' || plugin->deviceSynchronize == NULL ||
      plugin->deviceMemcpy == NULL || plugin->deviceMemset == NULL ||
      plugin->deviceMalloc == NULL || plugin->deviceFree == NULL ||
      plugin->setDevice == NULL || plugin->getDevice == NULL ||
      plugin->getDeviceCount == NULL || plugin->getVendor == NULL ||
      plugin->gdrMemAlloc == NULL || plugin->gdrMemFree == NULL ||
      plugin->streamCreate == NULL || plugin->streamDestroy == NULL ||
      plugin->streamCopy == NULL || plugin->streamFree == NULL ||
      plugin->streamSynchronize == NULL || plugin->streamQuery == NULL ||
      plugin->streamWaitEvent == NULL || plugin->streamWaitValue64 == NULL ||
      plugin->streamWriteValue64 == NULL || plugin->eventCreate == NULL ||
      plugin->eventDestroy == NULL || plugin->eventRecord == NULL ||
      plugin->eventSynchronize == NULL || plugin->eventQuery == NULL ||
      plugin->eventElapsedTime == NULL || plugin->ipcMemHandleCreate == NULL ||
      plugin->ipcMemHandleGet == NULL || plugin->ipcMemHandleOpen == NULL ||
      plugin->ipcMemHandleClose == NULL || plugin->ipcMemHandleFree == NULL ||
      plugin->launchHostFunc == NULL) {
    WARN("ADAPTOR/Plugin: Device adaptor plugin '%s' is missing required "
         "function pointers",
         envValue);
    sdcclAdaptorClosePluginLib(devicePluginDlHandle);
    devicePluginDlHandle = NULL;
    return sdcclSuccess;
  }

  defaultDeviceAdaptor = deviceAdaptor;
  deviceAdaptor = plugin;
  sdcclRebuildGlobalDeviceHandle();
  INFO(SDCCL_INIT, "ADAPTOR/Plugin: Loaded device adaptor plugin '%s'",
       plugin->name);
  return sdcclSuccess;
}

sdcclResult_t sdcclDeviceAdaptorPluginUnload() {
  if (defaultDeviceAdaptor != NULL) {
    deviceAdaptor = defaultDeviceAdaptor;
    defaultDeviceAdaptor = NULL;
    sdcclRebuildGlobalDeviceHandle();
  }
  sdcclAdaptorClosePluginLib(devicePluginDlHandle);
  devicePluginDlHandle = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclDeviceAdaptorPluginInit() {
  std::lock_guard<std::mutex> lock(devicePluginMutex);
  sdcclDeviceAdaptorPluginLoad();
  if (devicePluginDlHandle != NULL) {
    devicePluginRefCount++;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclDeviceAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(devicePluginMutex);
  if (devicePluginRefCount > 0 && --devicePluginRefCount == 0) {
    INFO(SDCCL_INIT, "Unloading device adaptor plugin");
    sdcclDeviceAdaptorPluginUnload();
  }
  return sdcclSuccess;
}
