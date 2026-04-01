/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example net adaptor plugin for SDCCL.
 * This is a minimal skeleton: it reports 0 devices so the runtime
 * will fall back to a built-in adaptor (IBRC or Socket).
 ************************************************************************/

#include "sdccl/sdccl_net.h"
#include "sdccl/sdccl_net_adaptor.h"

#include <string.h>

static sdcclResult_t pluginInit() { return sdcclSuccess; }

static sdcclResult_t pluginDevices(int *ndev) {
  *ndev = 0;
  return sdcclSuccess;
}

// Reference implementation showing the structure of sdcclNetProperties_t
// and the usage of different memory pointer types.
static sdcclResult_t pluginGetProperties(int dev, void *props) {
  sdcclNetProperties_t *p = (sdcclNetProperties_t *)props;
  memset(p, 0, sizeof(*p));

  // Human-readable device name (used in log messages)
  p->name = (char *)"ExampleNet0";

  // sysfs PCI path, e.g. "/sys/devices/pci0000:00/0000:00:01.0"
  // Set to NULL when no PCI device backs this network interface.
  p->pciPath = NULL;

  // Globally unique identifier for the NIC chip.
  // Important for multi-function cards (physical or SR-IOV virtual).
  p->guid = 0;

  // Bitmask of memory pointer types this device can send/recv from directly:
  //   SDCCL_PTR_HOST   - host (CPU) pinned memory
  //   SDCCL_PTR_CUDA   - device (GPU) memory (requires GPUDirect / peer
  //   access) SDCCL_PTR_DMABUF - DMA-BUF imported memory (requires kernel
  //   DMA-BUF support)
  // A pure CPU-based transport typically supports only SDCCL_PTR_HOST.
  // An RDMA transport with GPUDirect would set SDCCL_PTR_HOST |
  // SDCCL_PTR_CUDA.
  p->ptrSupport = SDCCL_PTR_HOST;

  // Set to 1 if regMr registrations are global (not tied to a particular comm).
  // When true, a single registration can be reused across multiple connections.
  p->regIsGlobal = 0;

  // Link speed in Mbps (e.g. 100000 for 100 Gbps)
  p->speed = 0;

  // Physical port number (0-based)
  p->port = 0;

  // One-way network latency in microseconds
  p->latency = 0;

  // Maximum number of concurrent comm objects (send + recv) this device
  // supports. Use 1 if the device has no such limit.
  p->maxComms = 1;

  // Maximum number of grouped receive operations in a single irecv call.
  // Use 1 if grouped receives are not supported.
  p->maxRecvs = 1;

  // Network device offload type:
  //   SDCCL_NET_DEVICE_HOST   - all processing on host CPU (default)
  //   SDCCL_NET_DEVICE_UNPACK - device-side unpack offload supported
  p->netDeviceType = SDCCL_NET_DEVICE_HOST;

  // Version number for the device offload protocol.
  // Set to SDCCL_NET_DEVICE_INVALID_VERSION when netDeviceType is HOST.
  p->netDeviceVersion = SDCCL_NET_DEVICE_INVALID_VERSION;

  return sdcclSuccess;
}

static sdcclResult_t pluginListen(int dev, void *handle, void **listenComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginConnect(int dev, void *handle, void **sendComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginAccept(void *listenComm, void **recvComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCloseSend(void *sendComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCloseRecv(void *recvComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginCloseListen(void *listenComm) {
  return sdcclInternalError;
}

static sdcclResult_t pluginRegMr(void *comm, void *data, size_t size, int type,
                                  int mrFlags, void **mhandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginRegMrDmaBuf(void *comm, void *data, size_t size,
                                        int type, uint64_t offset, int fd,
                                        int mrFlags, void **mhandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginDeregMr(void *comm, void *mhandle) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIsend(void *sendComm, void *data, size_t size,
                                  int tag, void *mhandle, void *phandle,
                                  void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIrecv(void *recvComm, int n, void **data,
                                  size_t *sizes, int *tags, void **mhandles,
                                  void **phandles, void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIflush(void *recvComm, int n, void **data,
                                   int *sizes, void **mhandles,
                                   void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginTest(void *request, int *done, int *sizes) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIput(void *sendComm, uint64_t srcOff,
                                 uint64_t dstOff, size_t size, int srcRank,
                                 int dstRank, void **srcHandles,
                                 void **dstHandles, void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIget(void *sendComm, uint64_t srcOff,
                                 uint64_t dstOff, size_t size, int srcRank,
                                 int dstRank, void **srcHandles,
                                 void **dstHandles, void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginIputSignal(void *sendComm, uint64_t srcOff,
                                       uint64_t dstOff, size_t size,
                                       int srcRank, int dstRank,
                                       void **srcHandles, void **dstHandles,
                                       uint64_t signalOff, void **signalHandles,
                                       uint64_t signalValue, void **request) {
  return sdcclInternalError;
}

static sdcclResult_t pluginGetDevFromName(char *name, int *dev) {
  return sdcclInternalError;
}

__attribute__((visibility("default"))) struct sdcclNetAdaptor_v1
    SDCCL_NET_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",           pluginInit,       pluginDevices,
        pluginGetProperties, pluginListen,     pluginConnect,
        pluginAccept,        pluginCloseSend,  pluginCloseRecv,
        pluginCloseListen,   pluginRegMr,      pluginRegMrDmaBuf,
        pluginDeregMr,       pluginIsend,      pluginIrecv,
        pluginIflush,        pluginTest,       pluginIput,
        pluginIget,          pluginIputSignal, pluginGetDevFromName,
};
