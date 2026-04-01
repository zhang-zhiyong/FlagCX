/*************************************************************************
 * Copyright (c) 2023-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_NET_DEVICE_H_
#define SDCCL_NET_DEVICE_H_

#include <cstddef>

#define SDCCL_NET_DEVICE_INVALID_VERSION 0x0
#define SDCCL_NET_MTU_SIZE 4096

// Arbitrary version number - A given SDCCL build will only be compatible with
// a single device networking plugin version. SDCCL will check the supplied
// version number from net->getProperties() and compare to its internal version.
#define SDCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  SDCCL_NET_DEVICE_HOST = 0,
  SDCCL_NET_DEVICE_UNPACK = 1
} sdcclNetDeviceType;

typedef struct {
  sdcclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
  void *handle;
  size_t size;
  int needsProxyProgress;
} sdcclNetDeviceHandle_v7_t;

typedef sdcclNetDeviceHandle_v7_t sdcclNetDeviceHandle_v8_t;
typedef sdcclNetDeviceHandle_v8_t sdcclNetDeviceHandle_v9_t;
typedef sdcclNetDeviceHandle_v9_t sdcclNetDeviceHandle_v10_t;
typedef sdcclNetDeviceHandle_v10_t sdcclNetDeviceHandle_t;

#endif
