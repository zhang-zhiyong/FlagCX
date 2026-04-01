/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_NET_H_
#define SDCCL_NET_H_

#include "net_device.h"
#include <stdint.h>

#define SDCCL_NET_HANDLE_MAXSIZE 128

#define SDCCL_PTR_HOST 0x1
#define SDCCL_PTR_CUDA 0x2
#define SDCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define SDCCL_NET_MAX_REQUESTS 32

#define SDCCL_NET_MAX_DEVS_PER_NIC 4

typedef struct {
  int ndevs;
  int devs[SDCCL_NET_MAX_DEVS_PER_NIC];
} sdcclNetVDeviceProps_t;

typedef struct {
  char *name;      // Used mostly for logging.
  char *pciPath;   // Path to the PCI device in /sys.
  uint64_t guid;   // Unique identifier for the NIC chip. Important for
                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;  // [SDCCL_PTR_HOST|SDCCL_PTR_CUDA|SDCCL_PTR_DMABUF]
  int regIsGlobal; // regMr is not tied to a particular comm
  int speed;       // Port speed in Mbps.
  int port;        // Port number.
  float latency;   // Network latency
  int maxComms;    // Maximum number of comms we can create
  int maxRecvs;    // Maximum number of grouped receives.
  sdcclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;              // Version number for network offload
} sdcclNetProperties_v1_t;

// Version history:
//   v1 (initial, was v8 in NCCL) — name, pciPath, guid, ptrSupport,
//                                   regIsGlobal, speed, port, latency,
//                                   maxComms, maxRecvs, netDeviceType,
//                                   netDeviceVersion

typedef sdcclNetProperties_v1_t sdcclNetProperties_t;

#endif // end include guard
