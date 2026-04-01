/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/

#ifndef SDCCL_ADAPTOR_H_
#define SDCCL_ADAPTOR_H_

#include "bootstrap.h"
#include "device_utils.h"
#include "sdccl.h"
#include "global_comm.h"
#include "topo.h"
#include "utils.h"

// Struct definitions are now in per-type public headers
#include "sdccl_ccl_adaptor.h"
#include "sdccl_device_adaptor.h"
#include "sdccl_net_adaptor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NCCLADAPTORS 2
#define sdcclCCLAdaptorHost 0
#define sdcclCCLAdaptorDevice 1

extern struct sdcclCCLAdaptor bootstrapAdaptor;
extern struct sdcclCCLAdaptor glooAdaptor;
extern struct sdcclCCLAdaptor mpiAdaptor;
extern struct sdcclCCLAdaptor ncclAdaptor;
extern struct sdcclCCLAdaptor hcclAdaptor;
extern struct sdcclCCLAdaptor ixncclAdaptor;
extern struct sdcclCCLAdaptor cnclAdaptor;
extern struct sdcclCCLAdaptor mcclAdaptor;
extern struct sdcclCCLAdaptor musa_mcclAdaptor;
extern struct sdcclCCLAdaptor xcclAdaptor;
extern struct sdcclCCLAdaptor duncclAdaptor;
extern struct sdcclCCLAdaptor rcclAdaptor;
extern struct sdcclCCLAdaptor tcclAdaptor;
extern struct sdcclCCLAdaptor ecclAdaptor;
extern struct sdcclCCLAdaptor *cclAdaptors[];

extern struct sdcclDeviceAdaptor cudaAdaptor;
extern struct sdcclDeviceAdaptor cannAdaptor;
extern struct sdcclDeviceAdaptor ixcudaAdaptor;
extern struct sdcclDeviceAdaptor mluAdaptor;
extern struct sdcclDeviceAdaptor macaAdaptor;
extern struct sdcclDeviceAdaptor musaAdaptor;
extern struct sdcclDeviceAdaptor kunlunAdaptor;
extern struct sdcclDeviceAdaptor ducudaAdaptor;
extern struct sdcclDeviceAdaptor hipAdaptor;
extern struct sdcclDeviceAdaptor tsmicroAdaptor;
extern struct sdcclDeviceAdaptor topsAdaptor;
extern struct sdcclDeviceAdaptor *deviceAdaptor;

extern struct sdcclNetAdaptor *netAdaptor;

// Network type enumeration
enum NetType {
  IBRC = 1,   // InfiniBand RC (or UCX when USE_UCX=1)
  SOCKET = 2, // Socket
#ifdef USE_IBUC
  IBUC = 3 // InfiniBand UC
#endif
};

// Unified network adaptor function declarations
struct sdcclNetAdaptor *getUnifiedNetAdaptor(int netType);

inline bool sdcclCCLAdaptorNeedSendrecv(size_t value) { return value != 0; }

const int MAX_VENDOR_LEN = 128;
typedef struct {
  char internal[MAX_VENDOR_LEN];
} sdcclVendor;

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
