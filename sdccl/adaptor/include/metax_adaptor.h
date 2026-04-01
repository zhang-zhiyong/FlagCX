/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#ifdef USE_METAX_ADAPTOR

#include "sdccl.h"
#include "mccl.h"
#include <map>
#include <mcr/mc_runtime.h>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  mcclComm_t base;
};

struct sdcclStream {
  mcStream_t base;
};

struct sdcclEvent {
  mcEvent_t base;
};

struct sdcclIpcMemHandle {
  mcIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != mcSuccess)                                                      \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // USE_METAX_ADAPTOR
