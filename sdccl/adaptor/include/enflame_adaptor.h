/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#ifdef USE_ENFLAME_ADAPTOR

#include "eccl.h"
#include "sdccl.h"
#include <map>
#include <tops/tops_runtime_api.h>

struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  ecclComm_t base;
};

struct sdcclStream {
  topsStream_t base;
};

struct sdcclEvent {
  topsEvent_t base;
};

struct sdcclIpcMemHandle {
  topsIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != topsSuccess)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // USE_ENFLAME_ADAPTOR
