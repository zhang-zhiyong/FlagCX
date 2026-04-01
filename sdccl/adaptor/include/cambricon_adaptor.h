#ifdef USE_CAMBRICON_ADAPTOR
#ifndef SRC_ADAPTOR_API_MLU_ADAPTOR_H
#define SRC_ADAPTOR_API_MLU_ADAPTOR_H

#include "cncl.h"
#include "cnrt.h"
#include "sdccl.h"
#include <map>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  cnclComm_t base;
};

struct sdcclStream {
  cnrtQueue_t base;
};

struct sdcclEvent {
  cnrtNotifier_t base;
};

struct sdcclIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cnrtSuccess)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // SRC_ADAPTOR_API_MLU_ADAPTOR_H
#endif // USE_CAMBRICON_ADAPTOR
