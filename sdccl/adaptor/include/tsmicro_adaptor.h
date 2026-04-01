#ifdef USE_TSM_ADAPTOR

#ifndef TSMICRO_ADAPTOR_H_
#define TSMICRO_ADAPTOR_H_

#include "sdccl.h"
#include "tccl.h"
#include "tx_runtime.h"

#include <map>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  tcclComm_t base;
};

struct sdcclStream {
  txStream_t base;
};

struct sdcclEvent {
  txEvent_t base;
};

struct sdcclIpcMemHandle {
  txIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != TX_SUCCESS)                                                     \
      return sdcclUnhandledDeviceError;                                       \
  }
#endif // end include guard
#endif // USE_TSM_ADAPTOR