#ifdef USE_MUSA_ADAPTOR

#include "sdccl.h"
#include "mccl.h"
#include <map>
#include <musa.h>
#include <musa_runtime.h>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  mcclComm_t base;
};

struct sdcclStream {
  musaStream_t base;
};

struct sdcclEvent {
  musaEvent_t base;
};

struct sdcclIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != musaSuccess)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // USE_MUSA_ADAPTOR