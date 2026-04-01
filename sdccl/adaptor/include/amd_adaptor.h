#ifdef USE_AMD_ADAPTOR

#include "sdccl.h"
#include "rccl.h"
#include <hip/hip_runtime.h>
#include <map>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  ncclComm_t base;
};

struct sdcclStream {
  hipStream_t base;
};

struct sdcclEvent {
  hipEvent_t base;
};

struct sdcclIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  do {                                                                         \
    int ret = func;                                                            \
    if (ret != hipSuccess)                                                     \
      return sdcclUnhandledDeviceError;                                       \
  } while (0);

#define CCLCHECKGOTO(call, RES, label)                                         \
  do {                                                                         \
    RES = call;                                                                \
    if (RES != ncclSuccess && RES != ncclInProgress) {                         \
      /* Print the back trace*/                                                \
      if (sdcclDebugNoWarn == 0)                                              \
        INFO(SDCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#endif // USE_AMD_ADAPTOR
