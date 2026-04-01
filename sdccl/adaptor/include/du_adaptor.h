#ifdef USE_DU_ADAPTOR

#include "sdccl.h"
#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  ncclComm_t base;
};

struct sdcclStream {
  cudaStream_t base;
};

struct sdcclEvent {
  cudaEvent_t base;
};

struct sdcclIpcMemHandle {
  cudaIpcMemHandle_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // USE_DU_ADAPTOR