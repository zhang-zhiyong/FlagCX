/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All
 *Rights Reserved.
 ************************************************************************/

#ifdef USE_KUNLUNXIN_ADAPTOR

#include <map>

#include <bkcl.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sdccl.h"

struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  BKCLContext_t base;
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

namespace baidu {
namespace xpu {
namespace bkcl {

// External declaration
extern int xccl_mmap(void **pcpuptr, void *devptr, size_t sz);
extern int xccl_munmap(void *cpuptr, size_t sz);

} // namespace bkcl
} // namespace xpu
} // namespace baidu

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }

#endif // USE_KUNLUNXIN_ADAPTOR