#ifdef USE_ASCEND_ADAPTOR
#include "acl/acl.h"
#include "sdccl.h"
#include "hccl/hccl.h"
#include <map>
struct sdcclInnerDevComm {};

struct sdcclInnerComm {
  HcclComm base;
};

struct sdcclStream {
  aclrtStream base;
};

struct sdcclEvent {
  aclrtEvent base;
};

struct sdcclIpcMemHandle {
  char *base; // to be implemented
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != ACL_SUCCESS)                                                    \
      return sdcclUnhandledDeviceError;                                       \
  }
#endif // USE_ASCEND_ADAPTOR
