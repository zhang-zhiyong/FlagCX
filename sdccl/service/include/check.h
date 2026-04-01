/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_CHECKS_H_
#define SDCCL_CHECKS_H_

#include "debug.h"
#include "type.h"
#include <errno.h>

// Check system calls
#define SYSCHECK(call, name)                                                   \
  do {                                                                         \
    int retval;                                                                \
    SYSCHECKVAL(call, name, retval);                                           \
  } while (false)

#define SYSCHECKVAL(call, name, retval)                                        \
  do {                                                                         \
    SYSCHECKSYNC(call, name, retval);                                          \
    if (retval == -1) {                                                        \
      WARN("Call to " name " failed : %s", strerror(errno));                   \
      return sdcclSystemError;                                                \
    }                                                                          \
  } while (false)

#define SYSCHECKSYNC(call, name, retval)                                       \
  do {                                                                         \
    retval = call;                                                             \
    if (retval == -1 &&                                                        \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      INFO(SDCCL_ALL, "Call to " name " returned %s, retrying",               \
           strerror(errno));                                                   \
    } else {                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (true)

#define SYSCHECKGOTO(statement, RES, label)                                    \
  do {                                                                         \
    if ((statement) == -1) {                                                   \
      /* Print the back trace*/                                                \
      RES = sdcclSystemError;                                                 \
      INFO(SDCCL_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define NEQCHECK(statement, value)                                             \
  do {                                                                         \
    if ((statement) != value) {                                                \
      /* Print the back trace*/                                                \
      INFO(SDCCL_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__,                 \
           sdcclSystemError, strerror(errno));                                \
      return sdcclSystemError;                                                \
    }                                                                          \
  } while (0);

#define NEQCHECKGOTO(statement, value, RES, label)                             \
  do {                                                                         \
    if ((statement) != value) {                                                \
      /* Print the back trace*/                                                \
      RES = sdcclSystemError;                                                 \
      INFO(SDCCL_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define EQCHECK(statement, value)                                              \
  do {                                                                         \
    if ((statement) == value) {                                                \
      /* Print the back trace*/                                                \
      INFO(SDCCL_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__,                 \
           sdcclSystemError, strerror(errno));                                \
      return sdcclSystemError;                                                \
    }                                                                          \
  } while (0);

#define EQCHECKGOTO(statement, value, RES, label)                              \
  do {                                                                         \
    if ((statement) == value) {                                                \
      /* Print the back trace*/                                                \
      RES = sdcclSystemError;                                                 \
      INFO(SDCCL_ALL, "%s:%d -> %d (%s)", __FILE__, __LINE__, RES,            \
           strerror(errno));                                                   \
      goto label;                                                              \
    }                                                                          \
  } while (0);

// Propagate errors up
#define SDCCLCHECK(call)                                                      \
  do {                                                                         \
    sdcclResult_t RES = call;                                                 \
    if (RES != sdcclSuccess && RES != sdcclInProgress) {                     \
      /* Print the back trace*/                                                \
      if (sdcclDebugNoWarn == 0)                                              \
        INFO(SDCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      return RES;                                                              \
    }                                                                          \
  } while (0);

#define SDCCLCHECKGOTO(call, RES, label)                                      \
  do {                                                                         \
    RES = call;                                                                \
    if (RES != sdcclSuccess && RES != sdcclInProgress) {                     \
      /* Print the back trace*/                                                \
      if (sdcclDebugNoWarn == 0)                                              \
        INFO(SDCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
  } while (0);

#define SDCCLWAIT(call, cond, abortFlagPtr)                                   \
  do {                                                                         \
    volatile uint32_t *tmpAbortFlag = (abortFlagPtr);                          \
    sdcclResult_t RES = call;                                                 \
    if (RES != sdcclSuccess && RES != sdcclInProgress) {                     \
      if (sdcclDebugNoWarn == 0)                                              \
        INFO(SDCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      return sdcclInternalError;                                              \
    }                                                                          \
    if (tmpAbortFlag)                                                          \
      NEQCHECK(*tmpAbortFlag, 0);                                              \
  } while (!(cond));

#define SDCCLWAITGOTO(call, cond, abortFlagPtr, RES, label)                   \
  do {                                                                         \
    volatile uint32_t *tmpAbortFlag = (abortFlagPtr);                          \
    RES = call;                                                                \
    if (RES != sdcclSuccess && RES != sdcclInProgress) {                     \
      if (sdcclDebugNoWarn == 0)                                              \
        INFO(SDCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);              \
      goto label;                                                              \
    }                                                                          \
    if (tmpAbortFlag)                                                          \
      NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label);                              \
  } while (!(cond));

#define SDCCLCHECKTHREAD(a, args)                                             \
  do {                                                                         \
    if (((args)->ret = (a)) != sdcclSuccess &&                                \
        (args)->ret != sdcclInProgress) {                                     \
      INFO(SDCCL_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,      \
           (args)->ret);                                                       \
      return args;                                                             \
    }                                                                          \
  } while (0)

#define CUDACHECKTHREAD(a)                                                     \
  do {                                                                         \
    if ((a) != cudaSuccess) {                                                  \
      INFO(SDCCL_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,      \
           args->ret);                                                         \
      args->ret = sdcclUnhandledCudaError;                                    \
      return args;                                                             \
    }                                                                          \
  } while (0)

// Type definitions
#define PTHREADCHECKGOTO(statement, name, RES, label)                          \
  do {                                                                         \
    int retval = (statement);                                                  \
    if (retval != 0) {                                                         \
      WARN("Call to " name " failed: %s", strerror(retval));                   \
      RES = sdcclSystemError;                                                 \
      goto label;                                                              \
    }                                                                          \
  } while (0)
#endif
