/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "core.h"
#include "sdccl_net.h"
#include "param.h"
#include <stdarg.h>
#include <stdlib.h>
#include <sys/syscall.h>

int sdcclDebugLevel = -1;
static int pid = -1;
static char hostname[1024];
thread_local int sdcclDebugNoWarn = 0;
char sdcclLastError[1024] =
    ""; // Global string for the last error in human readable form
uint64_t sdcclDebugMask =
    SDCCL_INIT | SDCCL_ENV; // Default debug sub-system mask is INIT and ENV
FILE *sdcclDebugFile = stdout;
pthread_mutex_t sdcclDebugLock = PTHREAD_MUTEX_INITIALIZER;
std::chrono::steady_clock::time_point sdcclEpoch;

static __thread int tid = -1;

void sdcclDebugInit() {
  pthread_mutex_lock(&sdcclDebugLock);
  if (sdcclDebugLevel != -1) {
    pthread_mutex_unlock(&sdcclDebugLock);
    return;
  }
  const char *sdccl_debug = sdcclGetEnv("SDCCL_DEBUG");
  int tempNcclDebugLevel = -1;
  if (sdccl_debug == NULL) {
    tempNcclDebugLevel = SDCCL_LOG_NONE;
  } else if (strcasecmp(sdccl_debug, "VERSION") == 0) {
    tempNcclDebugLevel = SDCCL_LOG_VERSION;
  } else if (strcasecmp(sdccl_debug, "WARN") == 0) {
    tempNcclDebugLevel = SDCCL_LOG_WARN;
  } else if (strcasecmp(sdccl_debug, "INFO") == 0) {
    tempNcclDebugLevel = SDCCL_LOG_INFO;
  } else if (strcasecmp(sdccl_debug, "ABORT") == 0) {
    tempNcclDebugLevel = SDCCL_LOG_ABORT;
  } else if (strcasecmp(sdccl_debug, "TRACE") == 0) {
    tempNcclDebugLevel = SDCCL_LOG_TRACE;
  }

  /* Parse the SDCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char *sdcclDebugSubsysEnv = sdcclGetEnv("SDCCL_DEBUG_SUBSYS");
  if (sdcclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (sdcclDebugSubsysEnv[0] == '^') {
      invert = 1;
      sdcclDebugSubsysEnv++;
    }
    sdcclDebugMask = invert ? ~0ULL : 0ULL;
    char *sdcclDebugSubsys = strdup(sdcclDebugSubsysEnv);
    char *subsys = strtok(sdcclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = SDCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = SDCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = SDCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = SDCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = SDCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = SDCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = SDCCL_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = SDCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = SDCCL_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = SDCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = SDCCL_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = SDCCL_NVLS;
      } else if (strcasecmp(subsys, "BOOTSTRAP") == 0) {
        mask = SDCCL_BOOTSTRAP;
      } else if (strcasecmp(subsys, "REG") == 0) {
        mask = SDCCL_REG;
      } else if (strcasecmp(subsys, "KERNEL") == 0) {
        mask = SDCCL_KERNEL;
      } else if (strcasecmp(subsys, "UNIRUNNER") == 0) {
        mask = SDCCL_UNIRUNNER;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = SDCCL_ALL;
      }
      if (mask) {
        if (invert)
          sdcclDebugMask &= ~mask;
        else
          sdcclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(sdcclDebugSubsys);
  }

  // Cache pid and hostname
  getHostName(hostname, 1024, '.');
  pid = getpid();

  /* Parse and expand the SDCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * SDCCL_DEBUG level is > VERSION
   */
  const char *sdcclDebugFileEnv = sdcclGetEnv("SDCCL_DEBUG_FILE");
  if (tempNcclDebugLevel > SDCCL_LOG_VERSION && sdcclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX + 1] = "";
    char *dfn = debugFn;
    while (sdcclDebugFileEnv[c] != '\0' && c < PATH_MAX) {
      if (sdcclDebugFileEnv[c++] != '%') {
        *dfn++ = sdcclDebugFileEnv[c - 1];
        continue;
      }
      switch (sdcclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = sdcclDebugFileEnv[c - 1];
          break;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setbuf(file, nullptr); // disable buffering
        sdcclDebugFile = file;
      }
    }
  }

  sdcclEpoch = std::chrono::steady_clock::now();
  __atomic_store_n(&sdcclDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&sdcclDebugLock);
}

SDCCL_PARAM(WarnSetDebugInfo, "WARN_ENABLE_DEBUG_INFO", 0);

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void sdcclDebugLog(sdcclDebugLogLevel level, unsigned long flags,
                    const char *filefunc, int line, const char *fmt, ...) {
  if (__atomic_load_n(&sdcclDebugLevel, __ATOMIC_ACQUIRE) == -1)
    sdcclDebugInit();
  if (sdcclDebugNoWarn != 0 && level == SDCCL_LOG_WARN) {
    level = SDCCL_LOG_INFO;
    flags = sdcclDebugNoWarn;
  }

  // Save the last error (WARN) as a human readable string
  if (level == SDCCL_LOG_WARN) {
    pthread_mutex_lock(&sdcclDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void)vsnprintf(sdcclLastError, sizeof(sdcclLastError), fmt, vargs);
    va_end(vargs);
    pthread_mutex_unlock(&sdcclDebugLock);
  }
  if (sdcclDebugLevel < level || ((flags & sdcclDebugMask) == 0))
    return;

  if (tid == -1) {
    tid = syscall(SYS_gettid);
  }

  int cudaDev = 0;
  /**
   * TODO: How to get the GPU currently in use
   **/

  char buffer[1024];
  size_t len = 0;
  if (level == SDCCL_LOG_WARN) {
    len = snprintf(buffer, sizeof(buffer), "\n%s:%d:%d [%d] %s:%d SDCCL WARN ",
                   hostname, pid, tid, cudaDev, filefunc, line);
    if (sdcclParamWarnSetDebugInfo())
      sdcclDebugLevel = SDCCL_LOG_INFO;
  } else if (level == SDCCL_LOG_INFO) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] SDCCL INFO ",
                   hostname, pid, tid, cudaDev);
  } else if (level == SDCCL_LOG_TRACE && flags == SDCCL_CALL) {
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d SDCCL CALL ", hostname,
                   pid, tid);
  } else if (level == SDCCL_LOG_TRACE) {
    auto delta = std::chrono::steady_clock::now() - sdcclEpoch;
    double timestamp =
        std::chrono::duration_cast<std::chrono::duration<double>>(delta)
            .count() *
        1000;
    len =
        snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] %f %s:%d SDCCL TRACE ",
                 hostname, pid, tid, cudaDev, timestamp, filefunc, line);
  }

  if (len) {
    va_list vargs;
    va_start(vargs, fmt);
    len += vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
    va_end(vargs);
    // vsnprintf may return len > sizeof(buffer) in the case of a truncated
    // output. Rewind len so that we can replace the final \0 by \n
    if (len > sizeof(buffer))
      len = sizeof(buffer) - 1;
    buffer[len++] = '\n';
    fwrite(buffer, 1, len, sdcclDebugFile);
  }
}

SDCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void sdcclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (sdcclParamSetThreadName() != 1)
    return;
  char threadName[SDCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, SDCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}
