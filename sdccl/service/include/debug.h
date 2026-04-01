/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_INT_DEBUG_H_
#define SDCCL_INT_DEBUG_H_

#include "type.h"
#include <chrono>
#include <stdio.h>
#include <type_traits>

#include <limits.h>
#include <pthread.h>
#include <string.h>

typedef enum {
  SDCCL_LOG_NONE = 0,
  SDCCL_LOG_VERSION = 1,
  SDCCL_LOG_WARN = 2,
  SDCCL_LOG_INFO = 3,
  SDCCL_LOG_ABORT = 4,
  SDCCL_LOG_TRACE = 5
} sdcclDebugLogLevel;
typedef enum {
  SDCCL_INIT = 1,
  SDCCL_COLL = 2,
  SDCCL_P2P = 4,
  SDCCL_SHM = 8,
  SDCCL_NET = 16,
  SDCCL_GRAPH = 32,
  SDCCL_TUNING = 64,
  SDCCL_ENV = 128,
  SDCCL_ALLOC = 256,
  SDCCL_CALL = 512,
  SDCCL_PROXY = 1024,
  SDCCL_NVLS = 2048,
  SDCCL_BOOTSTRAP = 4096,
  SDCCL_REG = 8192,
  SDCCL_KERNEL = 16384,
  SDCCL_UNIRUNNER = 32768,
  SDCCL_ALL = ~0
} sdcclDebugLogSubSys;

// Conform to pthread and NVTX standard
#define SDCCL_THREAD_NAMELEN 16

extern int sdcclDebugLevel;
extern uint64_t sdcclDebugMask;
extern pthread_mutex_t sdcclDebugLock;
extern FILE *sdcclDebugFile;
extern sdcclResult_t getHostName(char *hostname, int maxlen, const char delim);

void sdcclDebugLog(sdcclDebugLogLevel level, unsigned long flags,
                    const char *filefunc, int line, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int sdcclDebugNoWarn;
extern char sdcclLastError[];

#define ENABLE_TRACE
#define WARN(...)                                                              \
  sdcclDebugLog(SDCCL_LOG_WARN, SDCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...)                                                       \
  sdcclDebugLog(SDCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...)                                                        \
  sdcclDebugLog(SDCCL_LOG_TRACE, SDCCL_CALL, __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...)                                                      \
  sdcclDebugLog(SDCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::steady_clock::time_point sdcclEpoch;
#else
#define TRACE(...)
#endif

void sdcclSetThreadName(pthread_t thread, const char *fmt, ...);

typedef void (*sdcclDebugLogger_t)(sdcclDebugLogLevel level,
                                    unsigned long flags, const char *file,
                                    int line, const char *fmt, ...);

// time recorder
#define TIMER_COLL_TOTAL 0
#define TIMER_COLL_CALC 1
#define TIMER_COLL_COMM 2
#define TIMER_COLL_MEM 3
#define TIMER_COLL_MEM_D2H 4
#define TIMER_COLL_MEM_H2D 5
#define TIMER_COLL_ALLOC 6
#define TIMER_COLL_FREE 7
#define TIMERS_COLL_COUNT 8

#endif
