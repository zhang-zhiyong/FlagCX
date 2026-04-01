/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_PARAM_H_
#define SDCCL_PARAM_H_

#include "sdccl.h"
#include <stdint.h>

const char *userHomeDir();
void setEnvFile(const char *fileName);
void initEnv();
const char *sdcclGetEnv(const char *name);

void sdcclLoadParam(char const *env, int64_t deftVal, int64_t uninitialized,
                     int64_t *cache);

#define SDCCL_PARAM(name, env, deftVal)                                       \
  int64_t sdcclParam##name() {                                                \
    constexpr int64_t uninitialized = INT64_MIN;                               \
    static_assert(deftVal != uninitialized,                                    \
                  "default value cannot be the uninitialized value.");         \
    static int64_t cache = uninitialized;                                      \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) ==          \
                             uninitialized,                                    \
                         false)) {                                             \
      sdcclLoadParam("SDCCL_" env, deftVal, uninitialized, &cache);          \
    }                                                                          \
    return cache;                                                              \
  }

#endif
