/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sdccl.h"
#include <cstddef>

#ifndef SDCCL_TYPE_H_
#define SDCCL_TYPE_H_

#define SDCCL_VERSION(X, Y, Z) ((X)*10000 + (Y)*100 + (Z))

/* sdcclScalarResidence_t: Location and dereferencing logic for scalar
 * arguments. */
typedef enum {
  /* sdcclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  sdcclScalarDevice = 0,

  /* sdcclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the sdcclRedOpCreate***() function returns. */
  sdcclScalarHostImmediate = 1
} sdcclScalarResidence_t;

#define SDCCL_CONFIG_UNDEF_INT INT_MIN
#define SDCCL_CONFIG_UNDEF_PTR NULL
#define SDCCL_SPLIT_NOCOLOR -1

typedef struct sdcclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
} sdcclConfig_t;

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (SDCCL_STEPS / 4)
#define ALLREDUCE_CHUNKSTEPS (SDCCL_STEPS / 2)
#define ALLGATHER_SLICESTEPS (SDCCL_STEPS / 4)
#define ALLGATHER_CHUNKSTEPS (SDCCL_STEPS / 2)
#define REDUCESCATTER_SLICESTEPS (SDCCL_STEPS / 4)
#define REDUCESCATTER_CHUNKSTEPS (SDCCL_STEPS / 2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define SDCCL_MAX_SLICE_PER_CHUNK                                             \
  2 // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

#include <sys/types.h>

#define SDCCL_MODE_NORMAL 0
#define SDCCL_MODE_OFFSET 1
#define SDCCL_MODE_PTR 2
struct sdcclConnFifo {
  int mode;
  int offset;
  ssize_t size;
  void *ptr;
};

#endif