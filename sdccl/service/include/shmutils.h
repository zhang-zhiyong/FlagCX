/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_SHMUTILS_H_
#define SDCCL_SHMUTILS_H_

#include "sdccl.h"

#define SHM_PATH_MAX 128

typedef void *sdcclShmHandle_t;
sdcclResult_t sdcclShmOpen(char *shmPath, size_t shmPathSize, size_t shmSize,
                             void **shmPtr, void **devShmPtr, int refcount,
                             sdcclShmHandle_t *handle);
sdcclResult_t sdcclShmClose(sdcclShmHandle_t handle);
sdcclResult_t sdcclShmUnlink(sdcclShmHandle_t handle);

struct shmIpcDesc {
  char shmSuffix[7];
  sdcclShmHandle_t handle;
  size_t shmSize;
};
typedef struct shmIpcDesc sdcclShmIpcDesc_t;

sdcclResult_t sdcclShmAllocateShareableBuffer(size_t size,
                                                sdcclShmIpcDesc_t *descOut,
                                                void **hptr, void **dptr);
sdcclResult_t sdcclShmImportShareableBuffer(sdcclShmIpcDesc_t *desc,
                                              void **hptr, void **dptr,
                                              sdcclShmIpcDesc_t *descOut);
sdcclResult_t sdcclShmIpcClose(sdcclShmIpcDesc_t *desc);

#endif
