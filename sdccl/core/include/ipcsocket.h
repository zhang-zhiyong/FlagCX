/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_IPCSOCKET_H
#define SDCCL_IPCSOCKET_H

#include "core.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define SDCCL_IPC_SOCKNAME_LEN 64

struct sdcclIpcSocket {
  int fd;
  char socketName[SDCCL_IPC_SOCKNAME_LEN];
  volatile uint32_t *abortFlag;
};

sdcclResult_t sdcclIpcSocketInit(struct sdcclIpcSocket *handle, int rank,
                                   uint64_t hash, int block);
sdcclResult_t sdcclIpcSocketClose(struct sdcclIpcSocket *handle);
sdcclResult_t sdcclIpcSocketGetFd(struct sdcclIpcSocket *handle, int *fd);

sdcclResult_t sdcclIpcSocketRecvFd(struct sdcclIpcSocket *handle, int *fd);
sdcclResult_t sdcclIpcSocketSendFd(struct sdcclIpcSocket *handle,
                                     const int fd, int rank, uint64_t hash);

sdcclResult_t sdcclIpcSocketSendMsg(sdcclIpcSocket *handle, void *hdr,
                                      int hdrLen, const int sendFd, int rank,
                                      uint64_t hash);
sdcclResult_t sdcclIpcSocketRecvMsg(sdcclIpcSocket *handle, void *hdr,
                                      int hdrLen, int *recvFd);

#endif /* SDCCL_IPCSOCKET_H */