/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "ipcsocket.h"
#include "utils.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET

#define SDCCL_IPC_SOCKNAME_STR "/tmp/sdccl-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
sdcclResult_t sdcclIpcSocketInit(sdcclIpcSocket *handle, int rank,
                                   uint64_t hash, int block) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[SDCCL_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return sdcclInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
    WARN("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    return sdcclSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, SDCCL_IPC_SOCKNAME_LEN, SDCCL_IPC_SOCKNAME_STR,
                     rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot bind provided name to socket. Name too large");
    return sdcclInternalError;
  }
#ifndef USE_ABSTRACT_SOCKET
  unlink(temp);
#endif

  INFO(SDCCL_NET, "UDS: Creating socket %s", temp);

  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    WARN("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno),
         errno);
    close(fd);
    return sdcclSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  if (!block) {
    int flags;
    EQCHECK(flags = fcntl(fd, F_GETFL), -1);
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIpcSocketGetFd(struct sdcclIpcSocket *handle, int *fd) {
  if (handle == NULL) {
    WARN("sdcclSocketGetFd: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (fd)
    *fd = handle->fd;
  return sdcclSuccess;
}

sdcclResult_t sdcclIpcSocketClose(sdcclIpcSocket *handle) {
  if (handle == NULL) {
    return sdcclInternalError;
  }
  if (handle->fd <= 0) {
    return sdcclSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socketName[0] != '\0') {
    unlink(handle->socketName);
  }
#endif
  close(handle->fd);

  return sdcclSuccess;
}

sdcclResult_t sdcclIpcSocketRecvMsg(sdcclIpcSocket *handle, void *hdr,
                                      int hdrLen, int *recvFd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Receiving data over socket failed : %d", errno);
      return sdcclSystemError;
    }
    if (handle->abortFlag &&
        __atomic_load_n(handle->abortFlag, __ATOMIC_RELAXED))
      return sdcclInternalError;
  }

  if (recvFd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) ||
          (cmptr->cmsg_type != SCM_RIGHTS)) {
        WARN("UDS: Receiving data over socket failed");
        return sdcclSystemError;
      }

      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      WARN("UDS: Receiving data over socket %s failed", handle->socketName);
      return sdcclSystemError;
    }
    TRACE(SDCCL_INIT | SDCCL_P2P, "UDS: Got recvFd %d from socket %s",
          *recvFd, handle->socketName);
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIpcSocketRecvFd(sdcclIpcSocket *handle, int *recvFd) {
  return sdcclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

sdcclResult_t sdcclIpcSocketSendMsg(sdcclIpcSocket *handle, void *hdr,
                                      int hdrLen, const int sendFd, int rank,
                                      uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[SDCCL_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, SDCCL_IPC_SOCKNAME_LEN, SDCCL_IPC_SOCKNAME_STR,
                     rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    WARN("UDS: Cannot connect to provided name for socket. Name too large");
    return sdcclInternalError;
  }
  (void)strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

  TRACE(SDCCL_INIT, "UDS: Sending hdr %p len %d to UDS socket %s", hdr, hdrLen,
        temp);

  if (sendFd != -1) {
    TRACE(SDCCL_INIT, "UDS: Sending fd %d to UDS socket %s", sendFd, temp);

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      WARN("UDS: Sending data over socket %s failed : %s (%d)", temp,
           strerror(errno), errno);
      return sdcclSystemError;
    }
    if (handle->abortFlag &&
        __atomic_load_n(handle->abortFlag, __ATOMIC_RELAXED))
      return sdcclInternalError;
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIpcSocketSendFd(sdcclIpcSocket *handle, const int sendFd,
                                     int rank, uint64_t hash) {
  return sdcclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
