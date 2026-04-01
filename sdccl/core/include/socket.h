/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_SOCKET_H_
#define SDCCL_SOCKET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "type.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES                                                    \
  2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES                                                   \
  3 // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define SDCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union sdcclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum sdcclSocketState {
  sdcclSocketStateNone = 0,
  sdcclSocketStateInitialized = 1,
  sdcclSocketStateAccepting = 2,
  sdcclSocketStateAccepted = 3,
  sdcclSocketStateConnecting = 4,
  sdcclSocketStateConnectPolling = 5,
  sdcclSocketStateConnected = 6,
  sdcclSocketStateReady = 7,
  sdcclSocketStateClosed = 8,
  sdcclSocketStateError = 9,
  sdcclSocketStateNum = 10
};

enum sdcclSocketType {
  sdcclSocketTypeUnknown = 0,
  sdcclSocketTypeBootstrap = 1,
  sdcclSocketTypeProxy = 2,
  sdcclSocketTypeNetSocket = 3,
  sdcclSocketTypeNetIb = 4
};

struct sdcclSocket {
  int fd;
  int acceptFd;
  int timedOutRetries;
  int refusedRetries;
  union sdcclSocketAddress addr;
  volatile uint32_t *abortFlag;
  int asyncFlag;
  enum sdcclSocketState state;
  int salen;
  uint64_t magic;
  enum sdcclSocketType type;
};

const char *sdcclSocketToString(union sdcclSocketAddress *addr, char *buf,
                                 const int numericHostForm = 1);
sdcclResult_t sdcclSocketGetAddrFromString(union sdcclSocketAddress *ua,
                                             const char *ip_port_pair);
int sdcclFindInterfaceMatchSubnet(char *ifNames,
                                   union sdcclSocketAddress *localAddrs,
                                   union sdcclSocketAddress *remoteAddr,
                                   int ifNameMaxSize, int maxIfs);
int sdcclFindInterfaces(char *ifNames, union sdcclSocketAddress *ifAddrs,
                         int ifNameMaxSize, int maxIfs);

// Initialize a socket
sdcclResult_t
sdcclSocketInit(struct sdcclSocket *sock,
                 union sdcclSocketAddress *addr = NULL,
                 uint64_t magic = SDCCL_SOCKET_MAGIC,
                 enum sdcclSocketType type = sdcclSocketTypeUnknown,
                 volatile uint32_t *abortFlag = NULL, int asyncFlag = 0);
// Create a listening socket. sock->addr can be pre-filled with IP & port info.
// sock->fd is set after a successful call
sdcclResult_t sdcclSocketListen(struct sdcclSocket *sock);
sdcclResult_t sdcclSocketGetAddr(struct sdcclSocket *sock,
                                   union sdcclSocketAddress *addr);
// Connect to sock->addr. sock->fd is set after a successful call.
sdcclResult_t sdcclSocketConnect(struct sdcclSocket *sock);
// Return socket connection state.
sdcclResult_t sdcclSocketReady(struct sdcclSocket *sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file
// descriptor in sock->fd, with the remote side IP/port in sock->addr.
sdcclResult_t sdcclSocketAccept(struct sdcclSocket *sock,
                                  struct sdcclSocket *ulistenSock);
sdcclResult_t sdcclSocketGetFd(struct sdcclSocket *sock, int *fd);
sdcclResult_t sdcclSocketSetFd(int fd, struct sdcclSocket *sock);

#define SDCCL_SOCKET_SEND 0
#define SDCCL_SOCKET_RECV 1

sdcclResult_t sdcclSocketProgress(int op, struct sdcclSocket *sock,
                                    void *ptr, int size, int *offset);
sdcclResult_t sdcclSocketWait(int op, struct sdcclSocket *sock, void *ptr,
                                int size, int *offset);
sdcclResult_t sdcclSocketSend(struct sdcclSocket *sock, void *ptr, int size);
sdcclResult_t sdcclSocketRecv(struct sdcclSocket *sock, void *ptr, int size);
sdcclResult_t sdcclSocketSendRecv(struct sdcclSocket *sendSock,
                                    void *sendPtr, int sendSize,
                                    struct sdcclSocket *recvSock,
                                    void *recvPtr, int recvSize);
sdcclResult_t sdcclSocketTryRecv(struct sdcclSocket *sock, void *ptr,
                                   int size, int *closed, bool blocking);
sdcclResult_t sdcclSocketClose(struct sdcclSocket *sock);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
