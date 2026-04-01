/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "comm.h"
#include "core.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>

static int sdcclNetIfs = -1;
struct sdcclNetSocketDev {
  union sdcclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char *pciPath;
};
static struct sdcclNetSocketDev sdcclNetSocketDevs[MAX_IFS];

pthread_mutex_t sdcclNetSocketLock = PTHREAD_MUTEX_INITIALIZER;

static sdcclResult_t sdcclNetSocketGetPciPath(char *devName, char **pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketInit() {
  if (sdcclNetIfs == -1) {
    pthread_mutex_lock(&sdcclNetSocketLock);
    if (sdcclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE * MAX_IFS];
      union sdcclSocketAddress addrs[MAX_IFS];
      sdcclNetIfs =
          sdcclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (sdcclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return sdcclInternalError;
      } else {
#define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN + 1];
        char addrline[SOCKET_NAME_MAXLEN + 1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i = 0; i < sdcclNetIfs; i++) {
          strcpy(sdcclNetSocketDevs[i].devName, names + i * MAX_IF_NAME_SIZE);
          memcpy(&sdcclNetSocketDevs[i].addr, addrs + i,
                 sizeof(union sdcclSocketAddress));
          SDCCLCHECK(sdcclNetSocketGetPciPath(
              sdcclNetSocketDevs[i].devName, &sdcclNetSocketDevs[i].pciPath));
          snprintf(line + strlen(line), MAX_LINE_LEN - strlen(line),
                   " [%d]%s:%s", i, names + i * MAX_IF_NAME_SIZE,
                   sdcclSocketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(SDCCL_INIT | SDCCL_NET, "NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&sdcclNetSocketLock);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketDevices(int *ndev) {
  *ndev = sdcclNetIfs;
  return sdcclSuccess;
}

static sdcclResult_t sdcclNetSocketGetSpeed(char *devName, int *speed) {
  *speed = 0;
  char speedPath[PATH_MAX];
  sprintf(speedPath, "/sys/class/net/%s/speed", devName);
  int fd = open(speedPath, O_RDONLY);
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr) - 1) > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
    close(fd);
  }
  if (*speed <= 0) {
    INFO(SDCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.",
         speedPath);
    *speed = 10000;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketGetProperties(int dev, void *props) {
  sdcclNetProperties_t *netProps = (sdcclNetProperties_t *)props;
  netProps->name = sdcclNetSocketDevs[dev].devName;
  netProps->pciPath = sdcclNetSocketDevs[dev].pciPath;
  netProps->guid = dev;
  netProps->ptrSupport = SDCCL_PTR_HOST;
  netProps->regIsGlobal = 0;
  SDCCLCHECK(sdcclNetSocketGetSpeed(netProps->name, &netProps->speed));
  netProps->latency = 0; // Not set
  netProps->port = 0;
  netProps->maxComms = 65536;
  netProps->maxRecvs = 1;
  netProps->netDeviceType = SDCCL_NET_DEVICE_HOST;
  netProps->netDeviceVersion = SDCCL_NET_DEVICE_INVALID_VERSION;
  return sdcclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS SDCCL_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64 * 1024)

SDCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
SDCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

enum sdcclNetSocketCommState {
  sdcclNetSocketCommStateStart = 0,
  sdcclNetSocketCommStateConnect = 1,
  sdcclNetSocketCommStateAccept = 3,
  sdcclNetSocketCommStateSend = 4,
  sdcclNetSocketCommStateRecv = 5,
};

struct sdcclNetSocketCommStage {
  enum sdcclNetSocketCommState state;
  uint8_t iteration;
  struct sdcclSocket *sock;
  struct sdcclNetSocketComm *comm;
};

struct sdcclNetSocketHandle {
  union sdcclSocketAddress connectAddr;
  uint64_t magic; // random number to help debugging
  int nSocks;
  int nThreads;
  struct sdcclNetSocketCommStage stage;
};

struct sdcclNetSocketTask {
  int op;
  void *data;
  int size;
  struct sdcclSocket *sock;
  int offset;
  int used;
  sdcclResult_t result;
};

struct sdcclNetSocketRequest {
  int op;
  void *data;
  int size;
  struct sdcclSocket *ctrlSock;
  int offset;
  int used;
  struct sdcclNetSocketComm *comm;
  struct sdcclNetSocketTask *tasks[MAX_SOCKETS];
  int nSubs;
};

struct sdcclNetSocketTaskQueue {
  int next;
  int len;
  struct sdcclNetSocketTask *tasks;
};

struct sdcclNetSocketThreadResources {
  struct sdcclNetSocketTaskQueue threadTaskQueue;
  int stop;
  struct sdcclNetSocketComm *comm;
  pthread_mutex_t threadLock;
  pthread_cond_t threadCond;
};

struct sdcclNetSocketListenComm {
  struct sdcclSocket sock;
  struct sdcclNetSocketCommStage stage;
  int nSocks;
  int nThreads;
  int dev;
};

struct sdcclNetSocketComm {
  struct sdcclSocket ctrlSock;
  struct sdcclSocket socks[MAX_SOCKETS];
  int dev;
  int cudaDev;
  int nSocks;
  int nThreads;
  int nextSock;
  struct sdcclNetSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct sdcclNetSocketThreadResources threadResources[MAX_THREADS];
};

void *persistentSocketThread(void *args_) {
  struct sdcclNetSocketThreadResources *resource =
      (struct sdcclNetSocketThreadResources *)args_;
  struct sdcclNetSocketComm *comm = resource->comm;
  struct sdcclNetSocketTaskQueue *myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i = 0; i < myQueue->len; i += nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j = 0; j < nSocksPerThread; j++) {
          struct sdcclNetSocketTask *r = myQueue->tasks + i + j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = sdcclSocketProgress(r->op, r->sock, r->data, r->size,
                                             &r->offset);
            if (r->result != sdcclSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size)
              repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next &&
             resource->stop == 0) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (resource->stop)
      return NULL;
  }
}

sdcclResult_t sdcclNetSocketGetNsockNthread(int dev, int *ns, int *nt) {
  int nSocksPerThread = sdcclParamSocketNsocksPerThread();
  int nThreads = sdcclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : SDCCL_SOCKET_NTHREADS is greater than the maximum "
         "allowed, setting to %d",
         MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt = 0, autoNs = 1; // By default, we only use the main thread and
                                // do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor",
             sdcclNetSocketDevs[dev].devName);
    char *rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(SDCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      autoNt = 4;
      autoNs = 1;
    }
  end:
    if (nThreads == -2)
      nThreads = autoNt;
    if (nSocksPerThread == -2)
      nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS / nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum "
         "allowed, setting SDCCL_NSOCKS_PERTHREAD to %d",
         nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0)
    INFO(SDCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread",
         nThreads, nSocksPerThread);
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketListen(int dev, void *opaqueHandle,
                                     void **listenComm) {
  if (dev < 0 ||
      dev >= sdcclNetIfs) { // data transfer socket is based on specified dev
    return sdcclInternalError;
  }
  struct sdcclNetSocketHandle *handle =
      (struct sdcclNetSocketHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct sdcclNetSocketHandle));
  static_assert(sizeof(struct sdcclNetSocketHandle) <=
                    SDCCL_NET_HANDLE_MAXSIZE,
                "sdcclNetSocketHandle size too large");
  struct sdcclNetSocketListenComm *comm;
  SDCCLCHECK(sdcclCalloc(&comm, 1));
  handle->magic = SDCCL_SOCKET_MAGIC;
  SDCCLCHECK(sdcclSocketInit(&comm->sock, &sdcclNetSocketDevs[dev].addr,
                               handle->magic, sdcclSocketTypeNetSocket, NULL,
                               1));
  SDCCLCHECK(sdcclSocketListen(&comm->sock));
  SDCCLCHECK(sdcclSocketGetAddr(&comm->sock, &handle->connectAddr));
  SDCCLCHECK(
      sdcclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketConnect(int dev, void *opaqueHandle,
                                      void **sendComm) {
  if (dev < 0 ||
      dev >= sdcclNetIfs) { // data transfer socket is based on specified dev
    return sdcclInternalError;
  }

  int ready;
  struct sdcclNetSocketHandle *handle =
      (struct sdcclNetSocketHandle *)opaqueHandle;
  struct sdcclNetSocketCommStage *stage = &handle->stage;
  struct sdcclNetSocketComm *comm = stage->comm;
  uint8_t i = stage->iteration;
  struct sdcclSocket *sock = stage->sock;
  *sendComm = NULL;

  if (stage->state == sdcclNetSocketCommStateConnect)
    goto socket_connect_check;
  if (stage->state == sdcclNetSocketCommStateSend)
    goto socket_send;

  SDCCLCHECK(sdcclCalloc(&comm, 1));
  stage->comm = comm;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  comm->dev = dev;
  for (; i < comm->nSocks + 1; i++) {
    sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks + i;
    SDCCLCHECK(sdcclSocketInit(sock, &handle->connectAddr, handle->magic,
                                 sdcclSocketTypeNetSocket, NULL, 1));

    stage->sock = sock;
    stage->state = sdcclNetSocketCommStateConnect;
    stage->iteration = i;
    SDCCLCHECK(sdcclSocketConnect(sock));

  socket_connect_check:
    SDCCLCHECK(sdcclSocketReady(sock, &ready));
    if (!ready)
      return sdcclSuccess;
    stage->state = sdcclNetSocketCommStateSend;

  socket_send:
    int done = 0;
    SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_SEND, sock, &i,
                                     sizeof(uint8_t), &done));
    if (done == 0)
      return sdcclSuccess;
  }
  *sendComm = comm;
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketAccept(void *listenComm, void **recvComm) {
  struct sdcclNetSocketListenComm *lComm =
      (struct sdcclNetSocketListenComm *)listenComm;
  struct sdcclNetSocketCommStage *stage = &lComm->stage;
  struct sdcclNetSocketComm *rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct sdcclSocket *sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == sdcclNetSocketCommStateAccept)
    goto socket_accept_check;
  if (stage->state == sdcclNetSocketCommStateRecv)
    goto socket_recv;

  SDCCLCHECK(sdcclCalloc(&rComm, 1));
  stage->comm = rComm;
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  rComm->dev = lComm->dev;
  for (; i < rComm->nSocks + 1; i++) {
    uint8_t sendSockIdx;

    SDCCLCHECK(sdcclCalloc(&sock, 1));
    SDCCLCHECK(sdcclSocketInit(sock));
    stage->sock = sock;
    stage->state = sdcclNetSocketCommStateAccept;
    stage->iteration = i;
    SDCCLCHECK(sdcclSocketAccept(sock, &lComm->sock));

  socket_accept_check:
    SDCCLCHECK(sdcclSocketReady(sock, &ready));
    if (!ready)
      return sdcclSuccess;

    stage->state = sdcclNetSocketCommStateRecv;
  socket_recv:
    int done = 0;
    SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, sock, &sendSockIdx,
                                     sizeof(uint8_t), &done));
    if (done == 0)
      return sdcclSuccess;

    if (sendSockIdx == rComm->nSocks)
      memcpy(&rComm->ctrlSock, sock, sizeof(struct sdcclSocket));
    else
      memcpy(rComm->socks + sendSockIdx, sock, sizeof(struct sdcclSocket));
    free(sock);
  }
  *recvComm = rComm;

  /* reset lComm state */
  stage->state = sdcclNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketGetRequest(struct sdcclNetSocketComm *comm,
                                         int op, void *data, size_t size,
                                         struct sdcclNetSocketRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct sdcclNetSocketRequest *r = comm->requests + i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlSock = &comm->ctrlSock;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      *req = r;
      return sdcclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return sdcclInternalError;
}

sdcclResult_t sdcclNetSocketGetTask(struct sdcclNetSocketComm *comm, int op,
                                      void *data, int size,
                                      struct sdcclNetSocketTask **req) {
  int tid = comm->nextSock % comm->nThreads;
  struct sdcclNetSocketThreadResources *res = comm->threadResources + tid;
  struct sdcclNetSocketTaskQueue *queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    SDCCLCHECK(sdcclCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    pthread_create(comm->helperThread + tid, NULL, persistentSocketThread, res);
    sdcclSetThreadName(comm->helperThread[tid], "SDCCL Sock%c%1u%2u%2u",
                        op == SDCCL_SOCKET_SEND ? 'S' : 'R', comm->dev, tid,
                        comm->cudaDev);
  }
  struct sdcclNetSocketTask *r = queue->tasks + queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;
    r->offset = 0;
    r->result = sdcclSuccess;
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next + 1) % queue->len;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return sdcclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return sdcclInternalError;
}

sdcclResult_t sdcclNetSocketTest(void *request, int *done, int *size) {
  *done = 0;
  struct sdcclNetSocketRequest *r = (struct sdcclNetSocketRequest *)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return sdcclInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    SDCCLCHECK(
        sdcclSocketProgress(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    if (offset == 0)
      return sdcclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int))
      SDCCLCHECK(
          sdcclSocketWait(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == SDCCL_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union sdcclSocketAddress addr;
      sdcclSocketGetAddr(r->ctrlSock, &addr);
      WARN(
          "NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \
          there may be a mismatch in collective sizes or environment settings (e.g. SDCCL_PROTO, SDCCL_ALGO) between ranks",
          sdcclSocketToString(&addr, line), data, r->size);
      return sdcclInvalidUsage;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size - chunkOffset);
        SDCCLCHECK(sdcclNetSocketGetTask(r->comm, r->op,
                                           (char *)(r->data) + chunkOffset,
                                           chunkSize, r->tasks + i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i = 0; i < r->nSubs; i++) {
        struct sdcclNetSocketTask *sub = r->tasks[i];
        if (sub->result != sdcclSuccess)
          return sub->result;
        if (sub->offset == sub->size)
          nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size)
          *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i = 0; i < r->nSubs; i++) {
          struct sdcclNetSocketTask *sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        SDCCLCHECK(sdcclSocketProgress(r->op, r->ctrlSock, r->data, r->size,
                                         &r->offset));
      }
      if (r->offset == r->size) {
        if (size)
          *size = r->size;
        *done = 1;
        r->used = 0;
      }
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketRegMr(void *comm, void *data, size_t size,
                                    int type, int mrFlags, void **mhandle) {
  (void)mrFlags;
  return (type != SDCCL_PTR_HOST) ? sdcclInternalError : sdcclSuccess;
}

sdcclResult_t sdcclNetSocketDeregMr(void *comm, void *mhandle) {
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketIsend(void *sendComm, void *data, size_t size,
                                    int tag, void *mhandle, void *phandle,
                                    void **request) {
  struct sdcclNetSocketComm *comm = (struct sdcclNetSocketComm *)sendComm;
  SDCCLCHECK(
      sdcclNetSocketGetRequest(comm, SDCCL_SOCKET_SEND, data, size,
                                (struct sdcclNetSocketRequest **)request));
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketIrecv(void *recvComm, int n, void **data,
                                    size_t *sizes, int *tags, void **mhandles,
                                    void **phandles, void **request) {
  struct sdcclNetSocketComm *comm = (struct sdcclNetSocketComm *)recvComm;
  if (n != 1)
    return sdcclInternalError;
  SDCCLCHECK(
      sdcclNetSocketGetRequest(comm, SDCCL_SOCKET_RECV, data[0], sizes[0],
                                (struct sdcclNetSocketRequest **)request));
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketIflush(void *recvComm, int n, void **data,
                                     int *sizes, void **mhandles,
                                     void **request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return sdcclInternalError;
}

sdcclResult_t sdcclNetSocketCloseListen(void *opaqueComm) {
  struct sdcclNetSocketListenComm *comm =
      (struct sdcclNetSocketListenComm *)opaqueComm;
  if (comm) {
    int ready;
    SDCCLCHECK(sdcclSocketReady(&comm->sock, &ready));
    if (ready)
      SDCCLCHECK(sdcclSocketClose(&comm->sock));
    free(comm);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclNetSocketClose(void *opaqueComm) {
  struct sdcclNetSocketComm *comm = (struct sdcclNetSocketComm *)opaqueComm;
  if (comm) {
    for (int i = 0; i < comm->nThreads; i++) {
      struct sdcclNetSocketThreadResources *res = comm->threadResources + i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->stop = 1;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      free(res->threadTaskQueue.tasks);
    }
    int ready;
    SDCCLCHECK(sdcclSocketReady(&comm->ctrlSock, &ready));
    if (ready)
      SDCCLCHECK(sdcclSocketClose(&comm->ctrlSock));
    for (int i = 0; i < comm->nSocks; i++) {
      SDCCLCHECK(sdcclSocketReady(&comm->socks[i], &ready));
      if (ready)
        SDCCLCHECK(sdcclSocketClose(&comm->socks[i]));
    }
    free(comm);
  }
  return sdcclSuccess;
}

sdcclNetAdaptor sdcclNetSocket = {
    // Basic functions
    "Socket", sdcclNetSocketInit, sdcclNetSocketDevices,
    sdcclNetSocketGetProperties,

    // Setup functions
    sdcclNetSocketListen, sdcclNetSocketConnect, sdcclNetSocketAccept,
    sdcclNetSocketClose, // closeSend
    sdcclNetSocketClose, // closeRecv (same as closeSend for socket)
    sdcclNetSocketCloseListen,

    // Memory region functions
    sdcclNetSocketRegMr,
    NULL, // regMrDmaBuf - No DMA-BUF support
    sdcclNetSocketDeregMr,

    // Two-sided functions
    sdcclNetSocketIsend, sdcclNetSocketIrecv, sdcclNetSocketIflush,
    sdcclNetSocketTest,

    // One-sided functions
    NULL, // iput - not supported on socket
    NULL, // iget - not supported on socket
    NULL, // iputSignal - not supported on socket

    // Device name lookup
    NULL, // getDevFromName
};
