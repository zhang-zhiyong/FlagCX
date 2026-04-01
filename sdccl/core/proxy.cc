/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "proxy.h"
#include "adaptor.h"
#include "comm.h"
#include "device_api/sdccl_device.h" // sdcclDevCommInternal, devComm
#include "sdccl_hetero.h"
#include "sdccl_kernel.h" // SDCCL_DEVICE_CTA_COUNT
#include "info.h"
#include "net.h"
#include "onesided.h"
#include "p2p.h"
#include "socket.h"
#include "transport.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <assert.h>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

enum { proxyRecv = 0, proxySend = 1 };
extern union sdcclSocketAddress bootstrapNetIfAddr;

static bool proxyMatchOpType(int type) {
  switch (type) {
    case sdcclProxyMsgInit:
    case sdcclProxyMsgSharedInit:
    case sdcclProxyMsgSetup:
    case sdcclProxyMsgConnect:
    case sdcclProxyMsgGetFd:
    case sdcclProxyMsgRegister:
    case sdcclProxyMsgDeregister:
    case sdcclProxyMsgRegMr:
    case sdcclProxyMsgDeregMr:
    case sdcclProxyMsgSendRecv:
      return true;
    default:
      return false;
  }
}

SDCCL_TEMPLETELIST_DEFINE(ProdProgChannel, struct sdcclProxyOps,
                           prodPrevChannel, prodNextChannel);
SDCCL_TEMPLETELIST_DEFINE(ConsProgChannel, struct sdcclProxyOps,
                           consPrevChannel, consNextChannel);
SDCCL_TEMPLETELIST_DEFINE(ProgPeer, struct sdcclProxyOps::consPeer, prevPeer,
                           nextPeer);

sdcclResult_t
sdcclProxyProgressChannelJoin(struct sdcclProxyState *proxyState,
                               struct sdcclProxyState *) {

  return sdcclSuccess;
}

static sdcclResult_t asyncProxyOpEnqueue(sdcclProxyAsyncOp **opHead,
                                          sdcclProxyAsyncOp *newOp) {
  sdcclProxyAsyncOp *list = *opHead;
  if (list == NULL)
    *opHead = newOp;
  else {
    while (list->next)
      list = list->next;
    list->next = newOp;
    newOp->prev = list;
  }
  return sdcclSuccess;
}

static sdcclResult_t asyncProxyOpDequeue(sdcclProxyAsyncOp **opHead,
                                          sdcclProxyAsyncOp *op) {
  if (*opHead == op)
    *opHead = op->next;
  if (op->next)
    op->next->prev = op->prev;
  if (op->prev)
    op->prev->next = op->next;
  if (op->reqSize)
    free(op->reqBuff);
  if (op->respSize)
    free(op->respBuff);
  free(op);
  return sdcclSuccess;
}

static sdcclResult_t SaveProxy(struct sdcclHeteroComm *comm,
                                struct sdcclChannel *channel, int type,
                                int peer, struct sdcclProxyOp *op,
                                int connIndex, bool *justInquire) {
  if (peer < 0)
    return sdcclSuccess;

  if (justInquire)
    *justInquire = true;
  else {
    struct sdcclProxyOps *proxyOps;
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next> *queue;

    proxyOps = &comm->proxyState->proxyOps[op->channelId];
    queue = type == proxySend ? &proxyOps->prodPeers.sendQueue
                              : &proxyOps->prodPeers.recvQueue;

    pthread_mutex_lock(&comm->proxyState->mutex);
    sdcclProdProgChannelListEnList(&comm->proxyState->prodProgChannelHead,
                                    proxyOps);
    sdcclIntruQueueEnqueue(queue, op);
    pthread_cond_signal(&comm->proxyState->cond);
    pthread_mutex_unlock(&comm->proxyState->mutex);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclProxySaveOp(struct sdcclHeteroComm *comm,
                                 struct sdcclProxyOp *op, bool *justInquire) {
  struct sdcclChannel *channel = &comm->channels[op->channelId];
  if (justInquire)
    *justInquire = false;
  switch (op->pattern) {
    case sdcclPatternSend:
      // Self-copy will be saved as a send operation
      if (op->root == comm->rank)
        op->selfCopy = 1;
      SDCCLCHECK(
          SaveProxy(comm, channel, proxySend, op->root, op, 0, justInquire));
      break;
    case sdcclPatternRecv:
      if (op->root == comm->rank)
        return sdcclSuccess;
      SDCCLCHECK(
          SaveProxy(comm, channel, proxyRecv, op->root, op, 0, justInquire));
      break;
  }
  return sdcclSuccess;
}

// Only for double check purpose, we can check if the progress queue is empty
// It is safe to not call this function in the progress thread.
static void sdcclProgressQueEmptyCheck(struct sdcclProxyState *proxyState) {
  bool error = 0;
  if (!sdcclProdProgChannelListEmpty(proxyState->prodProgChannelHead) ||
      !sdcclConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    error = 1;
  }
  for (int i = 0; i < MAXCHANNELS; i++) {
    if (!sdcclProgPeerListEmpty(proxyState->proxyOps[i].consProgPeerHead))
      error = 1;
    for (int r = 0; r < proxyState->nRanks; r++) {
      if (!sdcclIntruQueueEmpty(
              &proxyState->proxyOps[i].consPeers[r].sendQueue) ||
          !sdcclIntruQueueEmpty(
              &proxyState->proxyOps[i].consPeers[r].recvQueue))
        error = 1;
    }
    if (!sdcclIntruQueueEmpty(&proxyState->proxyOps[i].prodPeers.sendQueue) ||
        !sdcclIntruQueueEmpty(&proxyState->proxyOps[i].prodPeers.recvQueue))
      error = 1;
  }
  if (error)
    INFO(SDCCL_INIT, "progress queue is not empty");
}

// process all the ProxyOps in the consumer queue
// idle is set to 1 if no operations are pending
// if idle is set to 0, it means there are pending operations
// For simplicity, if these are any pending operations in queue, we set idle to
// 0
static sdcclResult_t progressOps(struct sdcclProxyState *proxyState,
                                  int *idle) {
  *idle = 1;
  if (!sdcclConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    struct sdcclProxyOps *proxyOps = proxyState->consProgChannelHead;
    do {
      struct sdcclProxyOps *next = proxyOps->consNextChannel;

      if (!sdcclProgPeerListEmpty(proxyOps->consProgPeerHead)) {
        struct sdcclProxyOps::consPeer *peer = proxyOps->consProgPeerHead;
        do {
          struct sdcclProxyOps::consPeer *next = peer->nextPeer;
          struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next>
              *queue;
          queue = &peer->sendQueue;
          if (!sdcclIntruQueueEmpty(queue)) {
            *idle &= 0;
            struct sdcclProxyOp *op = sdcclIntruQueueHead(queue);
            if (op->connection->transport == TRANSPORT_NET) {
              struct sendNetResources *resources =
                  (sendNetResources *)op->connection->transportResources;
              sdcclProxySend(resources, op->recvbuff, op->nbytes, &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                op->args.semaphore.reset();
                sdcclIntruQueueDelete(queue, op);
                free(op);
              }
            } else if (op->connection->transport == TRANSPORT_P2P) {
              struct sdcclP2pResources *resources =
                  (sdcclP2pResources *)op->connection->transportResources;
              if (op->selfCopy == 0) {
                sdcclP2pProxySend(resources, op->recvbuff, op->nbytes,
                                   &op->args);
              } else {
                sdcclP2pProxySelfCopy(resources, op->sendbuff, op->recvbuff,
                                       op->nbytes, &op->args);
              }
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                op->args.semaphore.reset();
                sdcclIntruQueueDelete(queue, op);
                free(op);
              }
            }
          }
          queue = &peer->recvQueue;
          if (!sdcclIntruQueueEmpty(queue)) {
            *idle &= 0;
            struct sdcclProxyOp *op = sdcclIntruQueueHead(queue);
            if (op->connection->transport == TRANSPORT_NET) {
              struct recvNetResources *resources =
                  (recvNetResources *)op->connection->transportResources;
              sdcclProxyRecv(resources, op->recvbuff, op->nbytes, &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                // update refcount and delete semaphore when refcount = 0
                op->args.semaphore.reset();
                sdcclIntruQueueDelete(queue, op);
                free(op);
              }
            } else if (op->connection->transport == TRANSPORT_P2P) {
              struct sdcclP2pResources *resources =
                  (sdcclP2pResources *)op->connection->transportResources;
              sdcclP2pProxyRecv(resources, op->recvbuff, op->nbytes,
                                 &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                // update refcount and delete semaphore when refcount = 0
                op->args.semaphore.reset();
                sdcclIntruQueueDelete(queue, op);
                free(op);
              }
            }
          }
          if (sdcclIntruQueueEmpty(&peer->sendQueue) &&
              sdcclIntruQueueEmpty(&peer->recvQueue)) {
            sdcclProgPeerListDelete(&proxyOps->consProgPeerHead, peer);
          }
          peer = next;
        } while (peer != NULL);
      }
      if (sdcclProgPeerListEmpty(proxyOps->consProgPeerHead)) {
        sdcclConsProgChannelListDelete(&proxyState->consProgChannelHead,
                                        proxyOps);
      }
      proxyOps = next;
    } while (proxyOps != NULL);
  }
  return sdcclSuccess;
}

// get proxy operations from the producer queue
// and move them to the consumer queue
// added means the number of operations fetched from producer queue and added to
// the consumer queue.
static sdcclResult_t
sdcclProxyGetPostedOps(struct sdcclProxyState *proxyState, int *added) {
  struct sdcclProxyProgressState *state = &proxyState->progressState;
  // No need to block waiting for the lock to be available. Exit, continue
  // progress, and come back later.
  if (pthread_mutex_trylock(&proxyState->mutex) != 0) {
    *added = 0;
    return sdcclSuccess;
  }

  // If we have ops to progress, no need to block waiting for something to
  // arrive
  if (sdcclConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    while (sdcclProdProgChannelListEmpty(proxyState->prodProgChannelHead) &&
           state->stop == 0) {
      pthread_cond_wait(&proxyState->cond, &proxyState->mutex);
    }
    if (state->stop != 0) {
      pthread_mutex_unlock(&proxyState->mutex);
      *added = 0;
      return sdcclSuccess;
    }
  }

  // Put anything available right now in the producer queue into the consumer
  // queue.
  while (!sdcclProdProgChannelListEmpty(proxyState->prodProgChannelHead)) {
    struct sdcclProxyOps *proxyOps =
        sdcclProdProgChannelListDeList(&proxyState->prodProgChannelHead);

    sdcclConsProgChannelListEnList(&proxyState->consProgChannelHead, proxyOps);
    struct sdcclIntruQueue<struct sdcclProxyOp, &sdcclProxyOp::next> *queue;
    queue = &proxyOps->prodPeers.sendQueue;
    while (!sdcclIntruQueueEmpty(queue)) {
      struct sdcclProxyOp *op = sdcclIntruQueueDequeue(queue);
      sdcclProgPeerListEnList(&proxyOps->consProgPeerHead,
                               &proxyOps->consPeers[op->root]);
      sdcclIntruQueueEnqueue(&proxyOps->consPeers[op->root].sendQueue, op);
      (*added)++;
    }
    queue = &proxyOps->prodPeers.recvQueue;
    while (!sdcclIntruQueueEmpty(queue)) {
      struct sdcclProxyOp *op = sdcclIntruQueueDequeue(queue);
      sdcclProgPeerListEnList(&proxyOps->consProgPeerHead,
                               &proxyOps->consPeers[op->root]);
      sdcclIntruQueueEnqueue(&proxyOps->consPeers[op->root].recvQueue, op);
      (*added)++;
    }
  }
  pthread_mutex_unlock(&proxyState->mutex);
  return sdcclSuccess;
}

SDCCL_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);

inline void *sdcclProxyProgress(void *proxyState_) {
  struct sdcclProxyState *proxyState = (sdcclProxyState *)proxyState_;
  // flag indicating if there is any in-operating operation
  int idle = 1;
  /* Too frequent call of ncclProxyGetPostedOps() will result in perf regression
   * for small message communication. proxyOpAppendCounter is a counter that
   * helps us decide if we need to append proxy ops. After each progress,
   * proxyOpAppendCounter will increase by 1 and compare with environment
   * variable ncclParamProgressAppendOpFreq(). If they are equal, we will append
   * proxy ops. This will decrease the frequency of calling
   * ncclProxyGetPostedOps() and reduce the perf impact. */
  int proxyOpAppendCounter = 0;
  deviceAdaptor->setDevice(proxyState->cudaDev);
  struct sdcclProxyProgressState *state = &proxyState->progressState;

  while (state->stop == 0 || idle == 0) {
    idle = 1;
    // consume the operations in the consumer queue
    progressOps(proxyState, &idle);

    if (idle || (++proxyOpAppendCounter == sdcclParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      if (state->stop == 0) {
        // move all the operations from the producer queue to the consumer queue
        sdcclProxyGetPostedOps(proxyState, &added);
      }
      if (added == 0) {
        sched_yield(); // No request progressed. Let others run.
      }
    }
  }

  sdcclProgressQueEmptyCheck(proxyState);
  return NULL;
}

static sdcclResult_t expectedProxyResponseStore(struct sdcclProxyState *state,
                                                 void *opId, void *respBuff,
                                                 int respSize,
                                                 sdcclResult_t res) {
  struct sdcclExpectedProxyResponse *elem = state->expectedResponses;
  while (elem) {
    if (elem->opId == opId) {
      if (respSize != elem->respSize) {
        WARN("Mismatched response size for opId=%p", opId);
        return sdcclInternalError;
      }

      if (elem->done) {
        WARN("Storing response for already completed opId=%p", opId);
        return sdcclInternalError;
      }

      if (respSize > 0 && respBuff != NULL) {
        memcpy(elem->respBuff, respBuff, respSize);
        free(respBuff);
      }
      elem->done = true;
      elem->res = res;
      return sdcclSuccess;
    }
    elem = elem->next;
  }

  WARN("Proxy response for opId=%p doesn't match any expected response", opId);
  return sdcclInternalError;
}

static sdcclResult_t
expectedProxyResponseEnqueue(struct sdcclProxyState *state, void *opId,
                             int respSize) {
  struct sdcclExpectedProxyResponse *ex;
  SDCCLCHECK(sdcclCalloc(&ex, 1));
  ex->opId = opId;

  // Pre-alloc response buffer
  ex->respBuff = malloc(respSize);
  ex->respSize = respSize;
  ex->res = sdcclInternalError;
  ex->done = false;

  // Enqueue
  struct sdcclExpectedProxyResponse *list = state->expectedResponses;
  if (list == NULL) {
    state->expectedResponses = ex;
    return sdcclSuccess;
  }
  while (list->next)
    list = list->next;
  list->next = ex;
  return sdcclSuccess;
}

static sdcclResult_t
expectedProxyResponseDequeue(struct sdcclProxyState *state, void *opId,
                             void *respBuff, int *found) {
  struct sdcclExpectedProxyResponse *elem = state->expectedResponses;
  struct sdcclExpectedProxyResponse *prev = NULL;
  *found = 0;
  while (elem) {
    if ((elem->opId == opId) && elem->done) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(respBuff, elem->respBuff, elem->respSize);
      sdcclResult_t res = elem->res;
      free(elem->respBuff);
      free(elem);
      *found = 1;
      return res;
    }
    prev = elem;
    elem = elem->next;
  }
  return sdcclSuccess;
}

static sdcclResult_t
expectedProxyResponseRemove(struct sdcclProxyState *state, void *opId) {
  struct sdcclExpectedProxyResponse *elem = state->expectedResponses;
  struct sdcclExpectedProxyResponse *prev = NULL;
  while (elem) {
    if (elem->opId == opId) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      free(elem->respBuff);
      free(elem);
      return sdcclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  WARN("Couldn't find opId=%p", opId);
  return sdcclInternalError;
}

sdcclResult_t sdcclPollProxyResponse(struct sdcclHeteroComm *comm,
                                       struct sdcclProxyConnector *proxyConn,
                                       void *respBuff, void *opId) {
  struct sdcclProxyState *sharedProxyState = comm->proxyState;
  // Check response queue
  int found = 0;
  sdcclResult_t res =
      expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);

  if (found == 0) {
    // Attempt to read in a new response header from the proxy thread
    struct sdcclSocket *sock = &sharedProxyState->peerSock;
    sdcclProxyRpcResponseHeader resp = {0};
    int offset = 0;
    if (sdcclSuccess != sdcclSocketProgress(SDCCL_SOCKET_RECV, sock, &resp,
                                              sizeof(resp), &offset)) {
      WARN("Socket recv failed while polling for opId=%p", opId);
      return sdcclInternalError;
    }

    if (offset == 0) {
      return sdcclInProgress;
      // If we've returned a partial response, block to receive the rest of it
    } else if (offset < sizeof(resp)) {
      while (offset < sizeof(resp))
        SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, sock, &resp,
                                         sizeof(resp), &offset));
    }

    INFO(SDCCL_PROXY, "sdcclPollProxyResponse Received new opId=%p",
         resp.opId);

    // If there's a respSize to recv
    if (resp.respSize > 0) {
      if (resp.opId != opId) {
        // Unexpected response, need to buffer the socket data
        respBuff = malloc(resp.respSize);
      }
      assert(respBuff != NULL);
      SDCCLCHECK(sdcclSocketRecv(sock, respBuff, resp.respSize));
    }

    if (resp.opId == opId) {
      INFO(SDCCL_PROXY, "resp.opId=%p matches expected opId=%p", resp.opId,
           opId);
      SDCCLCHECK(expectedProxyResponseRemove(sharedProxyState, resp.opId));
      return resp.res;
    } else {
      INFO(SDCCL_PROXY, "Queuing opId=%p respBuff=%p respSize=%d", resp.opId,
           respBuff, resp.respSize);
      // Store the result and mark response as completed
      SDCCLCHECK(expectedProxyResponseStore(
          sharedProxyState, resp.opId, respBuff, resp.respSize, resp.res));
      return sdcclInProgress;
    }
  } else {
    INFO(SDCCL_PROXY, "sdcclPollProxyResponse Dequeued cached opId=%p", opId);
  }
  return res;
}

static sdcclResult_t proxyProgressAsync(sdcclProxyAsyncOp **opHead,
                                         sdcclProxyAsyncOp *op,
                                         int *asyncOpCount) {
  int done = 0;
  const char *dmaBufEnable = sdcclGetEnv("SDCCL_DMABUF_ENABLE");
  bool dmaEnabled = false; // disabled by default
  if (dmaBufEnable != NULL) {
    if (strcmp(dmaBufEnable, "1") == 0) {
      dmaEnabled = true;
    }
  }
  bool dmaBufferSupport = false;
  if (deviceAdaptor->dmaSupport != NULL) {
    deviceAdaptor->dmaSupport(&dmaBufferSupport);
  }
  dmaBufferSupport = dmaEnabled && dmaBufferSupport;
  if (op->type == sdcclProxyMsgConnect) {
    TRACE(SDCCL_PROXY,
          "proxyProgressAsync::sdcclProxyMsgConnect opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d, transport=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize,
          op->connection->transport);

    if (op->connection->transport == TRANSPORT_P2P) {
      // P2P transport
      if (op->connection->send) {
        INFO(SDCCL_PROXY, "Calling sdcclP2pSendProxyConnect");
        sdcclP2pSendProxyConnect(op->connection, NULL, op->reqBuff,
                                  op->reqSize, op->respBuff, op->respSize,
                                  &done);
        INFO(SDCCL_PROXY, "sdcclP2pSendProxyConnect completed, done=%d",
             done);
      } else {
        INFO(SDCCL_PROXY, "Calling sdcclP2pRecvProxyConnect");
        sdcclP2pRecvProxyConnect(op->connection, NULL, op->reqBuff,
                                  op->reqSize, op->respBuff, op->respSize,
                                  &done);
        INFO(SDCCL_PROXY, "sdcclP2pRecvProxyConnect completed, done=%d",
             done);
      }
    } else if (op->connection->transport == TRANSPORT_NET) {
      // NET transport (original logic)
      if (op->connection->send) {
        struct sendNetResources *resources =
            (struct sendNetResources *)op->connection->transportResources;
        if (!resources->netSendComm) {
          SDCCLCHECK(resources->netAdaptor->connect(
              resources->netDev, (void *)op->reqBuff, &resources->netSendComm));
        } else {
          if (dmaBufferSupport &&
              resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            INFO(SDCCL_PROXY,
                 "Registering memory region with DMA-BUF support");
            int dmabuf_fd;
            SDCCLCHECK(deviceAdaptor->getHandleForAddressRange(
                (void *)&dmabuf_fd, resources->buffers[0],
                resources->buffSizes[0], 0));
            SDCCLCHECK(resources->netAdaptor->regMrDmaBuf(
                resources->netSendComm, resources->buffers[0],
                resources->buffSizes[0], 2, 0ULL, dmabuf_fd, 0,
                &resources->mhandles[0]));
            (void)close(dmabuf_fd);
          } else {
            if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0], 2, 0, &resources->mhandles[0]));
            } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0], 1, 0, &resources->mhandles[0]));
            } else {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0],
                  (resources->ptrSupport & SDCCL_PTR_CUDA) ? SDCCL_PTR_CUDA
                                                            : SDCCL_PTR_HOST,
                  0, &resources->mhandles[0]));
            }
          }
          done = 1;
        }
      } else {
        struct recvNetResources *resources =
            (struct recvNetResources *)op->connection->transportResources;
        if (!resources->netRecvComm) {
          SDCCLCHECK(resources->netAdaptor->accept(resources->netListenComm,
                                                    &resources->netRecvComm));
        } else {
          if (dmaBufferSupport) {
            INFO(SDCCL_PROXY,
                 "Registering memory region with DMA-BUF support");
            int dmabuf_fd;
            SDCCLCHECK(deviceAdaptor->getHandleForAddressRange(
                (void *)&dmabuf_fd, resources->buffers[0],
                resources->buffSizes[0], 0));
            SDCCLCHECK(resources->netAdaptor->regMrDmaBuf(
                resources->netRecvComm, resources->buffers[0],
                resources->buffSizes[0], 2, 0ULL, dmabuf_fd, 0,
                &resources->mhandles[0]));
            (void)close(dmabuf_fd);
          } else {
            if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0], 2, 0, &resources->mhandles[0]));
            } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0], 1, 0, &resources->mhandles[0]));
            } else {
              SDCCLCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0],
                  (resources->ptrSupport & SDCCL_PTR_CUDA) ? SDCCL_PTR_CUDA
                                                            : SDCCL_PTR_HOST,
                  0, &resources->mhandles[0]));
            }
          }
          done = 1;
        }
      }
    }
  } else if (op->type == sdcclProxyMsgRegister) {
    TRACE(SDCCL_PROXY,
          "proxyProgressAsync::sdcclProxyMsgRegister opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize);
    if (op->connection->transport == TRANSPORT_P2P) {
      ;
    } else if (op->connection->transport == TRANSPORT_NET) {
      void *handle;
      struct netRegInfo *info = (struct netRegInfo *)op->reqBuff;
      assert(op->reqSize == sizeof(struct netRegInfo));
      assert(op->respSize == sizeof(void *));
      if (op->connection->send) {
        // send side
        struct sendNetResources *resources =
            (struct sendNetResources *)(op->connection->transportResources);
        if (dmaBufferSupport) {
          int dmabuf_fd;
          SDCCLCHECK(deviceAdaptor->getHandleForAddressRange(
              (void *)&dmabuf_fd, (void *)info->buffer, info->size, 0));
          SDCCLCHECK(resources->netAdaptor->regMrDmaBuf(
              resources->netSendComm, (void *)info->buffer, info->size, 2, 0ULL,
              dmabuf_fd, 0, &handle));
          (void)close(dmabuf_fd);
        } else {
          SDCCLCHECK(resources->netAdaptor->regMr(resources->netSendComm,
                                                   (void *)info->buffer,
                                                   info->size, 2, 0, &handle));
        }
      } else {
        // recv side
        struct recvNetResources *resources =
            (struct recvNetResources *)(op->connection->transportResources);
        if (dmaBufferSupport) {
          int dmabuf_fd;
          SDCCLCHECK(deviceAdaptor->getHandleForAddressRange(
              (void *)&dmabuf_fd, (void *)info->buffer, info->size, 0));
          SDCCLCHECK(resources->netAdaptor->regMrDmaBuf(
              resources->netRecvComm, (void *)info->buffer, info->size, 2, 0ULL,
              dmabuf_fd, 0, &handle));
          (void)close(dmabuf_fd);
        } else {
          SDCCLCHECK(resources->netAdaptor->regMr(resources->netRecvComm,
                                                   (void *)info->buffer,
                                                   info->size, 2, 0, &handle));
        }
      }
      memcpy(op->respBuff, (void *)&handle, sizeof(void *));
      done = 1;
    }
  } else if (op->type == sdcclProxyMsgDeregister) {
    TRACE(SDCCL_PROXY,
          "proxyProgressAsync::sdcclProxyMsgDeregister opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize);
    if (op->connection->transport == TRANSPORT_P2P) {
      ;
    } else if (op->connection->transport == TRANSPORT_NET) {
      void *handle;
      assert(op->reqSize == sizeof(void *));
      memcpy(&handle, op->reqBuff, sizeof(void *));
      if (op->connection->send) {
        // send side
        struct sendNetResources *resources =
            (struct sendNetResources *)(op->connection->transportResources);
        SDCCLCHECK(
            resources->netAdaptor->deregMr(resources->netSendComm, handle));
      } else {
        // recv side
        struct recvNetResources *resources =
            (struct recvNetResources *)(op->connection->transportResources);
        SDCCLCHECK(
            resources->netAdaptor->deregMr(resources->netRecvComm, handle));
      }
      done = 1;
    }
  } else if (op->type == sdcclProxyMsgSetup &&
             op->connection->transport == TRANSPORT_P2P) {
    if (op->connection->send) {
      // P2P Send side setup
      INFO(SDCCL_PROXY, "Calling sdcclP2pSendProxySetup");
      sdcclP2pSendProxySetup(op->connection, NULL, op->reqBuff, op->reqSize,
                              op->respBuff, op->respSize, &done);
      INFO(SDCCL_PROXY, "sdcclP2pSendProxySetup completed, done=%d", done);
    } else {
      // P2P Recv side setup
      INFO(SDCCL_PROXY, "Calling sdcclP2pRecvProxySetup");
      sdcclP2pRecvProxySetup(op->connection, NULL, op->reqBuff, op->reqSize,
                              op->respBuff, op->respSize, &done);
      INFO(SDCCL_PROXY, "sdcclP2pRecvProxySetup completed, done=%d", done);
    }
  } else {
    return sdcclInternalError;
  }
  if (done) {
    INFO(SDCCL_PROXY,
         "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d "
         "done",
         op->opId, op->type, op->reqBuff, op->respSize);
    if (op->connection->transport == TRANSPORT_NET) {
      if (op->type == sdcclProxyMsgConnect)
        __atomic_store_n(&op->connection->state, connConnected,
                         __ATOMIC_RELEASE);
    }

    /* if setup or connect is done, we should not return any error at this point
     * since sdcclSocketSend might already send the respBuff to the requester.
     * If we still choose to abort and close the connection, it can cause
     * segfault if the requester is using the respBuff. */

    sdcclProxyRpcResponseHeader resp = {op->opId, sdcclSuccess, op->respSize};

    // Send the opId for referencing async operation
    SDCCLCHECK(sdcclSocketSend(op->connection->sock, &resp, sizeof(resp)));
    if (op->respSize) {
      // Send the response
      SDCCLCHECK(
          sdcclSocketSend(op->connection->sock, op->respBuff, op->respSize));
    }

    asyncProxyOpDequeue(opHead, op);
    (*asyncOpCount)--;
    return sdcclSuccess;
  }

  return sdcclInProgress;
}

sdcclResult_t sdcclProxyCallAsync(struct sdcclHeteroComm *comm,
                                    struct sdcclProxyConnector *proxyConn,
                                    int type, void *reqBuff, int reqSize,
                                    int respSize, void *opId) {
  struct sdcclSocket *sock;
  sdcclResult_t ret = sdcclSuccess;
  struct sdcclProxyState *sharedProxyState = comm->proxyState;

  sock = &sharedProxyState->peerSock;
  if (sock == NULL)
    return sdcclInternalError;

  SDCCLCHECKGOTO(sdcclSocketSend(sock, &type, sizeof(int)), ret, error);
  SDCCLCHECKGOTO(
      sdcclSocketSend(sock, &proxyConn->connection, sizeof(void *)), ret,
      error);
  SDCCLCHECKGOTO(sdcclSocketSend(sock, &reqSize, sizeof(int)), ret, error);
  SDCCLCHECKGOTO(sdcclSocketSend(sock, &respSize, sizeof(int)), ret, error);
  if (reqSize)
    SDCCLCHECKGOTO(sdcclSocketSend(sock, reqBuff, reqSize), ret, error);

  // Send opId to proxy
  SDCCLCHECKGOTO(sdcclSocketSend(sock, &opId, sizeof(opId)), ret, error);

  SDCCLCHECK(expectedProxyResponseEnqueue(sharedProxyState, opId, respSize));
  return sdcclSuccess;
error:
  return ret;
}

static sdcclResult_t proxyServiceInitOp(int type, struct sdcclSocket *sock,
                                         struct sdcclProxyAsyncOp **opHead,
                                         sdcclHeteroComm_t comm,
                                         int *asyncOpCount) {
  struct sdcclProxyAsyncOp *asyncOp;
  SDCCLCHECK(sdcclCalloc(&asyncOp, 1));

  asyncOp->type = type;
  SDCCLCHECK(sdcclSocketRecv(sock, &asyncOp->connection, sizeof(void *)));

  SDCCLCHECK(sdcclSocketRecv(sock, &asyncOp->reqSize, sizeof(int)));
  SDCCLCHECK(sdcclSocketRecv(sock, &asyncOp->respSize, sizeof(int)));
  if (asyncOp->reqSize) {
    SDCCLCHECK(sdcclCalloc(&asyncOp->reqBuff, asyncOp->reqSize));
    SDCCLCHECK(sdcclSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize));
  }

  // Store opId for completion response
  SDCCLCHECK(sdcclSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)));

  asyncOp->connection->sock = sock;
  if (asyncOp->respSize)
    SDCCLCHECK(sdcclCalloc(&asyncOp->respBuff, asyncOp->respSize));

  SDCCLCHECK(asyncProxyOpEnqueue(opHead, asyncOp));
  (*asyncOpCount)++;
  SDCCLCHECK(proxyProgressAsync(opHead, asyncOp, asyncOpCount));
  return sdcclSuccess;
}

sdcclResult_t sdcclProxyCallBlocking(struct sdcclHeteroComm *comm,
                                       struct sdcclProxyConnector *proxyConn,
                                       int type, void *reqBuff, int reqSize,
                                       void *respBuff, int respSize) {
  // Alloc some memory to act as a handle
  sdcclResult_t res = sdcclSuccess;
  void *opId = malloc(1);

  SDCCLCHECKGOTO(sdcclProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize,
                                       respSize, opId),
                  res, fail);

  do {
    res = sdcclPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (res == sdcclInProgress);

exit:
  free(opId);
  return res;
fail:
  goto exit;
}

sdcclResult_t sdcclProxyInit(struct sdcclHeteroComm *comm) {
  INFO(SDCCL_INIT, "rank=%d sdcclProxyInit called.", comm->rank);
  SDCCLCHECK(sdcclSocketInit(&comm->proxyState->listenSock,
                               &bootstrapNetIfAddr, comm->magic,
                               sdcclSocketTypeProxy, NULL, 1));
  SDCCLCHECK(sdcclSocketListen(&comm->proxyState->listenSock));

  sdcclSocket *proxySock = &comm->proxyState->peerSock;
  SDCCLCHECK(sdcclSocketInit(proxySock, &comm->proxyState->listenSock.addr,
                               comm->magic, sdcclSocketTypeProxy));
  SDCCLCHECK(sdcclSocketConnect(proxySock));

  char proxyMsg[10];
  memcpy(proxyMsg, (string("Proxy: ") + to_string(comm->rank)).c_str(), 10);
  sdcclSocketSend(proxySock, proxyMsg, 10);
  comm->proxyState->cudaDev = comm->cudaDev;
  pthread_create(&comm->proxyState->thread, NULL, sdcclProxyService,
                 (void *)comm);
  pthread_create(&comm->proxyState->progressState.thread, NULL,
                 sdcclProxyProgress, comm->proxyState);
#ifdef COMPILE_KERNEL_HOST
  // Initialize synchronization primitives before creating thread
  pthread_mutex_init(&comm->proxyState->kernelState.initMutex, NULL);
  pthread_cond_init(&comm->proxyState->kernelState.initCond, NULL);
  comm->proxyState->kernelState.ready = 0;

  pthread_create(&comm->proxyState->kernelState.thread, NULL,
                 sdcclProxyKernelService, (void *)comm);

  // Wait for kernel proxy thread to finish initialization
  pthread_mutex_lock(&comm->proxyState->kernelState.initMutex);
  while (comm->proxyState->kernelState.ready == 0) {
    pthread_cond_wait(&comm->proxyState->kernelState.initCond,
                      &comm->proxyState->kernelState.initMutex);
  }
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);
#endif

  comm->proxyState->initialized = 1;
  return sdcclSuccess;
}

void *sdcclProxyService(void *args) {
  int stop = 0;
  int closeConn = 0;
  int asyncOpCount = 0;
  struct sdcclHeteroComm *comm = (struct sdcclHeteroComm *)args;
  struct sdcclProxyAsyncOp *opHead = NULL;
  struct sdcclProxyAsyncOp *list = NULL;
  struct sdcclSocket sock;
  sdcclResult_t res = sdcclSuccess;

  // Set device context
  SDCCLCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // One peer only
  SDCCLCHECKGOTO(sdcclSocketInit(&sock), res, out);
  SDCCLCHECKGOTO(sdcclSocketAccept(&sock, &comm->proxyState->listenSock), res,
                  out);
  char proxyMsg[10];
  sdcclSocketRecv(&sock, proxyMsg, 10);
  INFO(SDCCL_PROXY,
       "[Service thread] Receive proxy message : \033[31m%s\033[0m", proxyMsg);
  struct pollfd pollfds[1];
  pollfds[0].fd = sock.fd;
  pollfds[0].events = POLLIN;

  while (!stop || (stop && opHead)) {
    int ret;
    do {
      ret = poll(pollfds, 1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      closeConn = 1;
      break;
    }
    if (closeConn) {
      break;
    }

    // Progress all ops
    list = opHead;
    while (list) {
      struct sdcclProxyAsyncOp *opNext = list->next;
      res = proxyProgressAsync(&opHead, list, &asyncOpCount);
      if (res == sdcclSuccess || res == sdcclInProgress) {
        list = opNext;
      } else {
        WARN("[Service thread] Error encountered progressing operation with "
             "res=%d, closing connection",
             res);
        closeConn = 1;
        break;
      }
    }
    if (closeConn) {
      break;
    }

    // Check for additional ops coming in
    int type;
    if (pollfds[0].revents & POLLIN) {
      int closed = 0;
      res = sdcclSocketTryRecv(&sock, &type, sizeof(int), &closed,
                                false /*blocking*/);
      if (res != sdcclSuccess && res != sdcclInProgress) {
        WARN("[Service thread] Could not receive type from rank %d, "
             "res=%u, "
             "closed=%d",
             comm->rank, res, closed);
        closeConn = 1;
      } else if (closed) {
        INFO(SDCCL_PROXY, "[Service thread] Connection closed by rank %d",
             comm->rank);
        closeConn = 1;
      } else if (res == sdcclSuccess) {
        if (type == sdcclProxyMsgStop) {
          stop = 1;
          closeConn = 1;
        } else if (proxyMatchOpType(type)) {
          res = proxyServiceInitOp(type, &sock, &opHead, comm, &asyncOpCount);
          if (res != sdcclSuccess) {
            WARN("[Service thread] Error encountered initializing operation "
                 "with res=%d, closing connection",
                 res);
            closeConn = 1;
          }
        } else {
          INFO(SDCCL_PROXY, "[Service thread] Unknown command %d from rank %d",
               type, comm->rank);
          closeConn = 1;
        }
      }
    }
    if (closeConn) {
      break;
    }
  }
out:
  // Stop progress thread before freeing any resource
  pthread_mutex_lock(&comm->proxyState->mutex);
  comm->proxyState->progressState.stop = 1;
  pthread_cond_signal(&comm->proxyState->cond);
  pthread_mutex_unlock(&comm->proxyState->mutex);
  pthread_join(comm->proxyState->progressState.thread, nullptr);
#ifdef COMPILE_KERNEL_HOST
  // Stop kernel thread and cleanup its mutex/cond
  pthread_join(comm->proxyState->kernelState.thread, nullptr);
  pthread_mutex_destroy(&comm->proxyState->kernelState.initMutex);
  pthread_cond_destroy(&comm->proxyState->kernelState.initCond);
#endif

  // Free P2P resources in proxy thread (CUDA resources must be freed in the
  // same thread where they were created)
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers[peer]->recv[0].connected == 1) {
        struct sdcclConnector *conn = comm->channels[c].peers[peer]->recv;
        if (conn->proxyConn.connection->transport == TRANSPORT_P2P) {
          struct sdcclP2pResources *resources =
              (struct sdcclP2pResources *)
                  conn->proxyConn.connection->transportResources;
          sdcclP2pRecvProxyFree(resources);
        }
      }
      if (comm->channels[c].peers[peer]->send[0].connected == 1) {
        struct sdcclConnector *conn = comm->channels[c].peers[peer]->send;
        if (conn->proxyConn.connection->transport == TRANSPORT_P2P) {
          struct sdcclP2pResources *resources =
              (struct sdcclP2pResources *)
                  conn->proxyConn.connection->transportResources;
          sdcclP2pSendProxyFree(resources);
        }
      }
    }
  }

  // Close sockets
  sdcclSocketClose(&sock);
  sdcclSocketClose(&comm->proxyState->listenSock);

  // Dequeue unhandled ops
  list = opHead;
  while (list) {
    struct sdcclProxyAsyncOp *opNext = list->next;
    asyncProxyOpDequeue(&opHead, list);
    list = opNext;
  }

  INFO(SDCCL_PROXY,
       "[Service thread] Wait for progress thread joined and free resources");
  return NULL;
}

void *sdcclProxyKernelService(void *args) {
  int groupCount = 0;
  int termCount = 0;
  sdcclDeviceTrigger_t ptr = NULL;
  sdcclFifo_t fifo = NULL;
  struct sdcclHeteroComm *comm = (struct sdcclHeteroComm *)args;
  sdcclResult_t res = sdcclSuccess;

  auto validateOneSidedPeer = [](struct sdcclHeteroComm *comm,
                                 int peerRank) -> sdcclResult_t {
    if (globalOneSideHandleCount == 0 || globalOneSideHandleTable[0] == NULL)
      return sdcclNotSupported;
    if (peerRank < 0 || peerRank >= comm->nRanks)
      return sdcclInvalidArgument;

    // Check full-mesh connection exists for this peer (including self-loopback)
    struct sdcclOneSideHandleInfo *handles =
        (struct sdcclOneSideHandleInfo *)globalOneSideHandleTable[0];
    if (handles->fullSendComms == NULL ||
        handles->fullSendComms[peerRank] == NULL)
      return sdcclNotSupported;

    return sdcclSuccess;
  };

  // Set device context
  SDCCLCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // Create FIFO
  comm->proxyState->kernelState.fifo = new sdcclFifo();
  SDCCLCHECKGOTO(comm->proxyState->kernelState.fifo->sdcclFifoInit(), res,
                  out);
  fifo = comm->proxyState->kernelState.fifo;
  // comm->fifoBuffer = (void *)comm->proxyState->kernelState.fifo->buffer;
  SDCCLCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                      &comm->fifoBuffer,
                      (void *)comm->proxyState->kernelState.fifo->buffer),
                  res, out);

  // Create a dedicated stream
  sdcclStream_t stream;
  SDCCLCHECKGOTO(deviceAdaptor->streamCreate(&stream), res, out);
  INFO(SDCCL_P2P, "rank %d p2p stream %lu", comm->rank, (uintptr_t)stream);

  // Allocate trigger structure
  SDCCLCHECKGOTO(sdcclCalloc(&ptr, sizeof(sdcclDeviceTrigger)), res, out);

  // Signal that initialization is complete
  pthread_mutex_lock(&comm->proxyState->kernelState.initMutex);
  comm->proxyState->kernelState.ready = 1;
  pthread_cond_signal(&comm->proxyState->kernelState.initCond);
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);

  while (true) {
    if (comm->proxyState->kernelState.stop == 1)
      break;
    dequeue(fifo->buffer, ptr);
    if ((ptr->getPrim() == sdcclDevicePrimSend ||
         ptr->getPrim() == sdcclDevicePrimRecv) &&
        ptr->getAddr() == 0) {
      sched_yield();
      continue;
    }
    switch (ptr->getPrim()) {
      case sdcclDevicePrimSend:
        if (groupCount == 0) {
          res = sdcclHeteroGroupStart();
          TRACE(SDCCL_P2P,
                "rank=%d sdcclHeteroGroupStart called by proxyKernelService.",
                comm->rank);
          groupCount++;
        }
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimSend called by proxyKernelService.",
              comm->rank);
        res = sdcclHeteroSend((const void *)(uintptr_t)(ptr->getAddr()),
                               ptr->getCount(),
                               (sdcclDataType_t)(ptr->getDatatype()),
                               ptr->getPeerRank(), comm, stream);
        break;
      case sdcclDevicePrimRecv:
        if (groupCount == 0) {
          res = sdcclHeteroGroupStart();
          TRACE(SDCCL_P2P,
                "rank=%d sdcclHeteroGroupStart called by proxyKernelService.",
                comm->rank);
          groupCount++;
        }
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimRecv called by proxyKernelService.",
              comm->rank);
        res = sdcclHeteroRecv((void *)(uintptr_t)(ptr->getAddr()),
                               ptr->getCount(),
                               (sdcclDataType_t)(ptr->getDatatype()),
                               ptr->getPeerRank(), comm, stream);
        break;
      case sdcclDevicePrimTerm: {
        termCount++;
        int totalCoops = (int)ptr->getTotalCoops();
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimTerm called by proxyKernelService "
              "groupCount=%d termCount=%d/%d.",
              comm->rank, groupCount, termCount, totalCoops);
        if (groupCount > 0 && termCount >= totalCoops) {
          res = sdcclHeteroGroupEnd();
          TRACE(SDCCL_P2P,
                "rank=%d sdcclHeteroGroupEnd called by proxyKernelService.",
                comm->rank);
          groupCount--;
          termCount = 0;
        }
        break;
      }
      case sdcclDevicePrimPut: {
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimPut called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != sdcclSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        res = sdcclHeteroPut(comm, peerRank, srcOffset, dstOffset, size,
                              srcMrIdx, dstMrIdx);
        break;
      }
      case sdcclDevicePrimSignal: {
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimSignal called by proxyKernelService.",
              comm->rank);
        uint64_t bufType = ptr->getBufferType();
        int signalIdx = (int)ptr->getSignalIdx();
        uint64_t signalValue = ptr->getSignalValue();
        size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);

        if (bufType == 0) {
          // Signal buffer: RDMA FETCH_AND_ADD to peer's signalBuffer
          int peerRank = (int)ptr->getPeerRank();
          res = validateOneSidedPeer(comm, peerRank);
          if (res != sdcclSuccess)
            break;
          if (globalOneSideSignalHandles == NULL) {
            WARN("sdcclDevicePrimSignal: globalOneSideSignalHandles not "
                 "initialized — call sdcclOneSideSignalRegister() before use");
            res = sdcclInternalError;
            break;
          }
          res = sdcclHeteroPutSignal(comm, peerRank, 0, 0, 0, signalOff, 0, 0,
                                      signalValue);
        } else {
          // Counter buffer: local CPU atomic increment (no network operation)
          sdcclDevComm_t dc = comm->devCommHandle;
          if (dc == NULL || dc->counterBuffer == NULL) {
            WARN("sdcclDevicePrimSignal: counterBuffer not initialized");
            res = sdcclInternalError;
            break;
          }
          uint64_t *counterPtr = (uint64_t *)dc->counterBuffer + signalIdx;
          __atomic_fetch_add(counterPtr, signalValue, __ATOMIC_RELAXED);
        }
        break;
      }
      case sdcclDevicePrimWaitSignal: {
        TRACE(
            SDCCL_P2P,
            "rank=%d sdcclDevicePrimWaitSignal called by proxyKernelService.",
            comm->rank);
        uint64_t wsBufType = ptr->getBufferType(); // 0=signal, 1=counter
        int wsSignalIdx = (int)ptr->getSignalIdx();
        uint32_t wsExpected = (uint32_t)ptr->getExpectedValue();
        size_t wsSignalOff = (size_t)wsSignalIdx * sizeof(uint64_t);
        sdcclDevComm_t dc = comm->devCommHandle;
        if (dc == NULL) {
          WARN("sdcclDevicePrimWaitSignal: devComm not initialized");
          res = sdcclInternalError;
          break;
        }
        // Select target buffer based on buffer type
        uint64_t *targetBuffer =
            (wsBufType == 0) ? dc->signalBuffer : dc->counterBuffer;
        if (targetBuffer == NULL) {
          WARN("sdcclDevicePrimWaitSignal: %s buffer not allocated",
               wsBufType == 0 ? "signal" : "counter");
          res = sdcclInternalError;
          break;
        }
        void *waitAddr = (void *)((char *)targetBuffer + wsSignalOff);
        res = deviceAdaptor->streamWaitValue64(stream, waitAddr,
                                               (uint64_t)wsExpected, 0);
        break;
      }
      case sdcclDevicePrimPutSignal: {
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimPutSignal called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != sdcclSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        int signalIdx = (int)ptr->getSignalIdx();
        uint64_t signalValue = ptr->getSignalValue();
        size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);
        if (globalOneSideSignalHandles == NULL) {
          WARN("sdcclDevicePrimPutSignal: globalOneSideSignalHandles not "
               "initialized — call sdcclOneSideSignalRegister() before use");
          res = sdcclInternalError;
          break;
        }
        res = sdcclHeteroPutSignal(comm, peerRank, srcOffset, dstOffset, size,
                                    signalOff, srcMrIdx, dstMrIdx, signalValue);
        break;
      }
      case sdcclDevicePrimPutValue: {
        int peerRank = (int)ptr->getPeerRank();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        uint64_t value = ptr->getValue();
        res = sdcclHeteroPutValue(comm, peerRank, value, dstOffset, dstMrIdx);
        break;
      }
      case sdcclDevicePrimGet: {
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimGet called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != sdcclSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        res = sdcclHeteroGet(comm, peerRank, srcOffset, dstOffset, size,
                              srcMrIdx, dstMrIdx);
        break;
      }
      case sdcclDevicePrimWait:
        TRACE(SDCCL_P2P,
              "rank=%d sdcclDevicePrimWait called by proxyKernelService.",
              comm->rank);
        deviceAdaptor->streamSynchronize(stream);
        break;
      case sdcclDevicePrimBarrierSignal: {
        // Inter-node barrier: RDMA ATOMIC FETCH_AND_ADD to each peer's
        // interSignalFlagsHost counter via iputSignal (signal-only, size=0).
        sdcclDevComm_t dc = comm->devCommHandle;
        if (dc && dc->nInterPeers > 0 && dc->barrierHandleInfo) {
          uint32_t ctaIdx = (uint32_t)ptr->getAddr();
          struct sdcclNetAdaptor *net =
              (struct sdcclNetAdaptor *)dc->netAdaptorPtr;
          size_t signalOff = (size_t)ctaIdx * sizeof(uint64_t);

          void *reqs[SDCCL_MAX_INTER_PEERS];
          for (int p = 0; p < dc->nInterPeers; p++) {
            reqs[p] = nullptr;
            net->iputSignal(dc->signalSendComms[p], 0, 0, 0, comm->rank,
                            dc->interPeerRanks[p], NULL, NULL,
                            (uint64_t)signalOff, (void **)dc->barrierHandleInfo,
                            1, &reqs[p]);
          }
          for (int p = 0; p < dc->nInterPeers; p++) {
            if (reqs[p]) {
              int done = 0;
              while (!done) {
                net->test(reqs[p], &done, nullptr);
              }
            }
          }
        }
        break;
      }
      default:
        break;
    }
    // Mark item as consumed AFTER processing
    __sync_synchronize();
    ((volatile uint64_t *)fifo->buffer)[sdcclFifoIdxConsumed]++;
    if (res != sdcclSuccess)
      break;
  }
  // destroy stream
  res = deviceAdaptor->streamSynchronize(stream);
  res = deviceAdaptor->streamDestroy(stream);
  // deallocate trigger structure
  free(ptr);

out:
  // destroy fifo
  res = comm->proxyState->kernelState.fifo->sdcclFifoDestroy();
  delete comm->proxyState->kernelState.fifo;
  comm->fifoBuffer = NULL;
  return NULL;
}

sdcclResult_t sdcclProxyFree(struct sdcclHeteroComm *comm) {
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers[peer]->recv[0].connected == 1) {
        struct sdcclConnector *conn = comm->channels[c].peers[peer]->recv;
        int transport = conn->proxyConn.connection->transport;

        if (transport == TRANSPORT_NET) {
          struct recvNetResources *resources =
              (struct recvNetResources *)
                  conn->proxyConn.connection->transportResources;
          sdcclRecvProxyFree(resources);
        }
      }
      if (comm->channels[c].peers[peer]->send[0].connected == 1) {
        struct sdcclConnector *conn = comm->channels[c].peers[peer]->send;
        int transport = conn->proxyConn.connection->transport;

        if (transport == TRANSPORT_NET) {
          struct sendNetResources *resources =
              (struct sendNetResources *)
                  conn->proxyConn.connection->transportResources;
          sdcclSendProxyFree(resources);
        }
      }
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclProxyDestroy(struct sdcclHeteroComm *comm) {
  if (comm->proxyState->initialized == 1) {
    INFO(SDCCL_PROXY, "sdcclProxyDestroy: sending stop to service thread...");
    int type = sdcclProxyMsgStop;
    sdcclSocketSend(&comm->proxyState->peerSock, &type, sizeof(int));
    comm->proxyState->kernelState.stop = 1;
    INFO(SDCCL_PROXY, "sdcclProxyDestroy: joining service thread...");
    pthread_join(comm->proxyState->thread, nullptr);
    INFO(SDCCL_PROXY, "sdcclProxyDestroy: service thread joined, freeing...");
    sdcclProxyFree(comm);
    INFO(SDCCL_PROXY, "sdcclProxyDestroy: done");
  }
  return sdcclSuccess;
}
