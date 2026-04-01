/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "group.h"
#include "adaptor.h"
#include "assert.h"
#include "debug.h"
#include "sdccl_hetero.h"
#include "launch_kernel.h"
#include "net.h"
#include "p2p.h"
#include "transport.h"
#include "type.h"
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <vector>

__thread int sdcclGroupDepth = 0;
__thread bool sdcclGroupJobAbortFlag = false;
__thread struct sdcclHeteroComm *sdcclGroupCommHead = nullptr;
__thread struct sdcclHeteroComm *sdcclGroupCommPreconnectHead = nullptr;
__thread sdcclResult_t sdcclGroupError = sdcclSuccess;
__thread struct sdcclGroupJob *sdcclGroupJobMainPtr = NULL;
__thread struct sdcclGroupJob sdcclGroupJobMain;
__thread int sdcclGroupBlocking = 1; /* default mode */
__thread struct sdcclIntruQueue<struct sdcclAsyncJob, &sdcclAsyncJob::next>
    sdcclAsyncJobs;

SDCCL_PARAM(P2pScheduleDisable, "P2P_SCHEDULE_DISABLE", 0);

sdcclResult_t sdcclHeteroGroupStart() {
  sdcclResult_t ret = sdcclSuccess;
  SDCCLCHECK(sdcclGroupStartInternal());
  return ret;
}

sdcclResult_t sdcclHeteroGroupEnd() {
  sdcclResult_t ret = sdcclSuccess;
  SDCCLCHECKGOTO(sdcclGroupEndInternal(), ret, exit);
exit:
  return ret;
}

struct sdcclPreconnectJob {
  struct sdcclAsyncJob base;
  struct sdcclHeteroComm *comm;
};

sdcclResult_t sdcclPreconnectFunc(struct sdcclAsyncJob *job_) {
  struct sdcclPreconnectJob *job = (struct sdcclPreconnectJob *)job_;
  struct sdcclHeteroComm *comm = job->comm;
  if (comm->proxyState->initialized == 0) {
    SDCCLCHECK(sdcclProxyInit(comm));
  }
  SDCCLCHECK(sdcclTransportP2pSetup(comm, NULL, 0));
  return sdcclSuccess;
}

/**
 * TODO: add proxy block to make sure the connect is complete
 **/

void *sdcclAsyncJobMain(void *arg) {
  struct sdcclAsyncJob *job = (struct sdcclAsyncJob *)arg;
  // sdcclSetDevice(job->comm->cudaDev);
  deviceAdaptor->setDevice(job->comm->cudaDev);
  job->result = job->func(job);
  if (job->result != sdcclSuccess) {
    INFO(SDCCL_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,
         job->result);
  }
  __atomic_store_n(&job->state, sdcclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

static int64_t p2pScheduleDisable = sdcclParamP2pScheduleDisable();

static sdcclResult_t groupLaunch(struct sdcclAsyncJob *job_) {
  sdcclResult_t ret = sdcclSuccess;
  // bool errorJobAbortFlag = false;
  struct sdcclGroupJob *gjob = (struct sdcclGroupJob *)job_;
  struct sdcclHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;

  struct sdcclHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;

  struct sdcclIntruQueue<struct sdcclAsyncJob, &sdcclAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;
  // volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  // CustomizedSchedule has the highest priority, followed by P2PSchedule,
  // with DefaultSchedule as the fallback.
  // CustomizedSchedule: |op0{s0,s1,...,sN}|...|opN{s0,s1,...,sN}|
  // P2PSchedule: |recvOps{s0,s1,...,sN}|selfCopyOps{s0}|sendOps{s0,s1,...,sN}|
  // DefaultSchedule: |op0{s0}|op1{s0}|...|opN{s0}|
  int defaultOpId = 0;
  int defaultStep = 0;
  // Each groupLaunch we create a semaphore to track the
  // p2p ops and a stream to launch host or device func
  std::shared_ptr<sdcclSemaphore> semaphore;
  if (deviceAsyncKernel) {
    semaphore = std::make_shared<sdcclDeviceSemaphore>();
  } else {
    semaphore = std::make_shared<sdcclHostSemaphore>();
  }
  sdcclStream_t launchStream = nullptr;
  sdcclEvent_t launchEvent = nullptr;
  // temporary stored proxy ops in step order
  std::map<int, std::vector<std::pair<sdcclHeteroComm *, sdcclProxyOp *>>>
      proxyOps;

  if (groupCommPreconnectHeadMain != nullptr) {
    struct sdcclHeteroComm *comm = groupCommPreconnectHeadMain;
    do {
      struct sdcclPreconnectJob *job;
      SDCCLCHECKGOTO(sdcclCalloc(&job, 1), ret, fail);
      job->base.func = sdcclPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = sdcclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->comm = job->base.comm = comm;
      sdcclIntruQueueEnqueue(asyncJobsMain, &job->base);

      struct sdcclHeteroComm *next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct sdcclHeteroComm *>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  if (!sdcclIntruQueueEmpty(asyncJobsMain)) {
    struct sdcclAsyncJob *job = sdcclIntruQueueHead(asyncJobsMain);
    do {
      SYSCHECKGOTO(
          pthread_create(&job->thread, nullptr, sdcclAsyncJobMain, job), ret,
          fail);
      job = job->next;
    } while (job != nullptr);

    job = sdcclIntruQueueHead(asyncJobsMain);
    do {
      pthread_join(job->thread, nullptr);
      job = job->next;
    } while (job != nullptr);

    if (ret != sdcclSuccess)
      goto fail;
  }

  if (groupCommHeadMain != nullptr) {
    struct sdcclHeteroComm *comm = groupCommHeadMain;
    // post all send/recv tasks
    do {
      sdcclTasks *tasks = &comm->tasks;
      int nRanks = comm->nRanks;
      int localRanks = comm->localRanks;

      // Round 0: handle self send/recv (local copy)
      {
        int peer = comm->rank;
        std::vector<sdcclTaskP2p *> sendTasks;
        std::vector<sdcclTaskP2p *> recvTasks;
        while (!sdcclIntruQueueEmpty(&tasks->peers[peer].sendQueue))
          sendTasks.push_back(
              sdcclIntruQueueDequeue(&tasks->peers[peer].sendQueue));
        while (!sdcclIntruQueueEmpty(&tasks->peers[peer].recvQueue))
          recvTasks.push_back(
              sdcclIntruQueueDequeue(&tasks->peers[peer].recvQueue));

        for (size_t i = 0; i < sendTasks.size();) {
          bool matched = false;
          for (size_t j = 0; j < recvTasks.size(); j++) {
            if (sendTasks[i]->bytes == recvTasks[j]->bytes &&
                sendTasks[i]->dtype == recvTasks[j]->dtype) {
              if (sendTasks[i]->buff != recvTasks[j]->buff) {
                sdcclProxyOp *op;
                SDCCLCHECK(sdcclCalloc(&op, 1));
                op->pattern = sdcclPatternSend;
                op->nbytes = sendTasks[i]->bytes;
                op->sendbuff = (uint8_t *)sendTasks[i]->buff;
                op->recvbuff = (uint8_t *)recvTasks[j]->buff;
                op->channelId = 0;
                op->root = peer;
                op->connection = comm->channels[op->channelId]
                                     .peers[peer]
                                     ->send[0]
                                     .proxyConn.connection;
                op->stream = sendTasks[i]->stream;
                op->event = semaphore->getEvent();
                op->args.chunkSteps = 1; // single step
                op->args.semaphore = semaphore;
                op->args.opId = sendTasks[i]->opId == INT_MAX
                                    ? (p2pScheduleDisable ? defaultOpId : 0)
                                    : sendTasks[i]->opId;
                op->args.step = sendTasks[i]->step == -1
                                    ? (p2pScheduleDisable ? defaultStep : 0)
                                    : sendTasks[i]->step;
                semaphore->addCounter(op->args.opId);
                defaultOpId++;
                SDCCLCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
                if (launchStream == nullptr) {
                  launchStream = op->stream;
                  launchEvent = op->event;
                } else {
                  SDCCLCHECK(
                      deviceAdaptor->streamWaitEvent(launchStream, op->event));
                }
                if (proxyOps.find(op->args.step) == proxyOps.end()) {
                  proxyOps[op->args.step] = std::vector<
                      std::pair<sdcclHeteroComm *, sdcclProxyOp *>>();
                }
                proxyOps[op->args.step].push_back({comm, op});
              }
              free(sendTasks[i]);
              free(recvTasks[j]);
              sendTasks.erase(sendTasks.begin() + i);
              recvTasks.erase(recvTasks.begin() + j);
              matched = true;
              break;
            }
          }
          if (!matched)
            i++;
        }
        for (auto *task : sendTasks)
          sdcclIntruQueueEnqueue(&tasks->peers[peer].sendQueue, task);
        for (auto *task : recvTasks)
          sdcclIntruQueueEnqueue(&tasks->peers[peer].recvQueue, task);
      }

      // Round 1..nRanks-1: use p2pSchedule to pair recv/send with different
      // peers
      int roundSendStep = 0;
      int roundRecvStep = 0;
      int roundOpId = 1;
      for (int round = 1; round < nRanks; round++) {
        int tmpRoundOpId = round / localRanks + 1;
        if (roundOpId != tmpRoundOpId) {
          roundSendStep = 0;
          roundRecvStep = 0;
          roundOpId = tmpRoundOpId;
        }
        int recvPeer = comm->p2pSchedule[round].recvRank;
        int sendPeer = comm->p2pSchedule[round].sendRank;
        while (!sdcclIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue) ||
               !sdcclIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
          // Process one recv task (for IPC register)
          if (!sdcclIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue)) {
            sdcclTaskP2p *p2p =
                sdcclIntruQueueDequeue(&tasks->peers[recvPeer].recvQueue);
            int peer = recvPeer;
            sdcclProxyOp *op;
            SDCCLCHECK(sdcclCalloc(&op, 1));
            op->pattern = sdcclPatternRecv;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->recv[0]
                                 .proxyConn.connection;
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = sdcclP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Receiver: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL("Receiver: [peerRank(%d), rank(%d)] -> "
                         "[peerSlotIdx(%ld), peerOpHash(%ld)]",
                         peer, comm->rank, op->args.p2pPeerSlotIdx,
                         op->args.p2pPeerOpHash);

              sdcclConnector *recvConn =
                  &comm->channels[op->channelId].peers[peer]->recv[0];
              sdcclConnector *peerConns[] = {recvConn};
              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              op->args.regBufFlag = 0;
              SDCCLCHECK(sdcclP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, peerRanks, 1,
                  /*isSender=*/false, &op->args.regBufFlag, &regOffset,
                  &peerRmtAddr, op->args.p2pPeerSlotIdx));
              if (op->args.regBufFlag) {
                INFO(SDCCL_REG,
                     "sdcclGroup P2P recv reg rank %d <- %d buff %p size %zu "
                     "offset %zu remote %p",
                     comm->rank, peer, p2p->buff, p2p->bytes, (size_t)regOffset,
                     peerRmtAddr ? (void *)(*peerRmtAddr) : NULL);
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = sdcclNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + sdcclNetChunkSize - 1) / (sdcclNetChunkSize);
              op->args.sendStepMask = sdcclNetChunks - 1;
              sdcclConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->recv};
              SDCCLCHECK(sdcclNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId =
                p2p->opId == INT_MAX
                    ? (p2pScheduleDisable ? defaultOpId : -roundOpId)
                    : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundRecvStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundRecvStep++;
            SDCCLCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              SDCCLCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<sdcclHeteroComm *, sdcclProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
          // Process one send task (for IPC lookup - after recv's register)
          if (!sdcclIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
            sdcclTaskP2p *p2p =
                sdcclIntruQueueDequeue(&tasks->peers[sendPeer].sendQueue);
            int peer = sendPeer;
            sdcclProxyOp *op;
            SDCCLCHECK(sdcclCalloc(&op, 1));
            op->pattern = sdcclPatternSend;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->send[0]
                                 .proxyConn.connection;
            op->stream = p2p->stream;
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = sdcclP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Sender: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL(
                  "Sender: [peerRank(%d), rank(%d)] -> [peerSlotIdx(%ld), "
                  "peerOpHash(%ld)]",
                  peer, comm->rank, op->args.p2pPeerSlotIdx,
                  op->args.p2pPeerOpHash);
              sdcclConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              SDCCLCHECK(sdcclP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, peerRanks, 1,
                  /*isSender=*/true, &op->args.regBufFlag, &regOffset,
                  &peerRmtAddr, op->args.p2pSlotIdx));
              // peerRmtAddr is fully resolved (rmtRegAddr + peer's userOffset)
              if (op->args.regBufFlag && peerRmtAddr) {
                op->args.p2pRmtAddr = (void *)peerRmtAddr;
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = sdcclNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + sdcclNetChunkSize - 1) / (sdcclNetChunkSize);
              op->args.sendStepMask = sdcclNetChunks - 1;
              sdcclConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              SDCCLCHECK(sdcclNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId = p2p->opId == INT_MAX
                                ? (p2pScheduleDisable ? defaultOpId : roundOpId)
                                : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundSendStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundSendStep++;
            SDCCLCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              SDCCLCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<sdcclHeteroComm *, sdcclProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
        }
      }
      tasks->p2pOrderSteps = 0;
      comm = comm->groupNext;
    } while (comm != nullptr);
  }

  // Save all proxy ops in step order
  for (auto it = proxyOps.begin(); it != proxyOps.end(); ++it) {
    for (auto pair : it->second) {
      SDCCLCHECK(sdcclProxySaveOp(pair.first, pair.second));
    }
  }

  if (launchStream != nullptr && launchEvent != nullptr) {
    if (deviceAsyncKernel) {
      SDCCLCHECK(deviceAdaptor->launchDeviceFunc(
          launchStream, deviceAsyncKernel, (void *)semaphore->getSignals()));
      // device semaphore need this event to signal completion
      SDCCLCHECK(deviceAdaptor->eventRecord(launchEvent, launchStream));
    } else {
      SDCCLCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
    }
  }

  while (!sdcclIntruQueueEmpty(asyncJobsMain)) {
    struct sdcclAsyncJob *job = sdcclIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  while (groupCommHeadMain != nullptr) {
    struct sdcclHeteroComm *comm = groupCommHeadMain;
    struct sdcclHeteroComm *next = comm->groupNext;
    (void)sdcclGroupCommLeave(comm);
    groupCommHeadMain = next;
  }
exit:
  return ret;
fail:
  goto exit;
}

static sdcclResult_t groupCleanup(struct sdcclAsyncJob *job_) {
  struct sdcclGroupJob *gjob = (struct sdcclGroupJob *)job_;
  struct sdcclHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct sdcclHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;
  struct sdcclIntruQueue<struct sdcclAsyncJob, &sdcclAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;

  // clean up preconnect comms
  while (groupCommPreconnectHeadMain != nullptr) {
    struct sdcclHeteroComm *comm = groupCommPreconnectHeadMain;
    struct sdcclHeteroComm *next = comm->preconnectNext;
    comm->preconnectNext = reinterpret_cast<struct sdcclHeteroComm *>(0x1);
    groupCommPreconnectHeadMain = next;
  }

  // clean up async jobs
  while (!sdcclIntruQueueEmpty(asyncJobsMain)) {
    struct sdcclAsyncJob *job = sdcclIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  // clean up comms
  while (groupCommHeadMain != nullptr) {
    struct sdcclHeteroComm *comm = groupCommHeadMain;
    struct sdcclHeteroComm *next = comm->groupNext;
    (void)sdcclGroupCommLeave(comm);
    groupCommHeadMain = next;
  }

  return sdcclSuccess;
}

static inline void groupResetJobState() {
  sdcclGroupBlocking = 0;
  sdcclGroupJobMainPtr = NULL;
  sdcclGroupCommPreconnectHead = nullptr;
  sdcclGroupCommHead = nullptr;
  memset(&sdcclGroupJobMain, 0, sizeof(struct sdcclGroupJob));
}

sdcclResult_t sdcclGroupEndInternal() {
  sdcclResult_t ret = sdcclSuccess;
  sdcclGroupDepth--;
  if (sdcclGroupDepth < 0)
    return sdcclSystemError;
  if (sdcclGroupDepth == 0) {
    if (sdcclGroupCommPreconnectHead || sdcclGroupCommHead) {
      sdcclGroupJobMain.groupCommHeadPtr = &sdcclGroupCommHead;
      sdcclGroupJobMain.groupCommPreconnectHeadPtr =
          &sdcclGroupCommPreconnectHead;
      sdcclGroupJobMain.asyncJobsPtr = &sdcclAsyncJobs;
      sdcclGroupJobMain.initialized = true;
      sdcclGroupJobMainPtr = &sdcclGroupJobMain;
      SDCCLCHECKGOTO(groupLaunch(&sdcclGroupJobMainPtr->base), ret, fail);
      groupResetJobState();
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&sdcclGroupJobMainPtr->base);
  groupResetJobState();
  goto exit;
}
