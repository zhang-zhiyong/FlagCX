#include "sdccl_hetero.h"
#include "adaptor.h"
#include "group.h"
#include "net.h"
#include "onesided.h"
#include "transport.h"
#include "type.h"

#include <climits>
#include <sched.h>

sdcclResult_t sdcclHeteroSend(const void *sendbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclHeteroComm_t comm, sdcclStream_t stream,
                                int opId, int step) {
  sdcclHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->send[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->send[0].registered == 0) {
    comm->connectSend[peer] |= (1UL << channelId);
    sdcclGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->send[0].registered = 1;
  }
  struct sdcclTaskP2p *p2p;
  struct sdcclTasks *tasks = &comm->tasks;
  SDCCLCHECK(sdcclCalloc(&p2p, 1));
  p2p->buff = (void *)sendbuff;
  p2p->bytes = count * getSdcclDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (sdcclIntruQueueEmpty(&tasks->peers[peer].sendQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  sdcclIntruQueueEnqueue(&tasks->peers[peer].sendQueue, p2p);

  sdcclGroupCommJoin(comm);
  sdcclHeteroGroupEnd();
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroRecv(void *recvbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclHeteroComm_t comm, sdcclStream_t stream,
                                int opId, int step) {
  sdcclHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->recv[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->recv[0].registered == 0) {
    comm->connectRecv[peer] |= (1UL << channelId);
    sdcclGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->recv[0].registered = 1;
  }
  struct sdcclTaskP2p *p2p;
  struct sdcclTasks *tasks = &comm->tasks;
  SDCCLCHECK(sdcclCalloc(&p2p, 1));
  p2p->buff = (void *)recvbuff;
  p2p->bytes = count * getSdcclDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (sdcclIntruQueueEmpty(&tasks->peers[peer].recvQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  sdcclIntruQueueEnqueue(&tasks->peers[peer].recvQueue, p2p);

  sdcclGroupCommJoin(comm);
  sdcclHeteroGroupEnd();
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroPut(sdcclHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx) {
  // Check if netAdaptor->iput is available
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return sdcclNotSupported;

  // Validate peer range
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("sdcclHeteroPut: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return sdcclInvalidArgument;
  }

  // Get sendComm from full-mesh connections (handle table slot 0 owns them)
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("sdcclHeteroPut: no full-mesh connections");
    return sdcclInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("sdcclHeteroPut: no sendComm for peer %d", peer);
    return sdcclInternalError;
  }

  // Get per-window MR handles from handle table
  if (srcMrIdx < 0 || srcMrIdx >= globalOneSideHandleCount || dstMrIdx < 0 ||
      dstMrIdx >= globalOneSideHandleCount) {
    WARN("sdcclHeteroPut: invalid MR index src=%d dst=%d (count=%d)", srcMrIdx,
         dstMrIdx, globalOneSideHandleCount);
    return sdcclInternalError;
  }
  void **srcHandles = (void **)globalOneSideHandleTable[srcMrIdx];
  void **dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];

  int srcRank = comm->rank;
  int dstRank = peer;
  void *request = NULL;
  SDCCLCHECK(comm->netAdaptor->iput(
      sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
      dstRank, srcHandles, dstHandles, &request));
  // Poll completion to free the IB request
  if (request != NULL) {
    int done = 0;
    while (!done) {
      SDCCLCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroGet(sdcclHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iget == NULL)
    return sdcclNotSupported;

  // Validate peer range
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("sdcclHeteroGet: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return sdcclInvalidArgument;
  }

  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("sdcclHeteroGet: no full-mesh connections");
    return sdcclInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("sdcclHeteroGet: no sendComm for peer %d", peer);
    return sdcclInternalError;
  }

  if (srcMrIdx < 0 || srcMrIdx >= globalOneSideHandleCount || dstMrIdx < 0 ||
      dstMrIdx >= globalOneSideHandleCount) {
    WARN("sdcclHeteroGet: invalid MR index src=%d dst=%d (count=%d)", srcMrIdx,
         dstMrIdx, globalOneSideHandleCount);
    return sdcclInternalError;
  }
  void **srcHandles = (void **)globalOneSideHandleTable[srcMrIdx];
  void **dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];

  int srcRank = peer;       // remote peer is the data source
  int dstRank = comm->rank; // local rank is the data destination
  void *request = NULL;
  SDCCLCHECK(comm->netAdaptor->iget(
      sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
      dstRank, srcHandles, dstHandles, &request));
  if (request != NULL) {
    int done = 0;
    while (!done) {
      SDCCLCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroPutSignal(sdcclHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue) {
  // Check if netAdaptor->iputSignal is available
  if (comm->netAdaptor == NULL || comm->netAdaptor->iputSignal == NULL)
    return sdcclNotSupported;

  // Get sendComm from full-mesh connections
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("sdcclHeteroPutSignal: no full-mesh connections");
    return sdcclInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("sdcclHeteroPutSignal: no sendComm for peer %d", peer);
    return sdcclInternalError;
  }

  int srcRank = comm->rank;
  int dstRank = peer;

  // Data handles from per-window MR table
  void **srcHandles = NULL;
  void **dstHandles = NULL;
  if (size > 0) {
    if (srcMrIdx < 0 || srcMrIdx >= globalOneSideHandleCount || dstMrIdx < 0 ||
        dstMrIdx >= globalOneSideHandleCount) {
      WARN("sdcclHeteroPutSignal: invalid MR index src=%d dst=%d", srcMrIdx,
           dstMrIdx);
      return sdcclInternalError;
    }
    srcHandles = (void **)globalOneSideHandleTable[srcMrIdx];
    dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];
  }
  void **signalHandles = (void **)globalOneSideSignalHandles;
  if (signalHandles == NULL) {
    WARN("sdcclHeteroPutSignal: globalOneSideSignalHandles not initialized");
    return sdcclInternalError;
  }
  void *request = NULL;
  SDCCLCHECK(comm->netAdaptor->iputSignal(
      sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
      dstRank, srcHandles, dstHandles, (uint64_t)signalOffset, signalHandles,
      signalValue, &request));
  // Poll completion (single CQE for chained WRITE + ATOMIC)
  if (request != NULL) {
    int done = 0;
    while (!done) {
      SDCCLCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroFlush(sdcclHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo) {
  struct sdcclOneSideHandleInfo *info =
      (struct sdcclOneSideHandleInfo *)gHandleInfo;
  if (info == NULL || info->localRecvComm == NULL ||
      info->localMrHandle == NULL)
    return sdcclNotSupported;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iflush == NULL)
    return sdcclNotSupported;

  if (size > (size_t)INT_MAX) {
    WARN("sdcclHeteroFlush: size %zu exceeds int limit", size);
    return sdcclInternalError;
  }
  void *data_arr[1] = {gpuAddr};
  int sizes_arr[1] = {(int)size};
  void *mh_arr[1] = {info->localMrHandle};
  void *request = NULL;
  SDCCLCHECK(comm->netAdaptor->iflush(info->localRecvComm, 1, data_arr,
                                       sizes_arr, mh_arr, &request));
  if (request != NULL) {
    int done = 0;
    while (!done) {
      SDCCLCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroWaitSignal(sdcclHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      sdcclStream_t stream) {
  (void)peer;
  struct sdcclOneSideHandleInfo *info =
      (struct sdcclOneSideHandleInfo *)globalOneSideSignalHandles;
  if (info == NULL || info->baseVas == NULL)
    return sdcclNotSupported;

  int myRank = comm->rank;
  void *signalAddr = (void *)(info->baseVas[myRank] + signalOffset);

  // Device-side wait (streamWaitValue64) for GPU signal buffer.
  // RMA signal buffers are GPU memory (sdcclMemAlloc) — host-side volatile
  // polling would segfault. Non-CUDA platforms return sdcclNotSupported.
  // No flush needed: FORCE_SO on signal MR guarantees PCIe ordering.
  if (stream == NULL)
    return sdcclInternalError;

  return deviceAdaptor->streamWaitValue64(stream, signalAddr, expected, 0);
}

sdcclResult_t sdcclHeteroPutValue(sdcclHeteroComm_t comm, int peer,
                                    uint64_t value, size_t dstOffset,
                                    int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return sdcclNotSupported;

  // 1. Validate staging handles
  struct sdcclOneSideHandleInfo *stagingH = globalOneSideStagingHandles;
  if (stagingH == NULL || stagingH->baseVas == NULL) {
    WARN("sdcclHeteroPutValue: staging handles not initialized");
    return sdcclInternalError;
  }

  // 2. Write value to local staging buffer
  int myRank = comm->rank;
  *(volatile uint64_t *)(stagingH->baseVas[myRank]) = value;

  // 3. Get sendComm from full-mesh connections (data handle[0] owns them)
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("sdcclHeteroPutValue: no full-mesh connections");
    return sdcclInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("sdcclHeteroPutValue: no sendComm for peer %d", peer);
    return sdcclInternalError;
  }

  // 4. Validate dst MR index
  if (dstMrIdx < 0 || dstMrIdx >= globalOneSideHandleCount) {
    WARN("sdcclHeteroPutValue: invalid dstMrIdx=%d (count=%d)", dstMrIdx,
         globalOneSideHandleCount);
    return sdcclInternalError;
  }
  void **srcHandles = (void **)stagingH;
  void **dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];

  // 5. iput: srcOffset=0 (staging buffer start), size=8 bytes
  int dstRank = peer;
  void *request = NULL;
  SDCCLCHECK(comm->netAdaptor->iput(sendComm, 0, (uint64_t)dstOffset,
                                     sizeof(uint64_t), myRank, dstRank,
                                     srcHandles, dstHandles, &request));

  // 6. Poll completion
  if (request != NULL) {
    int done = 0;
    while (!done) {
      SDCCLCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return sdcclSuccess;
}
