/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "sdccl_hetero.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_impl.h"

SDCCL_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
SDCCL_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);
SDCCL_PARAM(UniRunnerUseSlicedAR, "UNIRUNNER_USE_SLICEDAR", 0);

sdcclResult_t uniRunnerReduce(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               sdcclRedOp_t op, int root, sdcclComm_t comm,
                               sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  sdcclHeteroComm_t hcomm = comm->heteroComm;
  sdcclUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  SDCCLCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, 2 * count * getSdcclDataTypeSize(datatype),
      sdcclMemDevice, stream));
  SDCCLCHECKGOTO(initUniRunner(comm, stream), res, out);
  SDCCLCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, comm),
                  res, out);
  SDCCLCHECKGOTO(runUniRunner(comm), res, out);
out:
  SDCCLCHECK(deviceAdaptor->deviceFree(scratchbuff, sdcclMemDevice, stream));
  SDCCLCHECK(cleanupUniRunner(comm));
  return res;
}

sdcclResult_t uniRunnerGather(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               int root, sdcclComm_t comm,
                               sdcclStream_t stream) {
  size_t size = count * getSdcclDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  SDCCLCHECK(sdcclHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      SDCCLCHECK(sdcclHeteroRecv(static_cast<void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  SDCCLCHECK(sdcclHeteroSend(sendbuff, count, datatype, root,
                               comm->heteroComm, stream));
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerScatter(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                int root, sdcclComm_t comm,
                                sdcclStream_t stream) {
  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  SDCCLCHECK(sdcclHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      SDCCLCHECK(sdcclHeteroSend(static_cast<const void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  SDCCLCHECK(sdcclHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclComm_t comm,
                                  sdcclStream_t stream) {
  SDCCLCHECK(sdcclHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      SDCCLCHECK(sdcclHeteroSend(sendbuff, count, datatype, r,
                                   comm->heteroComm, stream));
    }
  }
  SDCCLCHECK(sdcclHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  sdcclRedOp_t op, sdcclComm_t comm,
                                  sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  sdcclHeteroComm_t hcomm = comm->heteroComm;
  sdcclUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  SDCCLCHECK(initUniRunner(comm, stream));
  if (sdcclParamUniRunnerUseLocRed()) {
    /* initialize uniRunnerState for reduce test */
    SDCCLCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (sdcclParamUniRunnerUseRingAG()) {
    /* initialize uniRunnerState for p2p test */
    SDCCLCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (sdcclParamUniRunnerUseSlicedAR()) {
    /* initialize uniRunnerState for sliced AllReduce */
    SDCCLCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                    res, out);
  } else {
    /* initialize uniRunnerState for ring AllReduce */
    SDCCLCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  }
  SDCCLCHECK(runUniRunner(comm));
out:
  SDCCLCHECK(cleanupUniRunner(comm));
  return res;
}

sdcclResult_t uniRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      sdcclDataType_t datatype,
                                      sdcclRedOp_t op, sdcclComm_t comm,
                                      sdcclStream_t stream) {
  sdcclResult_t res = sdcclSuccess;
  sdcclHeteroComm_t hcomm = comm->heteroComm;
  sdcclUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  SDCCLCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, recvcount * comm->nranks * getSdcclDataTypeSize(datatype),
      sdcclMemDevice, stream));
  SDCCLCHECKGOTO(initUniRunner(comm, stream), res, out);
  SDCCLCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           comm),
                  res, out);
  SDCCLCHECKGOTO(runUniRunner(comm), res, out);
out:
  SDCCLCHECK(deviceAdaptor->deviceFree(scratchbuff, sdcclMemDevice, stream));
  SDCCLCHECK(cleanupUniRunner(comm));
  return res;
}

sdcclResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, sdcclDataType_t datatype,
                                  sdcclComm_t comm, sdcclStream_t stream) {
  size_t size = sendcount * getSdcclDataTypeSize(datatype);
  char *bufferOut = static_cast<char *>(recvbuff);
  SDCCLCHECK(sdcclHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    SDCCLCHECK(sdcclHeteroSend(sendbuff, sendcount, datatype, r,
                                 comm->heteroComm, stream));
    SDCCLCHECK(sdcclHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 sendcount, datatype, r, comm->heteroComm,
                                 stream));
  }
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 sdcclComm_t comm, sdcclStream_t stream) {
  size_t size = count * getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  SDCCLCHECK(sdcclHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    SDCCLCHECK(sdcclHeteroSend(static_cast<const void *>(bufferIn + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
    SDCCLCHECK(sdcclHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
  }
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  sdcclDataType_t datatype, sdcclComm_t comm,
                                  sdcclStream_t stream) {
  size_t size = getSdcclDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  SDCCLCHECK(sdcclHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    if (sdcclCCLAdaptorNeedSendrecv(sendcounts[r])) {
      SDCCLCHECK(sdcclHeteroSend(
          static_cast<const void *>(bufferIn + sdispls[r] * size),
          sendcounts[r], datatype, r, comm->heteroComm, stream));
    }
    if (sdcclCCLAdaptorNeedSendrecv(recvcounts[r])) {
      SDCCLCHECK(sdcclHeteroRecv(
          static_cast<void *>(bufferOut + rdispls[r] * size), recvcounts[r],
          datatype, r, comm->heteroComm, stream));
    }
  }
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerSend(const void *sendbuff, size_t count,
                             sdcclDataType_t datatype, int peer,
                             sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclHeteroSend(sendbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return sdcclSuccess;
}

sdcclResult_t uniRunnerRecv(void *recvbuff, size_t count,
                             sdcclDataType_t datatype, int peer,
                             sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(sdcclHeteroRecv(recvbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return sdcclSuccess;
}

sdcclResult_t uniRunnerGroupStart() {
  SDCCLCHECK(sdcclHeteroGroupStart());
  return sdcclSuccess;
}

sdcclResult_t uniRunnerGroupEnd() {
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

struct sdcclRunner uniRunner = {
    // Communication functions
    uniRunnerReduce, uniRunnerGather, uniRunnerScatter, uniRunnerBroadcast,
    uniRunnerAllReduce, uniRunnerReduceScatter, uniRunnerAllGather,
    uniRunnerAlltoAll, uniRunnerAlltoAllv, uniRunnerSend, uniRunnerRecv,
    // Group semantics
    uniRunnerGroupStart, uniRunnerGroupEnd};