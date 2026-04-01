/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "sdccl_tuner.h"
#include "runner.h"

sdcclResult_t homoRunnerReduce(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                sdcclRedOp_t op, int root, sdcclComm_t comm,
                                sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->reduce(
                            sendbuff, recvbuff, count, datatype, op, root,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpReduce, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerGather(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                int root, sdcclComm_t comm,
                                sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->gather(
        sendbuff, recvbuff, count, datatype, root, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->gather(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpGather, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclComm_t comm,
                                 sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->scatter(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpScatter, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->broadcast(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpBroadcast, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->allReduce(
                            sendbuff, recvbuff, count, datatype, op,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpAllReduce, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       sdcclDataType_t datatype,
                                       sdcclRedOp_t op, sdcclComm_t comm,
                                       sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->reduceScatter(
                            sendbuff, recvbuff, recvcount, datatype, op,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpReduceScatter, recvcount, datatype,
                        stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, sdcclDataType_t datatype,
                                   sdcclComm_t comm, sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(cclAdaptors[sdcclCCLAdaptorDevice]->allGather(
                            sendbuff, recvbuff, sendcount, datatype,
                            comm->tunerInnerComm, stream),
                        comm, sdcclCommOpAllGather, sendcount, datatype,
                        stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  sdcclComm_t comm, sdcclStream_t stream) {
  if (comm->tuner == NULL) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm->homoComm, stream));
  } else {
    SDCCLCALLWITHTUNER(
        cclAdaptors[sdcclCCLAdaptorDevice]->alltoAll(
            sendbuff, recvbuff, count, datatype, comm->tunerInnerComm, stream),
        comm, sdcclCommOpAlltoAll, count, datatype, stream);
  }
  return sdcclSuccess;
}

sdcclResult_t homoRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   sdcclDataType_t datatype, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->alltoAllv(
      sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
      comm->homoComm, stream));
  return sdcclSuccess;
}

sdcclResult_t homoRunnerSend(const void *sendbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->send(
      sendbuff, count, datatype, peer, comm->homoComm, stream));
  return sdcclSuccess;
}

sdcclResult_t homoRunnerRecv(void *recvbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclComm_t comm, sdcclStream_t stream) {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->recv(
      recvbuff, count, datatype, peer, comm->homoComm, stream));
  return sdcclSuccess;
}

sdcclResult_t homoRunnerGroupStart() {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->groupStart());
  return sdcclSuccess;
}

sdcclResult_t homoRunnerGroupEnd() {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->groupEnd());
  return sdcclSuccess;
}

struct sdcclRunner homoRunner = {
    // Communication functions
    homoRunnerReduce, homoRunnerGather, homoRunnerScatter, homoRunnerBroadcast,
    homoRunnerAllReduce, homoRunnerReduceScatter, homoRunnerAllGather,
    homoRunnerAlltoAll, homoRunnerAlltoAllv, homoRunnerSend, homoRunnerRecv,
    // Group semantics
    homoRunnerGroupStart, homoRunnerGroupEnd};