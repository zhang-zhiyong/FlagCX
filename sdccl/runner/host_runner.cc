/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "runner.h"

static int hostRunnerGroupDepth = 0;
static std::vector<void *> recvHostBuffers;
static std::vector<void *> recvDeviceBuffers;
static std::vector<size_t> recvBufferSizes;

sdcclResult_t hostRunnerReduce(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                sdcclRedOp_t op, int root, sdcclComm_t comm,
                                sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reduce
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->reduce(
      buffIn, buffOut, count, datatype, op, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  if (comm->rank == root) {
    SDCCLCHECK(deviceAdaptor->deviceMemcpy(
        recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  }
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s Reduce: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerGather(const void *sendbuff, void *recvbuff,
                                size_t count, sdcclDataType_t datatype,
                                int root, sdcclComm_t comm,
                                sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: gather
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->gather(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, totalSize, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s gather: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, sdcclDataType_t datatype,
                                 int root, sdcclComm_t comm,
                                 sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, totalSize, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          totalSize, sdcclMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: scatter
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->scatter(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s Scatter: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: broadcast
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->broadcast(
      buffIn, buffOut, count, datatype, root, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s Broadcast: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allreduce
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->allReduce(
      buffIn, buffOut, count, datatype, op, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s AllReduce: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       sdcclDataType_t datatype,
                                       sdcclRedOp_t op, sdcclComm_t comm,
                                       sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t recvSize = recvcount * getSdcclDataTypeSize(datatype);
  size_t sendSize = comm->nranks * recvSize;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, sendSize, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, recvSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          sendSize, sdcclMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: reducescatter
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->reduceScatter(
      buffIn, buffOut, recvcount, datatype, op, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, recvSize, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s ReduceScatter: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, sdcclDataType_t datatype,
                                   sdcclComm_t comm, sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = sendcount * getSdcclDataTypeSize(datatype);
  size_t totalSize = comm->nranks * size;

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, totalSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: allgather
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->allGather(
      buffIn, buffOut, sendcount, datatype, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, totalSize, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s AllGather: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  sdcclComm_t comm, sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  size_t size = comm->nranks * count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: alltoall
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->alltoAll(
      buffIn, buffOut, count, datatype, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s AlltoAll: rank %d nranks %d total %.2fms "
       "(memory alloc "
       "%.2fms, memory free %.2fms, memory d2h %.2fms, memory h2d %.2fms, "
       "comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   sdcclDataType_t datatype, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  void *buffOut;
  // Calculate max possible size needed for send and receive buffers
  size_t maxSendSize = 0, maxRecvSize = 0, sendSize = 0, recvSize = 0;
  for (int i = 0; i < comm->nranks; i++) {
    sendSize = (sendcounts[i] + sdispls[i]) * getSdcclDataTypeSize(datatype);
    recvSize = (recvcounts[i] + rdispls[i]) * getSdcclDataTypeSize(datatype);
    if (sendSize > maxSendSize)
      maxSendSize = sendSize;
    if (recvSize > maxRecvSize)
      maxRecvSize = recvSize;
  }

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, maxSendSize, 0));
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, maxRecvSize, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          maxSendSize, sdcclMemcpyDeviceToHost,
                                          NULL, NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: alltoallv
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->alltoAllv(
      buffIn, sendcounts, sdispls, buffOut, recvcounts, rdispls, datatype,
      comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, buffOut, maxRecvSize, sdcclMemcpyHostToDevice, NULL, NULL));
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 5: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s AlltoAllv: rank %d nranks %d total %.2fms "
       "(memory alloc %.2fms, memory free %.2fms, memory d2h %.2fms, "
       "memory h2d %.2fms, comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_D2H] / 1e6,
       timers[TIMER_COLL_MEM_H2D] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerSend(const void *sendbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclComm_t comm, sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffIn;
  size_t size = count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffIn, size, 0));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: memcpy d2h
  timers[TIMER_COLL_MEM_D2H] = clockNano();
  SDCCLCHECK(deviceAdaptor->deviceMemcpy(buffIn, const_cast<void *>(sendbuff),
                                          size, sdcclMemcpyDeviceToHost, NULL,
                                          NULL));
  timers[TIMER_COLL_MEM_D2H] = clockNano() - timers[TIMER_COLL_MEM_D2H];

  // step 3: send
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->send(
      buffIn, count, datatype, peer, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 4: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s Send: rank %d nranks %d total %.2fms (memory "
       "alloc "
       "%.2fms, memory d2h %.2fms, comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_MEM_D2H] / 1e6, timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerRecv(void *recvbuff, size_t count,
                              sdcclDataType_t datatype, int peer,
                              sdcclComm_t comm, sdcclStream_t stream) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();
  void *buffOut;
  size_t size = count * getSdcclDataTypeSize(datatype);

  // step 1: get staged buffer
  timers[TIMER_COLL_ALLOC] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->getStagedBuffer(
      comm->hostComm, &buffOut, size, 1));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  // step 2: recv
  timers[TIMER_COLL_COMM] = clockNano();
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->recv(
      buffOut, count, datatype, peer, comm->hostComm, NULL));
  timers[TIMER_COLL_COMM] = clockNano() - timers[TIMER_COLL_COMM];

  // step 3: memcpy h2d
  timers[TIMER_COLL_MEM_H2D] = clockNano();
  if (hostRunnerGroupDepth == 0) {
    SDCCLCHECK(deviceAdaptor->deviceMemcpy(
        recvbuff, buffOut, size, sdcclMemcpyHostToDevice, NULL, NULL));
  } else {
    recvHostBuffers.push_back(buffOut);
    recvDeviceBuffers.push_back(recvbuff);
    recvBufferSizes.push_back(size);
  }
  timers[TIMER_COLL_MEM_H2D] = clockNano() - timers[TIMER_COLL_MEM_H2D];

  // step 4: free staged buffer
  timers[TIMER_COLL_FREE] = clockNano();
  // do nothing
  timers[TIMER_COLL_FREE] = clockNano() - timers[TIMER_COLL_FREE];

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(SDCCL_COLL,
       "Sdccl timings - %s Recv: rank %d nranks %d total %.2fms (memory "
       "alloc "
       "%.2fms, memory free %.2fms, memory h2d %.2fms, comm %.2fms)",
       cclAdaptors[sdcclCCLAdaptorHost]->name, comm->rank, comm->nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_ALLOC] / 1e6,
       timers[TIMER_COLL_FREE] / 1e6, timers[TIMER_COLL_MEM_H2D] / 1e6,
       timers[TIMER_COLL_COMM] / 1e6);
  return sdcclSuccess;
}

sdcclResult_t hostRunnerGroupStart() {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->groupStart());
  hostRunnerGroupDepth++;
  return sdcclSuccess;
}

sdcclResult_t hostRunnerGroupEnd() {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorHost]->groupEnd());
  hostRunnerGroupDepth--;
  if (hostRunnerGroupDepth == 0) {
    for (size_t i = 0; i < recvHostBuffers.size(); ++i) {
      SDCCLCHECK(deviceAdaptor->deviceMemcpy(
          recvDeviceBuffers[i], recvHostBuffers[i], recvBufferSizes[i],
          sdcclMemcpyHostToDevice, NULL, NULL));
    }
    recvHostBuffers.clear();
    recvDeviceBuffers.clear();
    recvBufferSizes.clear();
  }
  return sdcclSuccess;
}

struct sdcclRunner hostRunner = {
    // Communication functions
    hostRunnerReduce, hostRunnerGather, hostRunnerScatter, hostRunnerBroadcast,
    hostRunnerAllReduce, hostRunnerReduceScatter, hostRunnerAllGather,
    hostRunnerAlltoAll, hostRunnerAlltoAllv, hostRunnerSend, hostRunnerRecv,
    // Group semantics
    hostRunnerGroupStart, hostRunnerGroupEnd};