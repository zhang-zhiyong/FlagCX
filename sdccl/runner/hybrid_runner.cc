/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "c2c_algo.h"
#include "runner.h"

#define SDCCL_CACHE_CAPACITY 16
static sdcclLRUCache<size_t, sdcclC2cPlanner>
    planCache(SDCCL_CACHE_CAPACITY);

sdcclResult_t hybridRunnerReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  sdcclRedOp_t op, int root, sdcclComm_t comm,
                                  sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, comm->clusterIds[root],
                                         sdcclCommOpReduce, op, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClsuterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->clusterIds[root], sdcclCommOpReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner =
        sdcclC2cPlanner(count, count, root, comm, sdcclCommOpReduce, op);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->clusterIds[root], sdcclCommOpReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerGather(const void *sendbuff, void *recvbuff,
                                  size_t count, sdcclDataType_t datatype,
                                  int root, sdcclComm_t comm,
                                  sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, root, sdcclCommOpGather,
                                         sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, sdcclCommOpGather, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(count, count * comm->nranks, root, comm,
                               sdcclCommOpGather, sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, sdcclCommOpGather, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerScatter(const void *sendbuff, void *recvbuff,
                                   size_t count, sdcclDataType_t datatype,
                                   int root, sdcclComm_t comm,
                                   sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, root, sdcclCommOpScatter,
                                         sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, sdcclCommOpScatter, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(count * comm->nranks, count, root, comm,
                               sdcclCommOpScatter, sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, sdcclCommOpScatter, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                     size_t count, sdcclDataType_t datatype,
                                     int root, sdcclComm_t comm,
                                     sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, comm->clusterIds[root],
                            sdcclCommOpBroadcast, sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->clusterIds[root], sdcclCommOpBroadcast, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(count, count, root, comm, sdcclCommOpBroadcast,
                               sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->clusterIds[root], sdcclCommOpBroadcast, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, sdcclDataType_t datatype,
                                     sdcclRedOp_t op, sdcclComm_t comm,
                                     sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, comm->nclusters, sdcclCommOpAllReduce, op,
                            comm); // use nclusters as rootClusterId for hash
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->nclusters, sdcclCommOpAllReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner =
        sdcclC2cPlanner(count, count, -1, comm, sdcclCommOpAllReduce, op);
    planCache.put(hashValue, planner);
    // TODO: add estimator part
    // sdcclAlgoTimeEstimator estimator(planner, datatype);
    // float time = 0.0;
    // SDCCLCHECK(estimator.getAlgoTime(&time));
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->nclusters, sdcclCommOpAllReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                         size_t recvcount,
                                         sdcclDataType_t datatype,
                                         sdcclRedOp_t op, sdcclComm_t comm,
                                         sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      recvcount, comm->nclusters, sdcclCommOpReduceScatter, op,
      comm); // use nclusters as rootClusterId for hash
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         recvcount, comm->nclusters, sdcclCommOpReduceScatter, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(comm->nranks * recvcount, recvcount, -1, comm,
                               sdcclCommOpReduceScatter, op);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         recvcount, comm->nclusters, sdcclCommOpReduceScatter, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerAllGather(const void *sendbuff, void *recvbuff,
                                     size_t sendcount,
                                     sdcclDataType_t datatype,
                                     sdcclComm_t comm, sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      sendcount, comm->nclusters,
      sdcclCommOpAllGather, // use nclusters as rootClusterId for hash
      sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         sendcount, comm->nclusters, sdcclCommOpAllGather, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(sendcount, sendcount * comm->nranks, -1, comm,
                               sdcclCommOpAllGather, sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         sendcount, comm->nclusters, sdcclCommOpAllGather, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                    size_t count, sdcclDataType_t datatype,
                                    sdcclComm_t comm, sdcclStream_t stream) {
  // Construct sdcclC2cPlanner and find corresponding strategy
  sdcclC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, 1, // use 1 as rootClusterId for hash
                            sdcclCommOpAlltoAll, sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, 1, sdcclCommOpAlltoAll, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = sdcclC2cPlanner(count, count, -1, comm, sdcclCommOpAlltoAll,
                               sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, 1, sdcclCommOpAlltoAll, sdcclRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                     size_t *sdispls, void *recvbuff,
                                     size_t *recvcounts, size_t *rdispls,
                                     sdcclDataType_t datatype,
                                     sdcclComm_t comm, sdcclStream_t stream) {
  sdcclC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      1, 1, // use 1 both as count and rootClusterId for hash
      sdcclCommOpAlltoAllv, sdcclRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(SDCCL_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%d, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         1, 1, sdcclCommOpAlltoAllv, sdcclRedNoOp, (size_t)((uintptr_t)comm),
         hashValue);
    planner =
        sdcclC2cPlanner(1, 1, -1, comm, sdcclCommOpAlltoAllv, sdcclRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(SDCCL_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%d, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         1, 1, sdcclCommOpAlltoAllv, sdcclRedNoOp, (size_t)((uintptr_t)comm),
         hashValue);
  }
  SDCCLCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream,
                              sendcounts, sdispls, recvcounts, rdispls));
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerSend(const void *sendbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclComm_t comm, sdcclStream_t stream) {
  if (comm->clusterIds[comm->rank] == comm->clusterIds[peer]) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->send(
        sendbuff, count, datatype, comm->globalRank2HomoRank[peer],
        comm->homoComm, stream));
  } else {
    SDCCLCHECK(sdcclHeteroSend(sendbuff, count, datatype, peer,
                                 comm->heteroComm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerRecv(void *recvbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclComm_t comm, sdcclStream_t stream) {
  if (comm->clusterIds[comm->rank] == comm->clusterIds[peer]) {
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->recv(
        recvbuff, count, datatype, comm->globalRank2HomoRank[peer],
        comm->homoComm, stream));
  } else {
    SDCCLCHECK(sdcclHeteroRecv(recvbuff, count, datatype, peer,
                                 comm->heteroComm, stream));
  }
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerGroupStart() {
  SDCCLCHECK(sdcclHeteroGroupStart());
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->groupStart());
  return sdcclSuccess;
}

sdcclResult_t hybridRunnerGroupEnd() {
  SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->groupEnd());
  SDCCLCHECK(sdcclHeteroGroupEnd());
  return sdcclSuccess;
}

struct sdcclRunner hybridRunner = {
    // Communication functions
    hybridRunnerReduce, hybridRunnerGather, hybridRunnerScatter,
    hybridRunnerBroadcast, hybridRunnerAllReduce, hybridRunnerReduceScatter,
    hybridRunnerAllGather, hybridRunnerAlltoAll, hybridRunnerAlltoAllv,
    hybridRunnerSend, hybridRunnerRecv,
    // Group semantics
    hybridRunnerGroupStart, hybridRunnerGroupEnd};