#include "cost_model.h"
#include "topo.h"

constexpr size_t CHUNK_SIZE = 4ULL * 1024 * 1024;
const float sdcclLatMap[SDCCL_VENDOR_NUM][2] = {
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

sdcclResult_t sdcclAlgoTimeEstimator::getAlgoTime(float *time) {
  const char *interServerTopoFile =
      sdcclGetEnv("SDCCL_INTERSERVER_ROUTE_FILE");
  if (interServerTopoFile) {
    // algo time estimator depends on cluster level topology detection
    float preHomoTime, heteroTime, postHomoTime;
    INFO(SDCCL_GRAPH, "COST_MODEL: getting time for prehomo funcs");
    SDCCLCHECK(getPreHomoAlgoTime(&preHomoTime));
    INFO(SDCCL_GRAPH, "COST_MODEL: getting time for hetero funcs");
    SDCCLCHECK(getHeteroAlgoTime(&heteroTime));
    INFO(SDCCL_GRAPH, "COST_MODEL: getting time for posthomo funcs");
    SDCCLCHECK(getPostHomoAlgoTime(&postHomoTime));
    *time = preHomoTime + heteroTime + postHomoTime;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclAlgoTimeEstimator::getPreHomoAlgoTime(float *time) {
  sdcclComm_t comm = planner_.comm_;
  auto &preHomoFuncs =
      planner_.preHomoFuncSteps_[0]; // all clusters perform the same algo
  float totalPreHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->clusterSizes[i]; // get how many ranks are in this cluster
    float preHomoTimeForCluster = 0.0;
    for (auto &func : preHomoFuncs) {
      float algoTime = 0.0;
      SDCCLCHECK(getHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      preHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPreHomoTime = std::max(totalPreHomoTime, preHomoTimeForCluster);
  }
  *time = totalPreHomoTime;
  return sdcclSuccess;
}

sdcclResult_t sdcclAlgoTimeEstimator::getPostHomoAlgoTime(float *time) {
  sdcclComm_t comm = planner_.comm_;
  auto &postHomoFuncs = planner_.postHomoFuncSteps_[0];
  float totalPostHomoTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterRankSize =
        comm->clusterSizes[i]; // get how many ranks are in this cluster
    float postHomoTimeForCluster = 0.0;
    for (auto &func : postHomoFuncs) {
      float algoTime = 0.0;
      SDCCLCHECK(getHomoAlgoTime(func, clusterRankSize, vendor, &algoTime));
      postHomoTimeForCluster += algoTime;
    }
    // get the max time for all clusters
    totalPostHomoTime = std::max(totalPostHomoTime, postHomoTimeForCluster);
  }
  *time = totalPostHomoTime;
  return sdcclSuccess;
}

sdcclResult_t sdcclAlgoTimeEstimator::getHomoAlgoTime(
    sdcclC2cHomoFunc &homoFunc, int rankSize, int vendor, float *time) {
  float defaultTime = 0.0;
  *time = defaultTime;
  return sdcclSuccess;
}

sdcclResult_t sdcclAlgoTimeEstimator::getHomoInterAlgoTime(int loop,
                                                             float *time) {
  sdcclComm_t comm = planner_.comm_;
  auto &homoFunc = planner_.homoInterFuncSteps_[0][loop];
  // getHomoAlgoTime
  float totalHomoInterTime = 0.0;
  // compute the execution time for all clusters
  // use the max time for all clusters
  for (int i = 0; i < comm->nclusters; i++) {
    int vendor = comm->clusterVendorMap[i];
    int clusterInterRankSize = planner_.clusterInterRankList_[i].size();
    float homoInterTimeForCluster = 0.0;
    SDCCLCHECK(getHomoAlgoTime(homoFunc, clusterInterRankSize, vendor,
                                &homoInterTimeForCluster));
    totalHomoInterTime = std::max(totalHomoInterTime, homoInterTimeForCluster);
  }
  *time = 0.0;
  return sdcclSuccess;
}

float sdcclAlgoTimeEstimator::getRefreshTime() {
  return 0.0; // return fixed time for now
}

sdcclResult_t sdcclAlgoTimeEstimator::getHeteroAlgoTime(float *time) {
  sdcclComm_t comm = planner_.comm_;
  sdcclHeteroComm_t heteroComm = comm->heteroComm;
  // filter out hetero funcs for each rank
  std::unordered_map<int, std::vector<sdcclC2cHeteroFunc>> heteroFuncMap;
  int heteroFuncLoops = planner_.nPipePreSteps_ + planner_.nSeqInterSteps_ +
                        planner_.nPipePostSteps_;
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  // get all interRanks
  std::vector<int> interRanks;
  std::unordered_map<uint64_t, std::vector<int>>
      nicRankMap; // {nicGuid: vector<rankId>} record the ranks that share the
                  // same nic
  INFO(SDCCL_GRAPH, "COST_MODEL: fill nicRankMap");
  for (size_t j = 0; j < clusterInterRankList.size(); j++) {
    for (size_t z = 0; z < clusterInterRankList[j].size(); z++) {
      int rank = clusterInterRankList[j][z];
      interRanks.push_back(rank);
      struct sdcclTopoServer *server;
      struct sdcclTopoNode *net;
      // get server of current rank
      SDCCLCHECK(sdcclTopoGetServerFromRank(rank, heteroComm->interServerTopo,
                                              heteroComm->topoServer, &server));
      // get local nic used by current rank
      SDCCLCHECK(sdcclTopoGetLocalNetNode(server, rank, &net));
      INFO(SDCCL_GRAPH, "COST_MODEL: nicRankMap[%lx] = %d", net->net.guid,
           rank);
      nicRankMap[net->net.guid].push_back(rank);
    }
  }
  INFO(SDCCL_GRAPH, "COST_MODEL: interRanks size = %lu", interRanks.size());
  for (int &rank : interRanks) {
    INFO(SDCCL_GRAPH, "COST_MODEL: generating heteroFunc for rank %d", rank);
    heteroFuncMap[rank].resize(heteroFuncLoops);
    for (int i = 0; i < heteroFuncLoops; i++) {
      INFO(SDCCL_GRAPH, "COST_MODEL: heteroFunc generation loop %d", i);
      sdcclC2cHeteroFunc &heteroFunc = heteroFuncMap[rank][i];
      if (planner_.multiNic_) {
        generateHeteroFuncForMultiNic(rank, i, heteroFunc);
      } else {
        generateHeteroFuncForSingleNic(rank, heteroFunc);
      }
    }
  }
  float totalTime = 0.0;
  for (int i = 0; i < heteroFuncLoops; i++) {
    INFO(SDCCL_GRAPH, "COST_MODEL: heteroFunc loop %d", i);
    // get total send/recv time for each nic in case multiple gpus share a nic
    float timePerLoop = 0.0;
    timePerLoop += getRefreshTime();
    float sendRecvTime = 0.0;
    for (auto it = nicRankMap.begin(); it != nicRankMap.end(); it++) {
      uint64_t netGuid = it->first;
      // total p2p time of a nic
      float p2pTime = getP2pTimePerNic(netGuid, nicRankMap, heteroFuncMap);
      sendRecvTime = std::max(sendRecvTime, p2pTime);
    }
    timePerLoop += sendRecvTime;
    float homoInterTime = 0.0;
    INFO(SDCCL_GRAPH, "COST_MODEL: getting homoInter time for loop %d", i);
    SDCCLCHECK(getHomoInterAlgoTime(i, &homoInterTime));
    timePerLoop += homoInterTime;
    totalTime += timePerLoop;
  }

  *time = totalTime;

  return sdcclSuccess;
}

void sdcclAlgoTimeEstimator::generateHeteroFuncForMultiNic(
    int rank, int loop, sdcclC2cHeteroFunc &heteroFunc) {
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  auto &interRankBufferInfoManager = planner_.interRankBufferInfoManager_;
  for (size_t j = 0; j < clusterInterRankList.size(); j++) {
    for (size_t z = 0; z < clusterInterRankList[j].size(); z++) {
      if (rank == clusterInterRankList[j][z]) {
        auto &rankList = interRankBufferInfoManager.getBufferInfoList(j, rank);
        INFO(SDCCL_GRAPH, "COST_MODEL: rankList size = %lu", rankList.size());
        for (auto it = rankList.begin(); it != rankList.end(); it++) {
          if (it->loopId_ == loop) {
            INFO(SDCCL_GRAPH, "COST_MODEL: heteroFunc addP2pOp");
            heteroFunc.addP2pOp(rank, it->peerRank_, it->offset_, it->count_,
                                it->isRecv_);
          }
        }
      }
    }
  }
}

void sdcclAlgoTimeEstimator::generateHeteroFuncForSingleNic(
    int rank, sdcclC2cHeteroFunc &heteroFunc) {
  sdcclComm_t comm = planner_.comm_;
  auto &clusterInterRankList = planner_.clusterInterRankList_;
  int cid = 0;
  int clusterId = comm->clusterIds[rank];
  int homoMyRank = comm->globalRank2HomoRank[rank];
  int homoRanks = comm->clusterSizes[clusterId];
  int totalCount = planner_.totalCount_;
  for (size_t j = 0; j < clusterInterRankList.size(); ++j) {
    if (clusterId == j) {
      continue;
    }
    int homoRankToRecvFromCluster =
        (comm->globalRank2HomoRank[clusterInterRankList[clusterId][0]] - cid -
         1 + homoRanks) %
        homoRanks;
    if (homoMyRank == homoRankToRecvFromCluster) {
      heteroFunc.addP2pOp(rank, clusterInterRankList[j][0], 0, totalCount, 1);
    }
    int homoRankToSendToCluster =
        (comm->globalRank2HomoRank[clusterInterRankList[j][0]] - cid - 1 +
         comm->clusterSizes[j]) %
        comm->clusterSizes[j];
    int globalRankToSendToCluster =
        homoRankToSendToCluster -
        comm->globalRank2HomoRank[clusterInterRankList[j][0]] +
        clusterInterRankList[j][0];
    if (homoMyRank ==
        comm->globalRank2HomoRank[clusterInterRankList[clusterId][0]]) {
      heteroFunc.addP2pOp(rank, globalRankToSendToCluster, 0, totalCount, 0);
    }
    cid += 1;
  }
}

float sdcclAlgoTimeEstimator::getP2pTimePerNic(
    uint64_t netGuid,
    std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
    std::unordered_map<int, std::vector<sdcclC2cHeteroFunc>> &heteroFuncMap) {
  sdcclComm_t comm = planner_.comm_;
  sdcclHeteroComm_t heteroComm = comm->heteroComm;
  auto &rankList = nicRankMap[netGuid];
  float sendTime = 0.0;
  float recvTime = 0.0;
  for (int &rank : rankList) {
    auto &funcList = heteroFuncMap[rank];
    // get clusterId of current rank
    int clusterId = comm->clusterIds[rank];         // {rank: clusterId}
    int vendor = comm->clusterVendorMap[clusterId]; // {clusterId: vendor}
    // get cluster lat and bw
    float curClusterLat =
        sdcclLatMap[vendor][SDCCL_INTER_LAT_IDX]; // {clusterId: lat}
    for (auto &func : funcList) {
      for (auto &p2pOp : func.p2pOps_) {
        int remoteRank = p2pOp.peerRank_;
        int remoteClusterId = comm->clusterIds[remoteRank];
        int remoteVendor = comm->clusterVendorMap[remoteClusterId];
        float remoteClusterLat =
            sdcclLatMap[remoteVendor][SDCCL_INTER_LAT_IDX];
        // get nic of remote rank
        struct sdcclTopoServer *remoteServer;
        struct sdcclTopoNode *remoteNet;
        // get server of current rank
        SDCCLCHECK(
            sdcclTopoGetServerFromRank(remoteRank, heteroComm->interServerTopo,
                                        heteroComm->topoServer, &remoteServer));
        // get local nic used by current rank
        SDCCLCHECK(
            sdcclTopoGetLocalNetNode(remoteServer, remoteRank, &remoteNet));
        INFO(SDCCL_GRAPH, "COST_MODEL: localNet = %lx, remoteNet = %lx",
             remoteNet->net.guid, netGuid);
        float routeBw =
            heteroComm->interServerTopo->routeMap[netGuid][remoteNet->net.guid]
                ->interBw; // we haven't recorded all route for all servers yet
        if (p2pOp.isRecv_) {
          recvTime += getSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        } else {
          sendTime += getSendRecvTime(curClusterLat, remoteClusterLat, routeBw,
                                      p2pOp.count_, CHUNK_SIZE);
        }
      }
    }
  }
  return std::max(sendTime, recvTime);
}

float sdcclAlgoTimeEstimator::getSendRecvTime(float curClusterLat,
                                               float remoteClusterLat, float bw,
                                               int totalCount,
                                               size_t chunkSize) {
  // in the current implementation, chunks are sent in serial order
  float lat =
      std::max(curClusterLat,
               remoteClusterLat); // use the higher latency between two clusters
  size_t bytes = totalCount * getSdcclDataTypeSize(datatype);
  int steps = (bytes + chunkSize - 1) / chunkSize;
  float time = 0.0;
  int sizeSent = 0;
  for (int s = 0; s < steps; s++) {
    size_t sendSize = std::min(chunkSize, bytes - sizeSent);
    time += lat + sendSize / (1000 * bw); // convert to us (bw in GB/s)
    sizeSent += sendSize;
  }
  return time;
}
