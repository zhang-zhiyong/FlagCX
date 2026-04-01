#include "cluster.h"
#include <cstring>

sdcclResult_t parseClusterSplitList(const char *input,
                                     std::vector<int> &output) {
  output.clear();
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ',')) {
    try {
      int value = std::stoi(token);
      output.push_back(value);
    } catch (const std::exception &e) {
      WARN("Invalid cluster split info, its format should be like '2,4,8,...'");
      return sdcclSystemError;
    }
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclCollectClusterInfos(const sdcclVendor *allData,
                                         sdcclCommunicatorType_t *type,
                                         int *homoRank, int *homoRootRank,
                                         int *homoRanks, int *clusterId,
                                         int *clusterInterRank, int *ncluster,
                                         int rank, int nranks) {
  *homoRank = rank;
  *homoRootRank = 0;
  *homoRanks = 1;
  *clusterId = 0;
  *clusterInterRank = -1; // deprecated, to be removed
  *ncluster = 1;
  *type = sdcclCommunicatorHomo;

  if (nranks <= 1)
    return sdcclSuccess;

  std::map<std::string, int> clusterMap;
  clusterMap[allData[0].internal] = 1;
  int numClusters = 1;
  int currCluster = 0;
  int aggRanks = 1;
  int localHomoRootRank = 0;
  std::string myCls = allData[rank].internal;
  for (int i = 1; i < nranks; ++i) {
    std::string cls = allData[i].internal;
    auto it = clusterMap.find(cls);
    if (it != clusterMap.end()) {
      it->second = it->second + 1;
    } else {
      clusterMap[cls] = 1;
      numClusters += 1;
      if (myCls == cls) {
        *homoRank = *homoRank - aggRanks;
        currCluster = numClusters - 1;
        localHomoRootRank = i;
      }
    }
    aggRanks += 1;

    if (i == rank) {
      *homoRootRank = localHomoRootRank;
    }
  }

  *homoRanks = clusterMap[myCls];

  if (clusterMap.size() > 1) {
    *type = sdcclCommunicatorHybrid;
  } else {
    *type = sdcclCommunicatorHomo;
  }

  if (*type == sdcclCommunicatorHybrid) {
    *clusterId = currCluster;
    *ncluster = numClusters;
  }

  // split and obtain sub-clusters
  const char *clusterSplitInfo = sdcclGetEnv("SDCCL_CLUSTER_SPLIT_LIST");
  if (clusterSplitInfo != NULL) {
    std::vector<int> clusterSplitList;
    SDCCLCHECK(parseClusterSplitList(clusterSplitInfo, clusterSplitList));
    if (*ncluster != int(clusterSplitList.size())) {
      WARN("Invalid cluster split info, its length should be equal to the "
           "number of homogeneous cluster");
      return sdcclSystemError;
    }

    int subClusterId = 0;
    for (int i = 0; i < currCluster; ++i) {
      subClusterId += clusterSplitList[i];
    }
    int subHomoRanks = (*homoRanks) / clusterSplitList[currCluster];
    int hasRes =
        (((*homoRank) / subHomoRanks) >= clusterSplitList[currCluster]) ? 1 : 0;
    subClusterId += (hasRes == 1) ? ((*homoRank) / subHomoRanks) - 1
                                  : ((*homoRank) / subHomoRanks);
    int subHomoRank = (hasRes == 1)
                          ? subHomoRanks + ((*homoRank) % subHomoRanks)
                          : ((*homoRank) % subHomoRanks);
    int subHomoRootRank = rank - subHomoRank;
    if (hasRes == 1 ||
        ((*homoRank) / subHomoRanks) == clusterSplitList[currCluster] - 1) {
      subHomoRanks += (*homoRanks) % clusterSplitList[currCluster];
    }
    int subNClusters = 0;
    for (int i = 0; i < (*ncluster); ++i) {
      subNClusters += clusterSplitList[i];
    }
    *homoRank = subHomoRank;
    *homoRootRank = subHomoRootRank;
    *homoRanks = subHomoRanks;
    *clusterId = subClusterId;
    *ncluster = subNClusters;
    *type =
        (subNClusters > 1) ? sdcclCommunicatorHybrid : sdcclCommunicatorHomo;
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclFillClusterVendorInfo(const sdcclVendor *allData,
                                           sdcclComm *comm, int *clusterIdData,
                                           int nranks, int ncluster) {
  comm->clusterVendorMap.resize(ncluster);
  for (int i = 0; i < nranks; i++) {
    std::string vendor = allData[i].internal;
    int cluster = clusterIdData[i];
    if (vendor == "NVIDIA") {
      comm->clusterVendorMap[cluster] = SDCCL_VENDOR_NVIDIA;
    } else if (vendor == "ILUVATAR_COREX") {
      comm->clusterVendorMap[cluster] = SDCCL_VENDOR_ILUVATAR_COREX;
    } else if (vendor == "MLU") {
      comm->clusterVendorMap[cluster] = SDCCL_VENDOR_MLU;
    } else if (vendor == "METAX") {
      comm->clusterVendorMap[cluster] = SDCCL_VENDOR_METAX;
    }
  }
  return sdcclSuccess;
}