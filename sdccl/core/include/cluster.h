#ifndef SDCCL_CLUSTER_H_
#define SDCCL_CLUSTER_H_

#include "adaptor.h"
#include "sdccl.h"
#include "param.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

sdcclResult_t parseClusterSplitList(const char *input,
                                     std::vector<int> &output);

sdcclResult_t sdcclCollectClusterInfos(const sdcclVendor *allData,
                                         sdcclCommunicatorType_t *type,
                                         int *homoRank, int *homoRootRank,
                                         int *homoRanks, int *clusterId,
                                         int *clusterInterRank, int *nclusters,
                                         int rank, int nranks);

sdcclResult_t sdcclFillClusterVendorInfo(const sdcclVendor *allData,
                                           sdcclComm *comm, int *clusterIdData,
                                           int nranks, int ncluster);

#endif // end include guard