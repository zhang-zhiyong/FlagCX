#ifndef SDCCL_COST_MODEL_H
#define SDCCL_COST_MODEL_H

#include "c2c_algo.h"
#include "sdccl.h"
#include <vector>

// typedef enum {
//     FullyConnected,
//     Ring,
// } sdcclHomoSimTopo;

// struct sdcclHomoSimInfo {
//     sdcclHomoSimTopo topo[2];
//     int npusCount[2];
//     float bandwidth[2];
//     float latency[2];
// };

constexpr int SDCCL_INTRA_LAT_IDX = 0;
constexpr int SDCCL_INTER_LAT_IDX = 1;

#define SDCCL_VENDOR_NUM 4

class sdcclAlgoTimeEstimator {
public:
  sdcclAlgoTimeEstimator(sdcclC2cPlanner &planner, sdcclDataType_t dtype)
      : planner_(planner), datatype(dtype) {}

  sdcclResult_t getAlgoTime(float *time);

private:
  sdcclResult_t getPreHomoAlgoTime(float *time);

  sdcclResult_t getPostHomoAlgoTime(float *time);

  sdcclResult_t getHomoAlgoTime(sdcclC2cHomoFunc &homoFunc, int rankSize,
                                 int vendor, float *time);

  sdcclResult_t getHeteroAlgoTime(float *time);

  sdcclResult_t getHomoInterAlgoTime(int loop, float *time);

  void generateHeteroFuncForMultiNic(int rank, int loop,
                                     sdcclC2cHeteroFunc &heteroFunc);

  void generateHeteroFuncForSingleNic(int rank,
                                      sdcclC2cHeteroFunc &heteroFunc);

  float getP2pTimePerNic(
      uint64_t netGuid,
      std::unordered_map<uint64_t, std::vector<int>> &nicRankMap,
      std::unordered_map<int, std::vector<sdcclC2cHeteroFunc>> &heteroFuncMap);

  float getRefreshTime();

  float getSendRecvTime(float curClusterLat, float remoteClusterLat, float bw,
                        int totalCount, size_t chunkSize);

  sdcclC2cPlanner &planner_;
  sdcclDataType_t datatype;
};

#endif