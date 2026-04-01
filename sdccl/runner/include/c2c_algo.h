#ifndef SDCCL_C2C_ALGO_H_
#define SDCCL_C2C_ALGO_H_

#include "adaptor.h"
#include "sdccl.h"
#include "sdccl_hetero.h"
#include "group.h"
#include "param.h"
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <unordered_map>

typedef enum {
  sdcclAlgoSequential = 0,
  sdcclAlgoPipeline = 1,
  sdcclAlgoInput = 2
} sdcclAlgorithm_t;

size_t getC2cCommPatternHash(size_t count, size_t rootClusterId,
                             sdcclCommOp_t commOp, sdcclRedOp_t redOp,
                             sdcclComm_t comm);

template <typename Key, typename Value>
class sdcclLRUCache {
public:
  sdcclLRUCache(size_t capacity) : capacity_(capacity) {}

  bool get(const Key &key, Value &value) {
    auto it = cacheMap_.find(key);
    if (it == cacheMap_.end())
      return false;

    // Move the accessed item to the front of the list
    cacheItems_.splice(cacheItems_.begin(), cacheItems_, it->second);
    value = it->second->second;
    return true;
  }

  void put(const Key &key, const Value &value) {
    auto it = cacheMap_.find(key);
    if (it != cacheMap_.end()) {
      // Update and move to front
      it->second->second = value;
      cacheItems_.splice(cacheItems_.begin(), cacheItems_, it->second);
    } else {
      // Insert new element
      if (cacheItems_.size() == capacity_) {
        // Remove least recently used item
        auto lru = cacheItems_.back();
        cacheMap_.erase(lru.first);
        cacheItems_.pop_back();
      }
      cacheItems_.emplace_front(key, value);
      cacheMap_[key] = cacheItems_.begin();
    }
  }

private:
  size_t capacity_;
  std::list<std::pair<Key, Value>> cacheItems_;
  std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator>
      cacheMap_;
};

struct sdcclBufferInfo {
public:
  sdcclBufferInfo(size_t offset, size_t count, int clusterIdToSend, int isRecv,
                   int isScheduled, int peerRank, int loopId)
      : offset_(offset), count_(count), clusterIdToSend_(clusterIdToSend),
        isRecv_(isRecv), isScheduled_(isScheduled), peerRank_(peerRank),
        loopId_(loopId) {}
  ~sdcclBufferInfo() {}

  size_t offset_;
  size_t count_;
  int clusterIdToSend_; // only required for send
  int isRecv_;          // 0: send, 1: recv
  int isScheduled_;     // 0: un-scheduled, 1: scheduled
  int peerRank_;
  int loopId_;
};

class sdcclInterRankBufferInfoManager {
public:
  sdcclInterRankBufferInfoManager(size_t totalCount);
  ~sdcclInterRankBufferInfoManager();
  sdcclInterRankBufferInfoManager() = default;
  sdcclInterRankBufferInfoManager(const sdcclInterRankBufferInfoManager &) =
      default;

  bool checkIfPossibleToPush(int clusterId, int rank, size_t offset,
                             size_t count);
  bool checkIfPossibleToSplitAndPush(int clusterId, int rank, size_t offset,
                                     size_t count, size_t *splitCount,
                                     int *pushMode);
  bool checkIsFull(int clusterId, int rank);
  bool checkIsScheduled(int clusterId, int rank);
  std::list<sdcclBufferInfo> &getBufferInfoList(int clusterId, int rank);
  void pushBackBufferInfo(int clusterId, int rank, size_t offset, size_t count,
                          int clusterIdToSend, int isRecv, int isScheduled,
                          int peerRank, int loopId);
  void popFrontBufferInfo(int clusterId, int rank);
  void resetBufferInfo();
  void printBufferInfo(int step); // 0: intial, 1: internal, 2: final

  size_t totalCount_; // total communication count
  std::map<int, std::map<int, std::list<sdcclBufferInfo>>>
      bufferInfos_; // map<clusterId, map<rank, list[struct{offset, count,
                    // isRecv, isScheduled}]>>
};

class sdcclC2cP2pOp {
public:
  sdcclC2cP2pOp(int rank, int peerRank, size_t offset, size_t count,
                 int isRecv);
  ~sdcclC2cP2pOp();

  sdcclResult_t run(void *buff, sdcclDataType_t datatype, sdcclComm_t comm,
                     sdcclStream_t stream);

  int rank_;
  int peerRank_;
  size_t offset_;
  size_t count_;
  int isRecv_; // 0: send, 1: recv
};

class sdcclC2cHomoFunc {
public:
  sdcclC2cHomoFunc(int rootRank, int sendType, int recvType, size_t sendOffset,
                    size_t recvOffset, size_t count, int homoType,
                    sdcclCommOp_t commOp);
  sdcclC2cHomoFunc(
      int rootRank, int sendType, int recvType, size_t sendOffset,
      size_t recvOffset, size_t count, int homoType, sdcclCommOp_t commOp,
      sdcclInterRankBufferInfoManager interRankBufferInfoManager);
  ~sdcclC2cHomoFunc();

  sdcclResult_t run(const void *sendbuff, void *recvbuff, void *scratchbuff,
                     sdcclDataType_t datatype, sdcclRedOp_t redOp, int root,
                     sdcclComm_t comm, sdcclStream_t stream,
                     size_t *sendCounts = nullptr, size_t *sDispls = nullptr,
                     size_t *recvCounts = nullptr, size_t *rDispls = nullptr);

  sdcclC2cHomoFunc(FILE *file, size_t chunksize);

  int rootRank_;
  int sendType_;
  int recvType_;
  size_t sendOffset_;
  size_t recvOffset_;
  size_t count_;
  int homoType_;
  sdcclCommOp_t commOp_;
  sdcclInterRankBufferInfoManager interRankBufferInfoManager_;
};

class sdcclC2cHeteroFunc {
public:
  friend class sdcclAlgoTimeEstimator;
  friend void serializeHeteroFunc(FILE *file, size_t chunksize,
                                  const sdcclC2cHeteroFunc &func, int indent);
  sdcclC2cHeteroFunc();
  ~sdcclC2cHeteroFunc();

  void addP2pOp(int rank, int peerRank, size_t offset, size_t count,
                int isRecv);
  sdcclResult_t run(void *sendbuff, void *recvbuff, sdcclDataType_t datatype,
                     sdcclComm_t comm, sdcclStream_t stream);
  sdcclC2cHeteroFunc(FILE *file, size_t chunksize);

private:
  std::vector<sdcclC2cP2pOp> p2pOps_;
};

class sdcclC2cRefreshFunc {
public:
  sdcclC2cRefreshFunc();
  sdcclC2cRefreshFunc(size_t offset, size_t count, size_t totalCount,
                       sdcclRedOp_t redOp);
  sdcclC2cRefreshFunc(int bufftype, size_t start, size_t offset, size_t count,
                       size_t totalCount, sdcclRedOp_t redOp);
  ~sdcclC2cRefreshFunc();

  sdcclResult_t run(void *recvbuff, void *scratchbuff,
                     sdcclDataType_t datatype, sdcclStream_t stream);

  int bufftype_;
  size_t start_;
  size_t offset_;
  size_t count_;
  size_t totalCount_;
  sdcclRedOp_t redOp_;
};

class sdcclC2cPlanner {
public:
  friend class sdcclAlgoTimeEstimator;
  sdcclC2cPlanner(size_t sendCount, size_t recvCount, int rootRank,
                   sdcclComm_t comm, sdcclCommOp_t commOp,
                   sdcclRedOp_t redOp);
  ~sdcclC2cPlanner();
  sdcclC2cPlanner() = default;
  sdcclC2cPlanner(const sdcclC2cPlanner &) = default;
  // constructor for reading a c2c algorithm from an xml input file
  sdcclC2cPlanner(const char *path);
  sdcclC2cPlanner &operator=(const sdcclC2cPlanner &) = default;

  sdcclCommOp_t getC2cHomoCommOp(int homoType, int mode);
  // import a planner from an xml file
  sdcclResult_t importXml(const char *prefix);
  // export a planner to an xml file
  sdcclResult_t exportXml(const char *prefix);
  sdcclResult_t refresh(
      int isSendRecv); // 0: refresh recv info only; 1: refresh send+recv info
  sdcclResult_t searchHeteroSendRecvOps(int searchMethod,
                                         int loopId); // 0: DFS; 1: BFS
  sdcclResult_t findStrategy();
  sdcclResult_t execute(const void *sendbuff, void *recvbuff,
                         sdcclDataType_t datatype, int root,
                         sdcclStream_t stream, size_t *sendCounts = nullptr,
                         size_t *sDispls = nullptr,
                         size_t *recvCounts = nullptr,
                         size_t *rDispls = nullptr);

private:
  int nSeqPreSteps_;
  int nPipePreSteps_;
  int nSeqInterSteps_;
  int nPipePostSteps_;
  int nSeqPostSteps_;
  size_t nchunks_;
  size_t sendCount_;
  size_t recvCount_;
  int rootRank_; // used for gather, scatter
  sdcclComm_t comm_;
  sdcclCommOp_t commOp_;
  sdcclRedOp_t redOp_;
  size_t *sendCounts_; // used for alltoallv, etc.
  size_t *sDispls_;
  size_t *recvCounts_;
  size_t *rDispls_;
  std::vector<std::vector<int>> clusterInterRankList_;
  int clusterId_;
  int rank_; // global rank
  int homoMyRank_;
  int homoRootRank_;
  int homoRanks_;
  int homoInterMyRank_;
  int homoInterRootRank_;
  int homoInterRanks_;
  size_t totalCount_; // equal to either sendCount_ or recvCount_
  int rootClusterId_;
  int isRootCluster_;
  int clusterCount_;
  int clusterOffset_;
  int multiNic_;
  int eachNicPerRank_;
  int strategyFound_;
  sdcclInterRankBufferInfoManager interRankBufferInfoManager_;
  sdcclC2cRefreshFunc refreshFunc_;
  sdcclAlgorithm_t algorithm_;
  std::vector<std::vector<sdcclC2cHomoFunc>> preHomoFuncSteps_;
  std::vector<std::vector<sdcclC2cHeteroFunc>> heteroFuncSteps_;
  std::vector<std::vector<sdcclC2cHomoFunc>> homoInterFuncSteps_;
  std::vector<std::vector<sdcclC2cHomoFunc>> postHomoFuncSteps_;
  void *scratchBuffer_; // used for intermediate processing
};

#endif // end include guard
