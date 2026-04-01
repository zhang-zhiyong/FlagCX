#ifndef SDCCL_GLOBAL_COMM_H_
#define SDCCL_GLOBAL_COMM_H_

#include "bootstrap.h"
#include "sdccl.h"
#include "sdccl_tuner.h"

#include <map>
#include <vector>

/* Opaque handle to sdcclInnerComm */
typedef struct sdcclInnerComm *sdcclInnerComm_t;

// IPC peer pointer table entry — owned by comm, referenced by devMem.
// Cleanup deferred to sdcclCommDestroy.
// to avoid cudaFree implicit device synchronization deadlock.
#define SDCCL_MAX_IPC_ENTRIES 16

struct sdcclIpcTableEntry {
  void **hostPeerPtrs; // host array: peer buffer ptrs (for ipcMemHandleClose)
  void **devPeerPtrs;  // device array: peer buffer ptrs (for cudaFree)
  int nPeers;          // number of local peers
  void *basePtr;       // own buffer ptr (skip in ipcMemHandleClose loop)
  bool inUse;          // true while a devMem references this entry
};

// Deferred device/host-pinned memory free — collected during cleanup,
// drained in sdcclCommDestroy.
#define SDCCL_MAX_DEFERRED_FREES 32

struct sdcclDeferredFree {
  void *ptr;
  int memType; // sdcclMemDevice, sdcclMemHost, etc.
};

/* Opaque handle to sdcclHeteroComm */
typedef struct sdcclHeteroComm *sdcclHeteroComm_t;

typedef enum {
  sdcclCommunicatorUnknown = 0,
  sdcclCommunicatorHomo = 1,  // Homogeneous Communicator
  sdcclCommunicatorHybrid = 2 // Hybrid Communicator
} sdcclCommunicatorType_t;

struct sdcclComm {
  // TODO: adjust code format
  int rank;
  int nranks;
  int nclusters;
  int homoRank;
  int homoRootRank;
  int homoRanks;
  int hasSingleRankHomoComm;
  sdcclCommunicatorType_t commType;
  uint64_t magic;
  volatile uint32_t *abortFlag;
  int *clusterSizes;
  int *clusterIds;
  int *globalRank2HomoRank;
  int *clusterInterRanks;
  bootstrapState *bootstrap;
  int localRank;        // intra-node rank index (computed from hostHash)
  int localRanks;       // number of ranks on this node
  int *localRankToRank; // mapping: local index -> global rank
  sdcclInnerComm_t hostComm;
  sdcclInnerComm_t homoComm;
  sdcclHeteroComm_t heteroComm;
  sdcclInnerComm_t homoInterComm;
  int homoInterRootRank;
  int homoInterMyRank;
  int homoInterRanks;
  std::vector<std::vector<int>> clusterInterRankList;
  std::vector<sdcclVendorType> clusterVendorMap;
  struct sdcclTuner *tuner;
  void *tunerContext;
  std::map<struct TunerCollCategory, sdcclInnerComm_t>
      homoCommMap; // key: commTag returned by tuner
  std::map<struct sdcclCommTag, sdcclInnerComm_t> commMap;
  std::map<struct TunerCollCategory, sdcclInnerComm_t>
      homoBestCommMap;              // key: commTag returned by tuner
  sdcclInnerComm_t tunerInnerComm; // innerComm selected by tuner
  sdcclUniqueId_t commId;
  sdcclUniqueId *uniqueIdData;
  bool isTuningWithFlagscale; // whether tuning with flagscale
  bool isTunningComm;         // whether tuning the communicator
  bool isUseSingleTunerComm;  // whether tuning with one communicator
  struct C2cSchedulePair {
    int sendCluster;
    int recvCluster;
  } * c2cSchedule; // C2C schedule for pairing send/recv operations

  // IPC peer pointer table — deferred cleanup
  struct sdcclIpcTableEntry ipcTable[SDCCL_MAX_IPC_ENTRIES];

  // Deferred device/host-pinned memory free list
  struct sdcclDeferredFree deferredFrees[SDCCL_MAX_DEFERRED_FREES];
  int deferredFreeCount;
};

#endif // end include guard
