#include "sdccl_tuner.h"
#include "adaptor.h"
#include "check.h"
#include "param.h"
#include "timer.h"
#include "tuner_util.h"
#include "utils.h"
#include <cfloat>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
// A category of collective operation. the minimal unit for tuning.

bool operator<(const struct TunerCollCategory &lhs,
               const struct TunerCollCategory &rhs) {
  if (lhs.collType != rhs.collType) {
    return lhs.collType < rhs.collType;
  }
  return lhs.nBytes < rhs.nBytes;
}

static_assert(SDCCL_PROFILE_KEY_MAX_LENGTH >= 20,
              "SDCCL_PROFILE_KEY_MAX_LENGTH < 20, too short");

// Key used for time profiling
struct TunerProfileKey {
  size_t nBytes;
  uint32_t collType; // sdcclCommOp_t
  uint32_t seqId;    // sequence id of collective within this TunerCollCategory
  uint32_t commTagIdx; // index of commTag in configList

  // constructors
  TunerProfileKey() : nBytes(0), collType(0), seqId(0), commTagIdx(0) {}
  TunerProfileKey(size_t n, uint32_t c, uint32_t s, uint32_t i)
      : nBytes(n), collType(c), seqId(s), commTagIdx(i) {}
  TunerProfileKey(const struct sdcclProfileKey &k) {
    const char *ptr = k.key;
    memcpy(&nBytes, ptr, sizeof(nBytes));
    ptr += sizeof(nBytes);
    memcpy(&collType, ptr, sizeof(collType));
    ptr += sizeof(collType);
    memcpy(&seqId, ptr, sizeof(seqId));
    ptr += sizeof(seqId);
    memcpy(&commTagIdx, ptr, sizeof(commTagIdx));
  }

  // conversion function
  operator struct sdcclProfileKey() const {
    struct sdcclProfileKey k;
    memset(k.key, 0, SDCCL_PROFILE_KEY_MAX_LENGTH);
    char *ptr = k.key;
    memcpy(ptr, &nBytes, sizeof(nBytes));
    ptr += sizeof(nBytes);
    memcpy(ptr, &collType, sizeof(collType));
    ptr += sizeof(collType);
    memcpy(ptr, &seqId, sizeof(seqId));
    ptr += sizeof(seqId);
    memcpy(ptr, &commTagIdx, sizeof(commTagIdx));
    return k;
  }

  bool operator<(const TunerProfileKey &other) const {
    if (nBytes != other.nBytes) {
      return nBytes < other.nBytes;
    } else if (collType != other.collType) {
      return collType < other.collType;
    } else if (seqId != other.seqId) {
      return seqId < other.seqId;
    }
    return commTagIdx < other.commTagIdx;
  }

  bool operator==(const TunerProfileKey &other) const {
    return (nBytes == other.nBytes) && (collType == other.collType) &&
           (seqId == other.seqId) && (commTagIdx == other.commTagIdx);
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "{nBytes=" << nBytes << ",collType=" << collType
        << ",seqId=" << seqId << ",commTagIdx=" << commTagIdx << "}";
    return oss.str();
  }
};

// (collType,nBytes,configIdx)
// Used for counting the number of configs corresponding to each Collective Op
struct TunerCommTagCounterKey {
  size_t nBytes;
  uint32_t collType;   // sdcclCommOp_t
  uint32_t commTagIdx; // index of commTag in configList
};

static bool operator<(const struct TunerCommTagCounterKey &lhs,
                      const struct TunerCommTagCounterKey &rhs) {
  if (lhs.nBytes != rhs.nBytes)
    return lhs.nBytes < rhs.nBytes;
  if (lhs.collType != rhs.collType)
    return lhs.collType < rhs.collType;
  return lhs.commTagIdx < rhs.commTagIdx;
}

// number loops of collectives call before using profiled data.
// Each loop will go thoroughly through all search space of all candidates.
#define TUNER_SEARCH_NLOOPS 5
#define PROFILE_ROUND                                                          \
  2 // Use data from the 3rd round, as it's likely more stable.

// customized context structure for internal use
struct sdcclTunerContext {
  void *bootstrap;

  int rank;
  int nranks;

  float *profilingResults;

  // configure related struct
  std::vector<struct sdcclEnvConfig> configList;
  sdcclDebugLogger_t logger = NULL;
  int envTagIdx = -1; // the index of envTag in configList
  uint32_t searchNLoops = TUNER_SEARCH_NLOOPS;

  // runtime related struct
  std::vector<int> activeCommList; // List of active communicator. Holds indices
                                   // of configList
  std::map<struct sdcclCommTag, int>
      commTagIdxMap; // map from commTag to configList index
  std::map<TunerCollCategory, uint32_t>
      collSeqMap; // record the sequence number of each collective category
  std::map<TunerCollCategory, int>
      collBestCommMap; // record the best communicator for each collective
                       // category. value is comm index in configList.
  std::map<struct TunerCommTagCounterKey, int>
      configCounterMap; // record per (collType,nBytes,configIdx) counter.

  int commConfigId = 0; // record the current communicator config id, used when
                        // tuning with FlagScale

  int bestConfigId = -1; // record the best communicator config id, used when
                         // tuning with FlagScale

  // for flagscale tuning
  bool tunerCommMatchingDone = false;
  int lastFlagscaleConfigId = -1;

  // timer
  sdcclTimer<TunerProfileKey> timer;
};

bool operator<(const struct sdcclCommTag &lhs,
               const struct sdcclCommTag &rhs) {
  return strcmp(lhs.tag, rhs.tag) < 0;
}

bool operator==(const struct sdcclCommTag &lhs,
                const struct sdcclCommTag &rhs) {
  return strcmp(lhs.tag, rhs.tag) == 0;
}

// A helper function set envs filtered by envType mask
static sdcclResult_t setEnvConfig(const struct sdcclEnvConfig &cfg,
                                   uint32_t mask) {
  for (int i = 0; i < cfg.envCount; i++) {
    const auto &item = cfg.envs[i];
    if (item.type & mask) {
      if (setenv(item.name, item.value, 1) != 0) {
        return sdcclInternalError;
      }
    }
  }
  return sdcclSuccess;
}

static bool needPatternMatching(struct sdcclTunerContext *ctx, int configId) {
  if (ctx->bestConfigId != -1 || configId != 0) {
    return false;
  }
  return !ctx->tunerCommMatchingDone;
}

sdcclResult_t sdcclTunerInit(size_t nRanks, size_t rank,
                               sdcclDebugLogger_t logFunction, void **context,
                               void *commState) {
  struct sdcclTunerContext *ctx = new struct sdcclTunerContext;
  ctx->bootstrap = commState;
  ctx->rank = rank;
  ctx->nranks = nRanks;
  SDCCLCHECK(generateCandidate(ctx->configList));
  INFO(SDCCL_TUNING, "Candidate number: %ld.", ctx->configList.size());
  ctx->logger = logFunction;
  *context = ctx;

  // Initialize commTagIdxMap and activeCommList
  for (size_t i = 0; i < ctx->configList.size(); ++i) {
    const auto &cfg = ctx->configList[i];
    ctx->commTagIdxMap[cfg.commTag] = i;
    ctx->activeCommList.push_back(i);
  }

  // Whether comm tag specified by environment variable
  const char *tagEnv = sdcclGetEnv("SDCCL_USE_COMM_TAG");
  if (tagEnv != nullptr) {
    struct sdcclCommTag envTag;
    snprintf(envTag.tag, SDCCL_COMM_TAG_MAX_LENGTH, "%s", tagEnv);
    auto it = ctx->commTagIdxMap.find(envTag);
    if (it == ctx->commTagIdxMap.end()) {
      WARN("Communicator tag %s set by environment not found in config list.",
           envTag.tag);
      return sdcclInvalidArgument;
    }
    ctx->envTagIdx = it->second;
    INFO(SDCCL_ENV | SDCCL_TUNING,
         "Communicator tag set by environment to %s.", envTag.tag);
  }

  // Whether to change search nloops by environment variable
  const char *nLoopsEnv = sdcclGetEnv("SDCCL_TUNER_SEARCH_NLOOPS");
  if (nLoopsEnv != nullptr) {
    try {
      int val = std::stoi(nLoopsEnv);
      if (val >= 5) {
        ctx->searchNLoops = val;
        INFO(SDCCL_ENV | SDCCL_TUNING,
             "Tuner search nloops set by environment to %d.",
             ctx->searchNLoops);
      }
    } catch (const std::exception &e) {
      WARN("Invalid value for SDCCL_TUNER_SEARCH_NLOOPS: %s. Using default.",
           nLoopsEnv);
    }
  }

  // initialize profilingResults pointer
  SDCCLCHECK(sdcclCalloc(&ctx->profilingResults, nRanks));
  // start timer
  ctx->timer.start();
  return sdcclSuccess;
}

sdcclResult_t sdcclTunerGetCandidateNumber(void *context,
                                             uint32_t *nCandidates) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  *nCandidates = ctx->configList.size();
  return sdcclSuccess;
}

sdcclResult_t sdcclTunerSetCandidate(void *context, uint32_t index,
                                       struct sdcclCommTag *commTag) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  if (index >= ctx->configList.size()) {
    WARN("invalid index, index %u must less than config size %zu.", index,
         ctx->configList.size());
    return sdcclInvalidArgument;
  }
  // Set env for that communicator index
  const auto &curCfg = ctx->configList[index];
  SDCCLCHECK(setEnvConfig(curCfg, SDCCL_ENV_TYPE_CREATION));
  *commTag = curCfg.commTag;
  return sdcclSuccess;
}

// Given a startup phase seqId, get the corresponding communicator index in
// configList. Logic must be consistent with getSeqIdForCommIdx.
static int getCommIdxFromSeqId(const struct sdcclTunerContext *ctx,
                               uint32_t seqId) {
  if (ctx->activeCommList.size() == 0) {
    return -1;
  }
  return ctx->activeCommList[seqId / ctx->searchNLoops];
}

// Given a communicator index in configList, get the corresponding startup phase
// seqId for specific round. Logic must be consistent with getCommIdxFromSeqId.
static int getSeqIdForCommIdx(const struct sdcclTunerContext *ctx, int commIdx,
                              uint32_t round) {
  int seqId = 0;
  bool found = false;
  for (const auto &idx : ctx->activeCommList) {
    if (idx != commIdx) {
      seqId++;
    } else {
      found = true;
      break;
    }
  }
  return (found ? (seqId * ctx->searchNLoops) + round : -1);
}

// add a small factor to avoid switching between two close communicators caused
// by measurement noise
const float tunerProfileFactor = 0.95f;

// Helper function to find the best communicator for a collective category based
// on profiling data Strategy: For each active communicator, check if we have
// profiling data for that collective category. If yes, use that data to
// calculate the time for that collective category. If no, skip that
// communicator. Finally, select the communicator with the minimum time as the
// best communicator.
static sdcclResult_t findBestComm(struct sdcclTunerContext *ctx,
                                   const struct TunerCollCategory &cat) {
  int bestCommIdx = -1; // index of best communicator in configList
  float minTime = FLT_MAX;
  // calculate the best communicator based on profiling data
  // get the profiling data for the 2/3th round for comparison
  // i.e. if searchNLoops = 5, this would be round 2
  const uint32_t profileDataRound = ctx->searchNLoops * 2 / 3 - 1;
  for (const auto &idx : ctx->activeCommList) {
    int seqId = getSeqIdForCommIdx(
        ctx, idx,
        std::min(profileDataRound,
                 static_cast<uint32_t>(ctx->searchNLoops - 1)));
    TunerProfileKey profileKey(cat.nBytes, static_cast<uint32_t>(cat.collType),
                               static_cast<uint32_t>(seqId), idx);
    struct sdcclRecordKey<TunerProfileKey> rkey(profileKey);
    float duration = ctx->timer.getRecord(rkey, true);

    if (duration <= 0) {
      // no profiling data for this communicator and collective category
      WARN("No profiling data for (commId=%d,coll=%d,size=%zu,seq=%u).", idx,
           cat.collType, cat.nBytes, seqId);
      continue;
    }

    memcpy(ctx->profilingResults + ctx->rank, &duration, sizeof(float));
    // get average duration across all ranks
    SDCCLCHECK(bootstrapAllGather(
        ctx->bootstrap, (void *)ctx->profilingResults, sizeof(float)));
    SDCCLCHECK(bootstrapBarrier(ctx->bootstrap, ctx->rank, ctx->nranks, 0));
    duration = 0.0f;
    for (int i = 0; i < ctx->nranks; ++i) {
      duration += ctx->profilingResults[i];
    }
    duration /= ctx->nranks;

    INFO(SDCCL_TUNING,
         "Profiling data for (commId=%d,coll=%d,size=%zu,seq=%u) is %.3fms.",
         idx, cat.collType, cat.nBytes, seqId, duration);

    if (duration < minTime * tunerProfileFactor) {
      minTime = duration;
      bestCommIdx = idx;
    }
  }
  if (bestCommIdx == -1) {
    WARN("No best communicator found for (coll=%d, size=%zu).", cat.collType,
         cat.nBytes);
    return sdcclInternalError;
  }

  const sdcclEnvConfig &bestConfig = ctx->configList[bestCommIdx];
  std::stringstream msg;
  msg << "Best Envs: ";
  for (int i = 0; i < bestConfig.envCount; i++) {
    msg << bestConfig.envs[i].name << "=" << bestConfig.envs[i].value
        << "(default=" << bestConfig.envs[i].defaultValue << ")";
    if (i < bestConfig.envCount - 1)
      msg << "  ";
  }
  // Output the best config
  INFO(SDCCL_TUNING, "Find (coll=%d,size=%zu) best CommId=%d. %s",
       cat.collType, cat.nBytes, bestCommIdx, msg.str().c_str());

  ctx->collBestCommMap[cat] = bestCommIdx;
  return sdcclSuccess;
}

sdcclResult_t
sdcclCreateOrReplaceHomoComm(sdcclComm_t *comm,
                              struct sdcclTunerContext *ctx, uint32_t seqId,
                              const struct TunerCollCategory &collCat,
                              sdcclStream_t stream, bool createBest) {

  // If a communicator has already been created for the corresponding collCat in
  // comm->homoCommMap, delete it before creating a new one to ensure that each
  // collCat has only one communicator.
  auto it = (*comm)->homoCommMap.find(collCat);
  if (it != (*comm)->homoCommMap.end()) {
    // make sure all operations on the comm stream is done before destroying
    // communicator
    deviceAdaptor->streamSynchronize(stream);
    // Destroy Comm of collCat
    SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commDestroy(it->second));
    // Remove entry from map
    (*comm)->homoCommMap.erase(it);
  }

  uint32_t nConfigs = 0;
  uint32_t idx = getCommIdxFromSeqId(ctx, seqId);
  struct sdcclCommTag tag = {""};
  SDCCLCHECK(sdcclTunerSetCandidate((*comm)->tunerContext, idx, &tag));
  SDCCLCHECK(
      (*comm)->tuner->getCandidateNumber((*comm)->tunerContext, &nConfigs));
  if (createBest) {
    INFO(SDCCL_INIT | SDCCL_TUNING,
         "create the communicator of the best Config (CommId = %d)",
         ctx->collBestCommMap[collCat]);
  } else {
    INFO(SDCCL_INIT | SDCCL_TUNING,
         "start to prepare communicator tag=%s(%u/%u)", tag.tag, idx, nConfigs);
  }

  sdcclInnerComm_t innerComm = NULL;
  SDCCLCHECK(sdcclHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                 (struct bootstrapState *)(ctx->bootstrap),
                                 *comm, &innerComm));
  // Store new communicator of collCat into homoCommMap
  (*comm)->homoCommMap[collCat] = innerComm;
  // For backward compatible, also assign homoComm field.
  (*comm)->homoComm = innerComm;
  return sdcclSuccess;
}

// Communicator selection logic:
// 1) Honor environment override when ctx->envTagIdx is set.
// 2) Otherwise, for the initial searchNLoops * activeCommCount invocations of
//    each {collType, nBytes}, cycle through ctx->activeCommList via seqId
//    (tuning phase).
// 3) After the tuning window, rely on the best communicator recorded in
//    ctx->collBestCommMap (populated via profiling). If no best entry exists,
//    return sdcclInternalError.
sdcclResult_t sdcclTunerGetCollInfo(void *context, sdcclCommOp_t collType,
                                      size_t nBytes, int numPipeOps,
                                      float **collCostTable, int regBuff,
                                      struct sdcclCommTag *commTag,
                                      sdcclComm_t *comm,
                                      sdcclStream_t stream) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  // Use env comm tag when possible.
  if (ctx->envTagIdx != -1) {
    SDCCLCHECK(
        setEnvConfig(ctx->configList[ctx->envTagIdx], SDCCL_ENV_TYPE_COLL));
    *commTag = ctx->configList[ctx->envTagIdx].commTag;
    INFO(SDCCL_TUNING, "Use Communicator tag %s set by environment.",
         commTag->tag);
    return sdcclSuccess;
  }

  // get a seqId for {collType, nBytes}
  struct TunerCollCategory collCat = {collType, nBytes};
  auto it = ctx->collSeqMap.find(collCat);
  uint32_t seqId = 0;
  if (it == ctx->collSeqMap.end()) {
    ctx->collSeqMap[collCat] = 0;
  } else {
    it->second++;
    seqId = it->second;
  }

  if (seqId < ctx->searchNLoops * ctx->activeCommList.size()) {

    // Every {collType, nBytes, commTagIdx} will be profiled searchNLoops times.
    int cfgIdx = getCommIdxFromSeqId(ctx, seqId);
    if (cfgIdx == -1) {
      WARN("No active communicator found for startup phase seqId=%u.", seqId);
      return sdcclInternalError;
    }
    if ((*comm)->isUseSingleTunerComm) {
      TunerCommTagCounterKey key{nBytes, static_cast<uint32_t>(collType),
                                 static_cast<uint32_t>(cfgIdx)};
      auto cit = ctx->configCounterMap.find(key);
      if (cit == ctx->configCounterMap.end()) {
        // create a new communicator and destroy old communicator
        SDCCLCHECK(sdcclCreateOrReplaceHomoComm(comm, ctx, seqId, collCat,
                                                  stream, false));
        (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
        ctx->configCounterMap[key] = 1;
      } else {
        // use old communicator
        (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
        ctx->configCounterMap[key]++;
      }
      const auto &cfg = ctx->configList[cfgIdx];
      *commTag = cfg.commTag;
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
    } else {
      const auto &cfg = ctx->configList[cfgIdx];
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
      *commTag = cfg.commTag;
      INFO(SDCCL_TUNING, "Use Communicator tag %s in startup phase seqId=%u.",
           commTag->tag, seqId);
      const auto it = (*comm)->commMap.find(*commTag);
      if (it == (*comm)->commMap.end()) {
        WARN("communicator %s was not initialized.", commTag->tag);
        return sdcclInternalError;
      }
      (*comm)->tunerInnerComm = it->second;
    }
    return sdcclSuccess;
  }

  // Select a communicator from active communicators based on profiling data
  // after searchNLoops * activeCommCount collectives. If we do not have a best
  // communicator recorded for this collective category, find it.
  if ((*comm)->homoBestCommMap[collCat] == nullptr) {
    // Find the best config
    SDCCLCHECK(findBestComm(ctx, collCat));
    // Check whether the optimal config has been found; if not, return an error.
    auto it2 = ctx->collBestCommMap.find(collCat);
    if (it2 == ctx->collBestCommMap.end()) {
      WARN("No best communicator found for collective type %d with size %zu.",
           collType, nBytes);
      return sdcclInternalError;
    }
    // If the optimal config has been found, create a communicator of best
    // config
    if ((*comm)->isUseSingleTunerComm) {
      const uint32_t profileDataRound = PROFILE_ROUND;
      uint32_t bestSeqId = getSeqIdForCommIdx(
          ctx, it2->second,
          std::min(profileDataRound,
                   static_cast<uint32_t>(ctx->searchNLoops - 1)));
      SDCCLCHECK(sdcclCreateOrReplaceHomoComm(comm, ctx, bestSeqId, collCat,
                                                stream, true));
      auto &cfg = ctx->configList[it2->second];
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
      *commTag = cfg.commTag;
      (*comm)->tunerInnerComm = (*comm)->homoCommMap[collCat];
      // Store the best communicator of collCat into homoBestCommMap
      (*comm)->homoBestCommMap[collCat] = (*comm)->homoCommMap[collCat];
    } else {
      auto &cfg = ctx->configList[it2->second];
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
      *commTag = cfg.commTag;
      INFO(SDCCL_TUNING, "Use Communicator tag %s based on profile data.",
           commTag->tag);
      const auto it = (*comm)->commMap.find(*commTag);
      if (it == (*comm)->commMap.end()) {
        WARN("communicator %s was not initialized.", commTag->tag);
        return sdcclInternalError;
      }
      (*comm)->tunerInnerComm = it->second;
      (*comm)->homoBestCommMap[collCat] = it->second;
    }
  } else {
    // The best communicator has been created
    // get it in collBestCommMap directly
    auto it2 = ctx->collBestCommMap.find(collCat);
    if (it2 == ctx->collBestCommMap.end()) {
      WARN("No best communicator found for collective type %d with size %zu.",
           collType, nBytes);
      return sdcclInternalError;
    }
    auto &cfg = ctx->configList[it2->second];
    SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
    *commTag = cfg.commTag;
    (*comm)->tunerInnerComm = (*comm)->homoBestCommMap[collCat];
    INFO(SDCCL_TUNING,
         "Use Communicator tag %s based on profile data, seqId=%d.",
         commTag->tag, seqId);
  }
  return sdcclSuccess;
}

// Handle flagscale tuning logic
// This function processes flagscale tuning configuration:
// 1. Matches the collective operation and size against tuneObjects to
// determine
//    if this comm needs tuning (sets comm->isTunningComm)
// 2. Reads SDCCL_TUNER_CONFIG_ID and SDCCL_TUNER_BEST_CONFIG_ID from
// environment
// 3. Switches communicator config if configId increments by 1
// Returns sdcclSuccess on success, sdcclInternalError if configId is invalid
sdcclResult_t sdcclHandleFlagscaleTuning(void *context, sdcclComm_t comm,
                                           sdcclCommOp_t commOp,
                                           size_t nBytes) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  // Execute matching only once when tuneObjects has values
  const char *configIdEnv = getenv("SDCCL_TUNER_CONFIG_ID");
  const int configId = (configIdEnv != NULL) ? atoi(configIdEnv) : -1;
  if (configId == -1) {
    // reset isTunningComm flag in case we are sequentially tuning multiple
    // communicators
    comm->isTunningComm = false;
    ctx->tunerCommMatchingDone = false;
  }
  // static bool matchingDone = false;
  if (needPatternMatching(ctx, configId)) {
    // Determine if this comm needs tuning
    FlagScaleConfig config = readFlagScaleJson();
    if (!config.tuneObjects.empty()) {
      bool isTuningComm = false;
      sdcclCommOp_t currentCommOp = commOp;
      INFO(SDCCL_TUNING, "sdcclTuner finding match for commOp=%d, nBytes=%zu",
           currentCommOp, nBytes);
      for (size_t idx = 0; idx < config.tuneObjects.size(); ++idx) {
        const TuneObject &item = config.tuneObjects[idx];
        std::string opStr = getTuneObjectCommOp(item);
        sdcclCommOp_t tuneCommOp = commOpStringToEnum(opStr);
        if (tuneCommOp == currentCommOp && item.nBytes == (int64_t)nBytes) {
          isTuningComm = true;
          break;
        }
      }
      comm->isTunningComm = isTuningComm;
      ctx->tunerCommMatchingDone = true;
    }
  }
  // If not tuning this comm, directly return
  if (!comm->isTunningComm) {
    return sdcclSuccess;
  }

  // Need tuning this comm
  // Handle configId logic
  // static int lastFlagscaleConfigId = -1;
  const char *bestConfigIdEnv = getenv("SDCCL_TUNER_BEST_CONFIG_ID");
  const int bestConfigId =
      (bestConfigIdEnv != NULL) ? atoi(bestConfigIdEnv) : -1;

  // if configId is -1, use the default communicator config
  if (configId == -1) {
    return sdcclSuccess;
  }

  // if configId is greater than lastFlagscaleConfigId by 1,
  // switch to the new communicator config
  if (configId - ctx->lastFlagscaleConfigId == 1) {
    ctx->lastFlagscaleConfigId = configId;
    INFO(SDCCL_TUNING, "call switchCommConfig with configId=%d", configId);
    SDCCLCHECK(
        comm->tuner->switchCommConfig(comm->tunerContext, &comm, bestConfigId));
    return sdcclSuccess;
  }

  // if configId is equal to lastFlagscaleConfigId, don't switch communicator
  // config
  if (configId - ctx->lastFlagscaleConfigId == 0) {
    return sdcclSuccess; // Should call call() and return
  }

  // Invalid configId
  WARN("configId=%d is invalid", configId);
  return sdcclInternalError;
}

sdcclResult_t sdcclTunerSwitchCommConfig(void *context, sdcclComm_t *comm,
                                           int bestConfigId) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);

  if (ctx->commConfigId < ctx->configList.size()) {
    if (bestConfigId != -1) {
      WARN("bestConfigId=%d is not -1, but commConfigId=%d is less than "
           "configList.size()=%zu",
           bestConfigId, ctx->commConfigId, ctx->configList.size());
      return sdcclInternalError;
    }

    const auto &cfg = ctx->configList[ctx->commConfigId];
    if ((*comm)->isUseSingleTunerComm) {
      auto inner = (*comm)->tunerInnerComm;
      if (inner == nullptr) {
        WARN("comm->tunerInnerComm is null");
        return sdcclInternalError;
      }

      SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commDestroy(inner));
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_CREATION));
      sdcclInnerComm_t newInner = NULL;
      SDCCLCHECK(sdcclHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                     (struct bootstrapState *)(ctx->bootstrap),
                                     *comm, &newInner));
      (*comm)->tunerInnerComm = newInner;
      (*comm)->homoComm = newInner;
    } else {
      const struct sdcclCommTag *commTag = &cfg.commTag;
      INFO(SDCCL_TUNING, "Use Communicator tag %s based on profile data.",
           commTag->tag);
      const auto it = (*comm)->commMap.find(*commTag);
      if (it == (*comm)->commMap.end()) {
        WARN("communicator %s was not initialized.", commTag->tag);
        return sdcclInternalError;
      }
      (*comm)->tunerInnerComm = it->second;
    }
    SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
    ctx->commConfigId += 1;
    // if all communicator configurations have been tested, set the environment
    // variable SDCCL_TUNER_DONE to 1
    if (ctx->commConfigId >= ctx->configList.size()) {
      setenv("SDCCL_TUNER_DONE", "1", 1);
      INFO(SDCCL_TUNING,
           "Tuning completed: all %zu communicator configurations have been "
           "tested. ENV SDCCL_TUNER_DONE=%s",
           ctx->configList.size(), getenv("SDCCL_TUNER_DONE"));
    }
    return sdcclSuccess;
  }

  if (bestConfigId != -1 && ctx->bestConfigId == -1) {
    ctx->bestConfigId = bestConfigId;
    const auto &cfg = ctx->configList[ctx->bestConfigId];
    if ((*comm)->isUseSingleTunerComm) {
      auto inner = (*comm)->tunerInnerComm;
      if (inner == nullptr) {
        WARN("comm->tunerInnerComm is null");
        return sdcclInternalError;
      }

      SDCCLCHECK(cclAdaptors[sdcclCCLAdaptorDevice]->commDestroy(inner));
      SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_CREATION));
      sdcclInnerComm_t newInner = NULL;
      SDCCLCHECK(sdcclHomoCommInit((*comm)->commId, (*comm)->uniqueIdData,
                                     (struct bootstrapState *)(ctx->bootstrap),
                                     *comm, &newInner));
      (*comm)->tunerInnerComm = newInner;
      (*comm)->homoComm = newInner;
    } else {

      const struct sdcclCommTag *commTag = &cfg.commTag;
      INFO(SDCCL_TUNING, "Use Communicator tag %s based on profile data.",
           commTag->tag);
      const auto it = (*comm)->commMap.find(*commTag);
      if (it == (*comm)->commMap.end()) {
        WARN("communicator %s was not initialized.", commTag->tag);
        return sdcclInternalError;
      }
      (*comm)->tunerInnerComm = it->second;
    }
    SDCCLCHECK(setEnvConfig(cfg, SDCCL_ENV_TYPE_COLL));
    std::stringstream msg;
    msg << "Best Envs: ";
    for (int i = 0; i < cfg.envCount; i++) {
      msg << cfg.envs[i].name << "=" << cfg.envs[i].value
          << "(default=" << cfg.envs[i].defaultValue << ")";
      if (i < cfg.envCount - 1) {
        msg << "  ";
      }
    }
    INFO(SDCCL_TUNING, "switch to the best config, configId=%d. %s",
         ctx->bestConfigId, msg.str().c_str());
    return sdcclSuccess;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTunerStartProfiling(void *context, sdcclCommOp_t collType,
                                         size_t nBytes, sdcclStream_t stream,
                                         const struct sdcclCommTag *commTag,
                                         struct sdcclProfileKey *key) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  struct TunerCollCategory collCat = {collType, nBytes};

  auto it = ctx->collSeqMap.find(collCat);
  uint32_t seqId = 0;
  if (it != ctx->collSeqMap.end()) {
    seqId = it->second;
  } else {
    WARN("Collective category (coll=%d,size=%zu) not found in collSeqMap.",
         collType, nBytes);
    return sdcclInvalidArgument;
  }

  auto it2 = ctx->commTagIdxMap.find(*commTag);
  if (it2 == ctx->commTagIdxMap.end()) {
    WARN("Communicator tag %s not found in config list.", commTag->tag);
    return sdcclInvalidArgument;
  }
  uint32_t commTagIdx = it2->second;

  // Always generate the key, even if we do not do profiling for this
  // collective.
  TunerProfileKey profileKey(nBytes, static_cast<uint32_t>(collType), seqId,
                             commTagIdx);
  /*
  INFO(SDCCL_TUNING, "Enter StartProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  *key = profileKey;

  // do profile only for startup collectives
  if (seqId < ctx->searchNLoops * ctx->activeCommList.size()) {
    struct sdcclRecordKey<TunerProfileKey> rkey(profileKey);
    SDCCLCHECK(ctx->timer.begin(rkey, stream));
  }
  /*
  INFO(SDCCL_TUNING, "Leave StartProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  return sdcclSuccess;
}

sdcclResult_t sdcclTunerStopProfiling(void *context,
                                        const struct sdcclProfileKey *key) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  TunerProfileKey profileKey(*key);
  /*
  INFO(SDCCL_TUNING, "Enter StopProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  // do profile only for startup collectives
  if (profileKey.seqId < ctx->searchNLoops * ctx->activeCommList.size()) {
    struct sdcclRecordKey<TunerProfileKey> rkey(profileKey);
    SDCCLCHECK(ctx->timer.end(rkey));
  }
  /*
  INFO(SDCCL_TUNING, "Leave StopProfiling for
  (commId=%d,coll=%d,size=%zu,seq=%u).", profileKey.commTagIdx,
  profileKey.collType, profileKey.nBytes, profileKey.seqId);
  */
  return sdcclSuccess;
}

sdcclResult_t sdcclTunerDestroy(void *context) {
  struct sdcclTunerContext *ctx =
      static_cast<struct sdcclTunerContext *>(context);
  // INFO(SDCCL_TUNING, "Enter sdcclTunerDestroy.");

  // stop timer
  ctx->timer.stop();
  free(ctx->profilingResults);
  delete ctx;
  return sdcclSuccess;
}

sdcclTuner_t internalTuner = {"internal tuner",
                               sdcclTunerInit,
                               sdcclTunerGetCandidateNumber,
                               sdcclTunerSetCandidate,
                               sdcclTunerGetCollInfo,
                               sdcclTunerStartProfiling,
                               sdcclTunerStopProfiling,
                               sdcclTunerDestroy,
                               sdcclCreateOrReplaceHomoComm,
                               sdcclTunerSwitchCommConfig,
                               sdcclHandleFlagscaleTuning};
