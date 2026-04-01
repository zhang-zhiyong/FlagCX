#ifndef SDCCL_TUNER_H_
#define SDCCL_TUNER_H_

#include "../adaptor/include/tuner.h"

// A category of collective operation. the minimal unit for tuning.
struct TunerCollCategory {
  sdcclCommOp_t collType;
  size_t nBytes;
};

bool operator<(const struct TunerCollCategory &lhs,
               const struct TunerCollCategory &rhs);

struct sdcclTuner {
  // Name of the tuner
  const char *name;

  // Initializes tuner states.
  // Inputs:
  //   - nRanks: number of ranks in current communicator. Each communicator
  //   initialize its own tuner.
  //   - nNodes: number of nodes in current communicator.
  //   - logFunction: a logFunction can be useful to integrate logging together
  //   with SDCCL core.
  // Outputs:
  //   - context: tuner context object
  sdcclResult_t (*init)(size_t nRanks, size_t rank,
                         sdcclDebugLogger_t logFunction, void **context,
                         void *commState);

  // Gets number of candidate communicator env settings available from this
  // tuner. Inputs:
  //   - context: tuner context object
  // Outputs:
  //   - nCandidates: number of candidate communicator
  sdcclResult_t (*getCandidateNumber)(void *context, uint32_t *nCandidates);

  // Set appropriate environment variables according to index, and return the
  // communicator tag. Note that all the env settings are set before returning
  // from this function. Only env of type SDCCL_ENV_TYPE_CREATION will be set
  // in this function. Inputs:
  //   - context: tuner context object
  //   - index: index of candidate communicator, range [0, nCandidates)
  // Outputs:
  //   - commTag: communicator tag for this particular candidate
  sdcclResult_t (*setCandidate)(void *context, uint32_t index,
                                 struct sdcclCommTag *commTag);

  // Select the best communicator candidate for this collective.
  // All the env of type SDCCL_ENV_TYPE_COLL and SDCCL_ENV_TYPE_ONETIME if
  // necessary will be set before returning from this function. Inputs:
  //   - context: tuner context object
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - numPipeOps: number of operations in the group
  //   - regBuff: If non-zero, register user buffer is used.
  // Outputs:
  //   - commTag: communicator tag, used to select the underlying communicator.
  //
  // InOut:
  //   - collCostTable: collective cost table.  the caller is responsible for
  //   allocating and
  //                    deallocating the memory
  //
  sdcclResult_t (*getCollInfo)(void *context, sdcclCommOp_t collType,
                                size_t nBytes, int numPipeOps,
                                float **collCostTable, int regBuff,
                                struct sdcclCommTag *commTag,
                                sdcclComm_t *comm, sdcclStream_t stream);

  // Start profiling for a specific collective with given parameters.
  // Inputs:
  //   - context: tuner context object
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - stream: the stream that the collective will be launched on
  //   - commTag: communicator tag
  // Outputs:
  //   - key: profiling key to pair with stopProfiling
  //
  sdcclResult_t (*startProfiling)(void *context, sdcclCommOp_t collType,
                                   size_t nBytes, sdcclStream_t stream,
                                   const struct sdcclCommTag *commTag,
                                   struct sdcclProfileKey *key);

  // Stop profiling for a specific collective with given key.
  // Inputs:
  //   - context: tuner context object
  //   - key: profiling key returned by startProfiling
  // Outputs:
  //   - None
  //
  sdcclResult_t (*stopProfiling)(void *context,
                                  const struct sdcclProfileKey *key);

  // Terminates the tuner and cleans up any resources that the tuner allocated.
  sdcclResult_t (*destroy)(void *context);

  // Create/destroy communicator
  sdcclResult_t (*createOrReplaceHomoComm)(
      sdcclComm_t *comm, struct sdcclTunerContext *ctx, uint32_t seqId,
      const struct TunerCollCategory &collCat, sdcclStream_t stream,
      bool createBest);

  // Switch communicator config
  sdcclResult_t (*switchCommConfig)(void *context, sdcclComm_t *comm,
                                     int bestConfigId);

  // Handle flagscale tuning logic
  sdcclResult_t (*handleFlagscaleTuning)(void *context, sdcclComm_t comm,
                                          sdcclCommOp_t commOp, size_t nBytes);
};

typedef struct sdcclTuner sdcclTuner_t;

bool operator<(const struct sdcclCommTag &lhs,
               const struct sdcclCommTag &rhs);
bool operator==(const struct sdcclCommTag &lhs,
                const struct sdcclCommTag &rhs);

extern sdcclTuner_t internalTuner;

// On-demand communicator lifecycle helpers implemented in sdccl/sdccl.cc
sdcclResult_t sdcclCreateHomoCommForTag(sdcclComm_t comm, uint32_t idx);
sdcclResult_t sdcclDestroyHomoCommByTag(sdcclComm_t comm, uint32_t idx);

// Switch communicator config
sdcclResult_t sdcclTunerSwitchCommConfig(void *context, sdcclComm_t *comm,
                                           int bestConfigId);

// Handle flagscale tuning logic
// Returns sdcclSuccess if should call the original function and return
// immediately, sdcclInProgress if should continue with profiling logic, or
// other error codes on failure
sdcclResult_t sdcclHandleFlagscaleTuning(void *context, sdcclComm_t comm,
                                           sdcclCommOp_t commOp,
                                           size_t nBytes);

#define SDCCLCALLWITHTUNER(call, comm, commOp, count, datatype, stream)       \
  do {                                                                         \
    size_t nBytes = count * getSdcclDataTypeSize(datatype);                   \
    if (comm->isTuningWithFlagscale) {                                         \
      SDCCLCHECK(comm->tuner->handleFlagscaleTuning(comm->tunerContext, comm, \
                                                     commOp, nBytes));         \
      SDCCLCHECK(call);                                                       \
      return sdcclSuccess;                                                    \
    }                                                                          \
    comm->tunerInnerComm = nullptr;                                            \
    struct sdcclCommTag tag = {""};                                           \
    SDCCLCHECK(comm->tuner->getCollInfo(comm->tunerContext, commOp, nBytes,   \
                                         0, NULL, 0, &tag, &comm, stream));    \
    sdcclProfileKey pkey;                                                     \
    SDCCLCHECK(comm->tuner->startProfiling(comm->tunerContext, commOp,        \
                                            nBytes, stream, &tag, &pkey));     \
    SDCCLCHECK(call);                                                         \
    SDCCLCHECK(comm->tuner->stopProfiling(comm->tunerContext, &pkey));        \
    return sdcclSuccess;                                                      \
  } while (0);

#endif // end of SDCCL_TUNER_H_
