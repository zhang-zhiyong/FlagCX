/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "bootstrap.h"
#include "check.h"
#include "sdccl.h"
#include "sdccl_hetero.h"
#include "group.h"
#include "net.h"
#include "p2p.h"
#include "topo.h"
#include "transport.h"
#include "type.h"
#include "utils.h"
#include <algorithm>
#include <string.h>

static bool initialized = false;
pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;

struct sdcclCommInitRankAsyncJob {
  struct sdcclAsyncJob base;
  struct sdcclHeteroComm *comm;
  struct sdcclHeteroComm **newcomm;
  int cudaDev;
  // For sdcclCommInitRank
  int nranks, myrank;
  sdcclUniqueId commId;
  // for sdcclCommSplit
  struct sdcclHeteroComm *parent;
  int color, key;
};

sdcclResult_t sdcclHeteroGetVersion(int *version) {
  if (version == NULL)
    return sdcclInvalidArgument;
  *version = SDCCL_VERSION(1, 0, 0);
  return sdcclSuccess;
}

static sdcclResult_t sdcclInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE))
    return sdcclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    // SDCCLCHECK(loadDeviceSymbol());
    SDCCLCHECK(bootstrapNetInit());
    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&initLock);
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroGetUniqueId(sdcclUniqueId *out) {
  SDCCLCHECK(sdcclInit());
  sdcclResult_t res =
      bootstrapGetUniqueId((struct sdcclBootstrapHandle *)out);
  return res;
}

static uint64_t hashUniqueId(sdcclUniqueId const &id) {
  char const *bytes = (char const *)&id;
  uint64_t h = 0xdeadbeef;
  for (int i = 0; i < (int)sizeof(sdcclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

static sdcclResult_t fillPeerInfo(sdcclHeteroComm_t comm,
                                   struct sdcclPeerInfo *info,
                                   uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->hostHash = getHostHash() + commHash;
  info->pidHash = getPidHash() + commHash;
  info->busId = comm->busId;
  info->comm = comm;

  return sdcclSuccess;
}

static sdcclResult_t initTransportsRank(sdcclHeteroComm_t comm,
                                         sdcclHeteroComm_t parent) {
  INFO(SDCCL_INIT, "inside initTransportsRank");
  sdcclResult_t ret = sdcclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;

  // fill peer info
  SDCCLCHECKGOTO(sdcclCalloc(&comm->peerInfo, nranks), ret, fail);
  INFO(SDCCL_INIT, "start fillPeerInfo");
  SDCCLCHECKGOTO(fillPeerInfo(comm, comm->peerInfo + rank, comm->commHash),
                  ret, fail);
  // Question: where did we initialize comm->bootstrap?
  INFO(SDCCL_INIT, "start bootstrapAllGather for peerInfo");
  SDCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, (void *)comm->peerInfo,
                                     sizeof(struct sdcclPeerInfo)),
                  ret, fail);
  SDCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, rank, nranks, 0), ret,
                  fail);

  // check for duplicate GPUs
  INFO(SDCCL_INIT, "start check for duplicate GPUs");
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash)
      nNodes++;
    if ((i != rank) &&
        (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
        (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device "
           "%lx",
           rank, i, comm->peerInfo[rank].busId);
      ret = sdcclInvalidUsage;
      goto fail;
    }
  }

  {
    SDCCLCHECKGOTO(sdcclCalloc(&comm->rankToNode, nranks), ret, fail);
    int *nodesFirstRank = NULL;
    SDCCLCHECKGOTO(sdcclCalloc(&nodesFirstRank, nranks), ret, fail);
    comm->nNodes = 0;
    for (int r = 0; r < nranks; r++) {
      int node;
      for (node = 0; node < comm->nNodes; node++) {
        if (comm->peerInfo[nodesFirstRank[node]].hostHash ==
            comm->peerInfo[r].hostHash)
          break;
      }
      if (node == comm->nNodes) {
        nodesFirstRank[comm->nNodes] = r;
        comm->nNodes++;
      }
      comm->rankToNode[r] = node;
    }

    // Allocate nodeRanks and count localRanks per node
    SDCCLCHECKGOTO(sdcclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
    SDCCLCHECKGOTO(sdcclCalloc(&comm->rankToLocalRank, nranks), ret, fail);
    for (int r = 0; r < nranks; r++) {
      int node = comm->rankToNode[r];
      comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
      comm->nodeRanks[node].localRanks++;
    }

    // Allocate localRankToRank arrays and find maxLocalRanks
    comm->maxLocalRanks = 0;
    for (int n = 0; n < comm->nNodes; n++) {
      SDCCLCHECKGOTO(sdcclCalloc(&comm->nodeRanks[n].localRankToRank,
                                   comm->nodeRanks[n].localRanks),
                      ret, fail);
      comm->maxLocalRanks =
          std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
      comm->nodeRanks[n].localRanks = 0; // Reset for filling
    }

    // Fill localRankToRank arrays
    for (int r = 0; r < nranks; r++) {
      int node = comm->rankToNode[r];
      comm->nodeRanks[node]
          .localRankToRank[comm->nodeRanks[node].localRanks++] = r;
    }

    // Set local info for this rank
    comm->node = comm->rankToNode[rank];
    comm->localRank = comm->rankToLocalRank[rank];
    comm->localRanks = comm->nodeRanks[comm->node].localRanks;
    comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;

    // Build p2pSchedule with two-level scheduling (like NCCL)
    int node = comm->node;
    int local = comm->localRank;
    int nLocals = comm->maxLocalRanks;
    struct sdcclNodeRanks *nodeRanks = comm->nodeRanks;
    bool flat = false;
    for (int n = 0; n < comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks != nLocals) {
        flat = true;
        comm->nNodes = 1;
        node = 0;
        nLocals = nranks;
        local = rank;
        break;
      }
    }
    int nNodesPow2 = pow2Up(comm->nNodes);
    int nLocalsPow2 = pow2Up(nLocals);
    uint32_t nodeRound = 0;
    uint32_t nodeDelta = 0;
    int round = 0;
    do {
      if ((int)nodeDelta < comm->nNodes) { // Filter nonsensical node deltas
        int sendNode = (node + nodeDelta) % comm->nNodes;
        int recvNode = (node - nodeDelta + comm->nNodes) % comm->nNodes;
        uint32_t localRound = 0;
        uint32_t localDelta = 0;
        do {
          if ((int)localDelta < nLocals) { // Filter nonsensical local deltas
            int sendLocal = (local + localDelta) % nLocals;
            int recvLocal = (local - localDelta + nLocals) % nLocals;
            comm->p2pSchedule[round].sendRank =
                flat ? sendLocal
                     : nodeRanks[sendNode].localRankToRank[sendLocal];
            comm->p2pSchedule[round].recvRank =
                flat ? recvLocal
                     : nodeRanks[recvNode].localRankToRank[recvLocal];
            round += 1;
          }
          localRound += 1;
          localDelta =
              (localDelta + localRound) & (nLocalsPow2 - 1); // Quadratic update
        } while (localRound != (uint32_t)nLocalsPow2);
      }
      nodeRound += 1;
      nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1);
    } while (nodeRound != (uint32_t)nNodesPow2);

    if (round != nranks) {
      WARN("P2p schedule creation has bugs: round=%d nranks=%d", round, nranks);
      ret = sdcclInternalError;
      free(nodesFirstRank);
      goto fail;
    }
    free(nodesFirstRank);
  }

  if (!sdcclParamTopoDetectionDisable()) {
    INFO(SDCCL_INIT, "start sdcclTopoGetServerTopo");
    SDCCLCHECKGOTO(sdcclTopoGetServerTopo(comm, &comm->topoServer), ret,
                    fail);
    SDCCLCHECKGOTO(sdcclTopoComputePaths(comm->topoServer, comm), ret, fail);
    if (comm->rank == 0) {
      SDCCLCHECK(sdcclTopoPrint(comm->topoServer));
    }
    INFO(SDCCL_INIT, "start getting local net from gpu");
    SDCCLCHECKGOTO(
        sdcclGetLocalNetFromGpu(comm->cudaDev, &comm->netDev, comm), ret,
        fail);

    INFO(SDCCL_INIT, "start getting topoServer from other servers");
    SDCCLCHECKGOTO(sdcclGetInterServerTopo(comm, &comm->interServerTopo,
                                             comm->topoServer),
                    ret, fail);
  } else {
    INFO(SDCCL_INIT,
         "topology detection disabled by SDCCL_DISABLE_TOPO_DETECTION");
  }

  return ret;
fail:
  return sdcclInternalError;
}

SDCCL_PARAM(P2pBufferSize, "P2P_BUFFER_SIZE",
             64L * 1024 * 1024); // default value to 64MB
SDCCL_PARAM(P2pChunkSize, "P2P_CHUNK_SIZE",
             16L * 1024 * 1024); // default value to 16MB
SDCCL_PARAM(NetBufferSize, "NET_BUFFER_SIZE",
             64L * 1024 * 1024); // default value to 64MB
SDCCL_PARAM(NetChunkSize, "NET_CHUNK_SIZE",
             4L * 1024 * 1024); // default value to 4MB

static sdcclResult_t sdcclCommInitRankFunc(struct sdcclAsyncJob *job_) {
  struct sdcclCommInitRankAsyncJob *job =
      (struct sdcclCommInitRankAsyncJob *)job_;
  sdcclHeteroComm_t comm = job->comm;
  sdcclResult_t res = sdcclSuccess;

  if (!job->parent) {
    // New version of calling bootstrapInit
    struct bootstrapState *state;
    SDCCLCHECK(sdcclCalloc(&state, 1));
    state->rank = comm->rank;
    state->nranks = comm->nRanks;
    state->abortFlag = comm->abortFlag;
    comm->bootstrap = state;
    state->magic = ((struct sdcclBootstrapHandle *)&job->commId)->magic;
    comm->magic = ((struct sdcclBootstrapHandle *)&job->commId)->magic;
    SDCCLCHECKGOTO(
        bootstrapInit((struct sdcclBootstrapHandle *)&job->commId, state), res,
        fail);
  }

  if (!job->parent) {
    // Setting up proxy network
    int nranks = comm->nRanks;
    for (int i = 0; i < MAXCHANNELS; i++) {
      SDCCLCHECK(sdcclCalloc(&comm->channels[i].peers, nranks));
      for (int r = 0; r < nranks; r++)
        SDCCLCHECK(sdcclCalloc(&comm->channels[i].peers[r], nranks));
    }
    SDCCLCHECK(sdcclCalloc(&comm->connectSend, nranks));
    SDCCLCHECK(sdcclCalloc(&comm->connectRecv, nranks));
    SDCCLCHECK(sdcclCalloc(&comm->proxyState, 1));
    SDCCLCHECK(sdcclCalloc(&comm->tasks.peers, nranks));
    SDCCLCHECK(sdcclCalloc(&comm->tasks.p2pOrder, 2 * nranks));
    SDCCLCHECK(sdcclCalloc(&comm->p2pSchedule, nranks));
    // Setup mutex/cond to work inter-process
    pthread_mutexattr_t mutexAttr;
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&comm->proxyState->mutex, &mutexAttr);
    pthread_condattr_t condAttr;
    pthread_condattr_init(&condAttr);
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&comm->proxyState->cond, &condAttr);

    for (int i = 0; i < MAXCHANNELS; i++) {
      SDCCLCHECK(
          sdcclCalloc(&comm->proxyState->proxyOps[i].consPeers, nranks));
      comm->proxyState->proxyOps[i].consNextChannel =
          reinterpret_cast<struct sdcclProxyOps *>(0x1);
      comm->proxyState->proxyOps[i].prodNextChannel =
          reinterpret_cast<struct sdcclProxyOps *>(0x1);
      pthread_mutex_init(&comm->proxyState->proxyOps[i].mutex, 0);
      for (int peer = 0; peer < nranks; peer++) {
        comm->proxyState->proxyOps[i].consPeers[peer].nextPeer =
            reinterpret_cast<struct sdcclProxyOps::consPeer *>(0x1);
      }
    }

    comm->groupNext = reinterpret_cast<struct sdcclHeteroComm *>(0x1);
    comm->preconnectNext = reinterpret_cast<struct sdcclHeteroComm *>(0x1);
    comm->proxyState->nRanks = comm->nRanks;

    bool runtimeProxy = false;
    const char *runtimeEnv = sdcclGetEnv("SDCCL_RUNTIME_PROXY");
    if (runtimeEnv) {
      runtimeProxy = (std::stoi(runtimeEnv) == 1) ? true : false;
    }
    INFO(SDCCL_INIT, "Sdccl RuntimeProxy flag set to %d", runtimeProxy);
    if (!runtimeProxy) {
      SDCCLCHECK(sdcclProxyInit(comm));
    }
  }

  sdcclNetBufferSize = sdcclParamNetBufferSize();
  sdcclNetChunkSize = sdcclParamNetChunkSize();
  sdcclNetChunks =
      (sdcclNetBufferSize + sdcclNetChunkSize - 1) / sdcclNetChunkSize;
  sdcclP2pBufferSize = sdcclParamP2pBufferSize();
  sdcclP2pChunkSize = sdcclParamP2pChunkSize();
  sdcclP2pChunks =
      (sdcclP2pBufferSize + sdcclP2pChunkSize - 1) / sdcclP2pChunkSize;
  assert(sdcclNetChunks <= SDCCL_NET_MAX_STEPS);
  assert(sdcclP2pChunks <= SDCCL_P2P_MAX_STEPS);

  SDCCLCHECK(sdcclNetInit(comm));
  INFO(SDCCL_INIT, "Using network %s", comm->netAdaptor->name);
  INFO(SDCCL_INIT, "getting busId for cudaDev %d", comm->cudaDev);
  SDCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  INFO(SDCCL_INIT, "getting commHash for rank %d", comm->rank);
  comm->commHash = getHash(job->commId.internal, SDCCL_UNIQUE_ID_BYTES);
  INFO(SDCCL_INIT, "commHash for rank %d is %lu", comm->rank, comm->commHash);
  // TODO: put net init into a separate function

  INFO(SDCCL_INIT, "start initTransportsRank");
  SDCCLCHECKGOTO(initTransportsRank(comm, NULL), res, fail);

exit:
  return res;
fail:
  comm->initState = res;
  goto exit;
}

static sdcclResult_t sdcclCommInitRankDev(sdcclHeteroComm_t *newcomm,
                                            int nranks, sdcclUniqueId commId,
                                            int myrank, int cudaDev,
                                            sdcclConfig_t *config) {
  sdcclResult_t res = sdcclSuccess;
  sdcclHeteroComm_t comm = NULL;
  struct sdcclCommInitRankAsyncJob *job = NULL;
  const char *env = sdcclGetEnv("SDCCL_COMM_ID");

  if (env && myrank == 0) {
    INFO(SDCCL_ENV, "SDCCL_COMM_ID set by environment to %s", env);
    SDCCLCHECKGOTO(
        bootstrapCreateRoot((struct sdcclBootstrapHandle *)&commId, true), res,
        fail);
  }

  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = sdcclInvalidArgument;
    goto fail;
  }

  SDCCLCHECKGOTO(sdcclCalloc(&comm, 1), res, fail);
  comm->startMagic = comm->endMagic =
      SDCCL_MAGIC; // Used to detect comm corruption.
  SDCCLCHECKGOTO(sdcclCalloc((uint32_t **)&comm->abortFlagRefCount, 1), res,
                  fail);
  *comm->abortFlagRefCount = 1;
  /* start with sdcclInternalError and will be changed to sdcclSuccess if init
   * succeeds. */
  comm->initState = sdcclInternalError;
  comm->nRanks = nranks;
  comm->rank = myrank;
  comm->cudaDev = cudaDev;
  *newcomm = comm;

  SDCCLCHECKGOTO(sdcclCalloc(&job, 1), res, fail);
  job->comm = comm;
  job->nranks = nranks;
  job->commId = commId; // C++ struct assignment
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  SDCCLCHECKGOTO(sdcclCommInitRankFunc(&job->base), res, fail);
  free(job);
exit:
  return sdcclGroupErrCheck(res);
fail:
  if (comm) {
    if (comm->abortFlagRefCount)
      free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm)
    *newcomm = NULL;
  goto exit;
}

sdcclResult_t sdcclHeteroCommInitRank(sdcclHeteroComm_t *newcomm, int nranks,
                                        sdcclUniqueId commId, int myrank) {
  SDCCLCHECK(sdcclInit());
  int cudaDev = 0;
  sdcclConfig_t config;
  // sdcclGetDevice(&cudaDev);
  deviceAdaptor->getDevice(&cudaDev);
  SDCCLCHECK(
      sdcclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, &config));
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroCommCount(const sdcclHeteroComm_t comm,
                                     int *count) {
  *count = comm->nRanks;
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroCommUserRank(const sdcclHeteroComm_t comm,
                                        int *rank) {
  *rank = comm->rank;
  return sdcclSuccess;
}

sdcclResult_t sdcclHeteroCommDestroy(sdcclHeteroComm_t comm) {
  sdcclProxyDestroy(comm);
  for (int i = 0; i < MAXCHANNELS; i++) {
    for (int r = 0; r < comm->nRanks; r++) {
      free(comm->channels[i].peers[r]);
    }
    free(comm->channels[i].peers);
  }
  for (int i = 0; i < MAXCHANNELS; i++) {
    free(comm->proxyState->proxyOps[i].consPeers);
  }

  free(comm->connectSend);
  free(comm->connectRecv);
  free(comm->proxyState);
  free(comm->tasks.peers);
  free(comm->tasks.p2pOrder);
  free(comm->abortFlagRefCount);
  if (comm->topoServer) {
    sdcclTopoFree(comm->topoServer);
  }
  if (comm->interServerTopo) {
    sdcclInterServerTopoFree(comm->interServerTopo);
  }
  free(comm->peerInfo);
  free(comm);

  return sdcclSuccess;
}
