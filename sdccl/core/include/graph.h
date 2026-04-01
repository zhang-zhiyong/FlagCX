/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_GRAPH_H_
#define SDCCL_GRAPH_H_

#include "device.h"
#include <ctype.h>
#include <limits.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

sdcclResult_t sdcclTopoCudaPath(int cudaDev, char **path);

struct sdcclTopoServer;
// Build the topology
sdcclResult_t sdcclTopoSortSystem(struct sdcclTopoServer *topoServer);
sdcclResult_t sdcclTopoPrint(struct sdcclTopoServer *topoServer);

sdcclResult_t sdcclTopoComputePaths(struct sdcclTopoServer *topoServer,
                                      struct sdcclHeteroComm *comm);
sdcclResult_t sdcclTopoTrimSystem(struct sdcclTopoServer *topoServer,
                                    struct sdcclHeteroComm *comm);
sdcclResult_t sdcclTopoComputeP2pChannels(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclTopoGetNvbGpus(struct sdcclTopoServer *topoServer,
                                    int rank, int *nranks, int **ranks);
int sdcclTopoPathAllNVLink(struct sdcclTopoServer *topoServer);

void sdcclTopoFree(struct sdcclTopoServer *topoServer);

void sdcclInterServerTopoFree(struct sdcclInterServerTopo *interServerTopo);

// Query topology
sdcclResult_t sdcclTopoGetNetDev(struct sdcclHeteroComm *comm, int rank,
                                   struct sdcclTopoGraph *graph, int channelId,
                                   int peerRank, int64_t *id, int *dev,
                                   int *proxyRank);
sdcclResult_t sdcclTopoCheckP2p(struct sdcclTopoServer *topoServer,
                                  int64_t id1, int64_t id2, int *p2p, int *read,
                                  int *intermediateRank);
sdcclResult_t sdcclTopoCheckMNNVL(struct sdcclTopoServer *topoServer,
                                    struct sdcclPeerInfo *info1,
                                    struct sdcclPeerInfo *info2, int *ret);
sdcclResult_t sdcclTopoCheckGdr(struct sdcclTopoServer *topoServer,
                                  int64_t busId, int64_t netId, int read,
                                  int *useGdr);
sdcclResult_t sdcclTopoNeedFlush(struct sdcclTopoServer *topoServer,
                                   int64_t busId, int *flush);
sdcclResult_t sdcclTopoCheckNet(struct sdcclTopoServer *topoServer,
                                  int64_t id1, int64_t id2, int *net);
int sdcclPxnDisable(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclTopoGetPxnRanks(struct sdcclHeteroComm *comm,
                                     int **intermediateRanks, int *nranks);

// Find CPU affinity
sdcclResult_t sdcclTopoGetCpuAffinity(struct sdcclTopoServer *topoServer,
                                        int rank, cpu_set_t *affinity);

#define SDCCL_TOPO_CPU_ARCH_X86 1
#define SDCCL_TOPO_CPU_ARCH_POWER 2
#define SDCCL_TOPO_CPU_ARCH_ARM 3
#define SDCCL_TOPO_CPU_VENDOR_INTEL 1
#define SDCCL_TOPO_CPU_VENDOR_AMD 2
#define SDCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define SDCCL_TOPO_CPU_TYPE_BDW 1
#define SDCCL_TOPO_CPU_TYPE_SKL 2
#define SDCCL_TOPO_CPU_TYPE_YONGFENG 1
sdcclResult_t sdcclTopoCpuType(struct sdcclTopoServer *topoServer, int *arch,
                                 int *vendor, int *model);
sdcclResult_t sdcclTopoGetGpuCount(struct sdcclTopoServer *topoServer,
                                     int *count);
sdcclResult_t sdcclTopoGetNetCount(struct sdcclTopoServer *topoServer,
                                     int *count);
sdcclResult_t sdcclTopoGetNvsCount(struct sdcclTopoServer *topoServer,
                                     int *count);
// TODO: get nearest NIC to GPU from a xml topology structure, might need to
// change function signature
sdcclResult_t sdcclGetLocalNetFromGpu(int apu, int *dev,
                                        struct sdcclHeteroComm *comm);
sdcclResult_t sdcclTopoGetLocalGpu(struct sdcclTopoServer *topoServer,
                                     int64_t netId, int *gpuIndex);
sdcclResult_t getLocalNetCountByBw(struct sdcclTopoServer *topoServer,
                                    int gpu, int *count);

#define SDCCL_TOPO_MAX_NODES 256

// Init search. Needs to be done before calling sdcclTopoCompute
sdcclResult_t sdcclTopoSearchInit(struct sdcclTopoServer *topoServer);

#define SDCCL_TOPO_PATTERN_BALANCED_TREE                                      \
  1 // Spread NIC traffic between two GPUs (Tree parent + one child on first
    // GPU, second child on second GPU)
#define SDCCL_TOPO_PATTERN_SPLIT_TREE                                         \
  2 // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree
    // children on the second GPU)
#define SDCCL_TOPO_PATTERN_TREE 3 // All NIC traffic going to/from the same GPU
#define SDCCL_TOPO_PATTERN_RING 4 // Ring
#define SDCCL_TOPO_PATTERN_NVLS 5 // NVLS+SHARP and NVLS+Tree
struct sdcclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS * SDCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS * 2];
};
sdcclResult_t sdcclTopoCompute(struct sdcclTopoServer *topoServer,
                                 struct sdcclTopoGraph *graph);

sdcclResult_t sdcclTopoPrintGraph(struct sdcclTopoServer *topoServer,
                                    struct sdcclTopoGraph *graph);
sdcclResult_t sdcclTopoDumpGraphs(struct sdcclTopoServer *topoServer,
                                    int ngraphs,
                                    struct sdcclTopoGraph **graphs);

struct sdcclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

sdcclResult_t sdcclTopoPreset(struct sdcclHeteroComm *comm,
                                struct sdcclTopoGraph **graphs,
                                struct sdcclTopoRanks *topoRanks);

sdcclResult_t sdcclTopoPostset(struct sdcclHeteroComm *comm, int *firstRanks,
                                 int *treePatterns,
                                 struct sdcclTopoRanks **allTopoRanks,
                                 int *rings, struct sdcclTopoGraph **graphs,
                                 struct sdcclHeteroComm *parent);

sdcclResult_t sdcclTopoTuneModel(struct sdcclHeteroComm *comm,
                                   int minCompCap, int maxCompCap,
                                   struct sdcclTopoGraph **graphs);
#include "info.h"
sdcclResult_t sdcclTopoGetAlgoTime(struct sdcclInfo *info, int algorithm,
                                     int protocol, int numPipeOps, float *time,
                                     bool *backup = NULL);

#endif
