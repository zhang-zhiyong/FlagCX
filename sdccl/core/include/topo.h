/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_TOPO_H_
#define SDCCL_TOPO_H_

#include "core.h"
#include "sdccl_device_adaptor.h"
#include "graph.h"
#include <unordered_map>
#include <unordered_set>

#define LOC_BW 5000.0
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define PCI_BW 12.0 // PCI Gen3 x16
#define QPI_BW 6.0
#define AMD_BW 16.0
#define SKL_QPI_BW 10.0
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_BW 32.0
#define ARM_BW 6.0
#define NET_BW 12.0 // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P_OVERHEAD(bw) (bw * 6 / 5)

#define SDCCL_TOPO_NODE_TYPES 7
#define APU 0
#define PCI 1
#define CCI 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
#define HBD 6
extern const char *topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_CCI 1
// Skipping 2 for PATH_NVB
#define LINK_PCI 3
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PHB
#define LINK_SYS 7
#define LINK_NET 8
extern const char *topoLinkTypeStr[];

// Local (myself)
#define PATH_LOC 0

// Connection traversing CCI link
#define PATH_CCI 1

// Connection through CCI link using an intermediate APU
#define PATH_CCB 2

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 3

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host
// Bridge)
#define PATH_PXB 4

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable
// rail-local, aggregated network send/recv operations.
#define PATH_PXN 5

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 6

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
// (e.g., QPI/UPI)
#define PATH_SYS 7

// Connection through the network
#define PATH_NET 8

// Disconnected
#define PATH_DIS 9
extern const char *topoPathTypeStr[];

struct sdcclTopoNode;
struct sdcclTopoLink {
  int type;
  float bw;
  struct sdcclTopoNode *remNode;
};
#define SDCCL_TOPO_MAX_LINKS 128
#define SDCCL_TOPO_MAX_HOPS (SDCCL_TOPO_MAX_NODES * SDCCL_TOPO_NODE_TYPES)
#define SDCCL_MAX_INTER_SERVER_HOPS                                           \
  16 // TODO: decide on a decent number for this variable
#define SDCCL_MAX_SERVER_NUM                                                  \
  16 // TODO: decide on a decent number for this variable

struct sdcclTopoPath {
  struct sdcclTopoLink *list[SDCCL_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

#define SDCCL_TOPO_CPU_INTEL_BDW 1
#define SDCCL_TOPO_CPU_INTEL_SKL 2

#define SDCCL_TOPO_UNDEF (-1)

#define SDCCL_TOPO_ID_SERVER_ID(id) (id >> 56)
#define SDCCL_TOPO_ID_LOCAL_ID(id) (id & 0x00ffffffffffffff)
#define SDCCL_TOPO_ID(serverid, localid) (((int64_t)serverid << 56) + localid)

struct sdcclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int gdrSupport;
      int vendor;
      int interServerRouteCnt;
      struct sdcclInterServerRoute *interServerRoutes[SDCCL_TOPO_MAX_NODES];
    } apu;
    struct {
      int dev; // Plugin dev number
      uint64_t guid;
      int port;
      int ip;
      float bw;
      float latency;
      int gdrSupport;
      int maxConn;
      // int64_t guid;
    } net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    } cpu;
    struct {
      uint64_t device;
    } pci;
  };
  int nlinks;
  struct sdcclTopoLink links[SDCCL_TOPO_MAX_LINKS];
  // Pre-computed paths to GPUs and NICs
  struct sdcclTopoPath *paths[SDCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct sdcclTopoNodeSet {
  int count;
  struct sdcclTopoNode nodes[SDCCL_TOPO_MAX_NODES];
};

struct sdcclTopoServer {
  int serverId;
  uint64_t hostHashes[SDCCL_TOPO_MAX_NODES];
  int nHosts;
  struct sdcclTopoNodeSet nodes[SDCCL_TOPO_NODE_TYPES];
  float maxBw;
  float totalBw;
};

struct sdcclSwitch {
  float downBw;
  float upBw;
  int upLink;
  int downLink;
  bool isTop;
};

// inter-server topo sturcture might need to be changed
struct sdcclInterServerRoute {
  int switchCount;
  struct sdcclTopoNode *localNic;
  struct sdcclTopoNode *remoteNic;
  int remoteRank;
  float interBw;
  struct sdcclSwitch switchInfos[SDCCL_MAX_INTER_SERVER_HOPS];
};

struct sdcclInterServerTopo {
  int numServers;
  struct sdcclTopoServer
      *servers; // contain topology of all servers except current server, topo
                // of current server is stored in comm->topoServer
  std::unordered_map<uint64_t, int>
      netToServerMap; // {{netGuid, serverId}, ...}
  std::unordered_map<
      uint64_t, std::unordered_map<uint64_t, struct sdcclInterServerRoute *>>
      routeMap; // {{localNetGuid, {remoteNetGuid, route}}, ...}
  char interServerTopoFile[256];
};

struct topoArgs {
  int rank;
  int nranks;
  sdcclUniqueId uniqueId;
  void *bootstrap;
};

struct flatTopoLink {
  int type;
  float bw;
  int remNodeIdx;
  int remNodeType;
};

struct flatTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int vendor;
    } apu;
    struct {
      int dev; // Plugin dev number
      int port;
      float bw;
      float latency;
      int maxConn;
      uint64_t guid;
    } net;
    struct {
      int arch;
      int vendor;
      int model;
    } cpu;
    struct {
      uint64_t device;
    } pci;
  };
  int nlinks;
  struct flatTopoLink links[SDCCL_TOPO_MAX_LINKS];
};

struct flatTopoNodeSet {
  int count;
  struct flatTopoNode nodes[SDCCL_TOPO_MAX_NODES];
};

struct flatTopoServer {
  int serverId;
  uint64_t hostHashes[SDCCL_TOPO_MAX_NODES];
  int nHosts;
  struct flatTopoNodeSet nodes[SDCCL_TOPO_NODE_TYPES];
};

struct sdcclNicDistance {
  int distance;
  uint64_t netGuid;
};

sdcclResult_t sdcclTopoGetNode(struct sdcclTopoServer *topoServer,
                                 struct sdcclTopoNode **node, int type,
                                 uint64_t id);
sdcclResult_t sdcclTopoCreateNode(struct sdcclTopoServer *topoServer,
                                    struct sdcclTopoNode **node, int type,
                                    uint64_t id);
sdcclResult_t sdcclTopoRemoveNode(struct sdcclTopoServer *topoServer,
                                    int type, int id);
sdcclResult_t sdcclTopoConnectNodes(struct sdcclTopoNode *node,
                                      struct sdcclTopoNode *remNode, int type,
                                      float bw);
sdcclResult_t sdcclTopoPrintPaths(struct sdcclTopoServer *topoServer);
sdcclResult_t sdcclTopoLoadServer(const char *xmlTopoFile,
                                    struct sdcclTopoServer *topoServer);
sdcclResult_t
sdcclTopoGetIntermediateRank(struct sdcclTopoServer *topoServer, int rank,
                              int64_t netId, int *intermediateRank);

sdcclResult_t sdcclTopoPrint(struct sdcclTopoServer *topoServer);

sdcclResult_t sdcclTopoPrintPaths(struct sdcclTopoServer *topoServer);

sdcclResult_t
sdcclGetInterServerTopo(struct sdcclHeteroComm *comm,
                         struct sdcclInterServerTopo **interServerTopo,
                         struct sdcclTopoServer *topoServer);

#define SDCCL_TOPO_XML_MAX_NODES 256
#define SDCCL_GRAPH_XML_MAX_NODES 4096
sdcclResult_t
sdcclTopoGetServerTopoFromXml(struct sdcclXml *xml,
                               struct sdcclTopoServer **topoServer,
                               uint64_t localHostHash);
sdcclResult_t sdcclTopoGetGraphFromXml(struct sdcclXmlNode *xmlGraphs,
                                         struct sdcclTopoServer *topoServer,
                                         struct sdcclTopoGraph *graph,
                                         int *nChannels);
sdcclResult_t sdcclTopoGetXmlFromGraphs(int ngraphs,
                                          struct sdcclTopoGraph **graphs,
                                          struct sdcclTopoServer *topoServer,
                                          struct sdcclXml *xml);
sdcclResult_t sdcclTopoGetXmlTopo(struct sdcclHeteroComm *comm,
                                    struct sdcclXml *xml);
sdcclResult_t sdcclTopoGetServerTopo(struct sdcclHeteroComm *comm,
                                       struct sdcclTopoServer **topoServer);

sdcclResult_t sdcclTopoGetCompCap(struct sdcclTopoServer *topoServer,
                                    int *ccMin, int *ccMax);

sdcclResult_t sdcclGetNicDistance(struct sdcclTopoServer *topoServer,
                                    int rank,
                                    struct sdcclNicDistance *distInfo);
sdcclResult_t sdcclTopoGetLocalNetNode(struct sdcclTopoServer *topoServer,
                                         int rank,
                                         struct sdcclTopoNode **netNode);

sdcclResult_t
sdcclTopoGetServerFromRank(int rank, struct sdcclInterServerTopo *interServer,
                            struct sdcclTopoServer *currServer,
                            struct sdcclTopoServer **retServer);

// static sdcclResult_t sdcclTopoIdToIndex(struct sdcclTopoServer*
// serverTopo, int type, int64_t id, int* index) {
//   *index = -1;
//   for (int i=0; i<serverTopo->nodes[type].count; i++) {
//     if (serverTopo->nodes[type].nodes[i].id == id) {
//       *index = i;
//       return sdcclSuccess;
//     }
//   }
//   return sdcclInternalError;
// }

// static sdcclResult_t sdcclTopoRankToIndex(struct sdcclTopoServer*
// serverTopo, int rank, int* index) {
//   *index = -1;
//   for (int i=0; i<serverTopo->nodes[GPU].count; i++) {
//     if (serverTopo->nodes[GPU].nodes[i].apu.rank == rank) {
//       *index = i;
//       return sdcclSuccess;
//     }
//   }
//   return sdcclInternalError;
// }

// static sdcclResult_t sdcclTopoDevToRank(struct sdcclTopoServer*
// serverTopo, int dev, int* rank) {
//   *rank = -1;
//   for (int i=0; i<serverTopo->nodes[GPU].count; i++) {
//     if (SDCCL_TOPO_ID_SERVER_ID(serverTopo->nodes[GPU].nodes[i].id) !=
//     serverTopo->serverId) continue; // Only consider GPUs on our node if
//     (serverTopo->nodes[GPU].nodes[i].apu.dev == dev) {
//       *rank = serverTopo->nodes[GPU].nodes[i].apu.rank;
//       return sdcclSuccess;
//     }
//   }
//   return sdcclInternalError;
// }

// static sdcclResult_t sdcclTopoIdToNetDev(struct sdcclTopoServer*
// serverTopo, int64_t id, int* netDev) {
//   *netDev = -1;
//   for (int i=0; i<serverTopo->nodes[NET].count; i++) {
//     if (serverTopo->nodes[NET].nodes[i].id == id) {
//       *netDev = serverTopo->nodes[NET].nodes[i].net.dev;
//       return sdcclSuccess;
//     }
//   }
//   WARN("Could not find NET with id %lx\n", id);
//   return sdcclInternalError;
// }

// // Returns NVLink bw in GB/s
// static float sdcclTopoNVLinkBw(int cudaCompCap) {
//   return
//     cudaCompCap >= 90 ? SM90_NVLINK_BW :
//     cudaCompCap == 86 ? SM86_NVLINK_BW :
//     cudaCompCap >= 80 ? SM80_NVLINK_BW :
//     cudaCompCap >= 70 ? SM70_NVLINK_BW :
//     cudaCompCap >= 60 ? SM60_NVLINK_BW :
//     SM80_NVLINK_BW;
// }

// Mirror bits
static bool isPow2(int val) { return (val & (val - 1)) == 0; }
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b = 1, mb = (pow2 >> 1); b < pow2; b <<= 1, mb >>= 1)
    if (val & b)
      mirror |= mb;
  return mirror;
}

#ifdef CREATE_DEVICE_TOPO_API
#define DEVICE_TOPO_API_EXTERN
#else
#define DEVICE_TOPO_API_EXTERN extern
#endif

// DEVICE_TOPO_API_EXTERN sdcclResult_t (*sdcclTopoGetLocalNet)(int gpu,
//                                                                char *name);

#endif
