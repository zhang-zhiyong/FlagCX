/*************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "net.h"
#include "topo.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct sdcclTopoNodeList {
  struct sdcclTopoNode *list[SDCCL_TOPO_MAX_NODES];
  int count;
};

static sdcclResult_t getPath(struct sdcclTopoServer *topoServer,
                              struct sdcclTopoNode *node, int t, int64_t id,
                              struct sdcclTopoPath **path) {
  for (int i = 0; i < topoServer->nodes[t].count; i++) {
    if (topoServer->nodes[t].nodes[i].id == id) {
      *path = node->paths[t] + i;
      return sdcclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return sdcclInternalError;
}

static sdcclResult_t sdcclTopoSetPaths(struct sdcclTopoNode *baseNode,
                                         struct sdcclTopoServer *topoServer) {
  if (baseNode->paths[baseNode->type] == NULL) {
    SDCCLCHECK(sdcclCalloc(baseNode->paths + baseNode->type,
                             topoServer->nodes[baseNode->type].count));
    for (int i = 0; i < topoServer->nodes[baseNode->type].count; i++)
      baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }

  // breadth-first search to set all paths to that node in the system
  struct sdcclTopoNodeList nodeList;
  struct sdcclTopoNodeList nextNodeList = {{0}, 0};
  nodeList.count = 1;
  nodeList.list[0] = baseNode;
  struct sdcclTopoPath *basePath;
  SDCCLCHECK(
      getPath(topoServer, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->bw = LOC_BW;
  basePath->type = PATH_LOC;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n = 0; n < nodeList.count; n++) {
      struct sdcclTopoNode *node = nodeList.list[n];
      struct sdcclTopoPath *path;
      SDCCLCHECK(
          getPath(topoServer, node, baseNode->type, baseNode->id, &path));
      for (int l = 0; l < node->nlinks; l++) {
        struct sdcclTopoLink *link = node->links + l;
        struct sdcclTopoNode *remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          SDCCLCHECK(sdcclCalloc(remNode->paths + baseNode->type,
                                   topoServer->nodes[baseNode->type].count));
          for (int i = 0; i < topoServer->nodes[baseNode->type].count; i++)
            remNode->paths[baseNode->type][i].type = PATH_DIS;
        }
        struct sdcclTopoPath *remPath;
        SDCCLCHECK(getPath(topoServer, remNode, baseNode->type, baseNode->id,
                            &remPath));
        float bw = std::min(path->bw, link->bw);

        // allow routing through a APU only as 1 hop (not supported)

        if ((remPath->bw == 0 || remPath->count > path->count) &&
            remPath->bw < bw) {
          // Find reverse link
          for (int l = 0; l < remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node &&
                remNode->links[l].type == link->type) {
              remPath->list[0] = remNode->links + l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to "
                 "node %d/%lx",
                 remNode->type, remNode->id, remNode->nlinks, node->type,
                 node->id);
            return sdcclInternalError;
          }
          // Copy the rest of the path
          for (int i = 0; i < path->count; i++)
            remPath->list[i + 1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->bw = bw;

          // Start with path type = link type. PATH and LINK types are supposed
          // to match. Don't consider LINK_NET as we only care about the
          // NIC->APU path.
          int type = link->type == LINK_NET ? LINK_LOC : link->type;
          // Differentiate between one and multiple PCI switches
          if (node->type == PCI && remNode->type == PCI)
            type = PATH_PXB;
          // Consider a path going through the CPU as PATH_PHB
          if (link->type == LINK_PCI &&
              (node->type == CPU || link->remNode->type == CPU))
            type = PATH_PHB;
          // Set 1 hop CCI as CCB
          // if (node->type == APU && path->type == PATH_CCI && type == PATH_CCI
          // && remPath->count > 1) type = PATH_CCB;

          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          int i;
          for (i = 0; i < nextNodeList.count; i++)
            if (nextNodeList.list[i] == remNode)
              break;
          if (i == nextNodeList.count)
            nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return sdcclSuccess;
}

// Remove/free all paths
static void sdcclTopoRemovePaths(struct sdcclTopoServer *topoServer) {
  for (int t1 = 0; t1 < SDCCL_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < topoServer->nodes[t1].count; n++) {
      struct sdcclTopoNode *node = topoServer->nodes[t1].nodes + n;
      for (int t2 = 0; t2 < SDCCL_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2])
          free(node->paths[t2]);
        node->paths[t2] = NULL;
      }
    }
  }
}

// This is a tailored version of the original one.
sdcclResult_t sdcclTopoComputePaths(struct sdcclTopoServer *topoServer,
                                      struct sdcclHeteroComm *comm) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing
  INFO(SDCCL_GRAPH, "Removing paths");
  sdcclTopoRemovePaths(topoServer);

  // Set direct paths to CPUs. We need them in many cases.
  INFO(SDCCL_GRAPH, "Setting paths to CPUs");
  for (int c = 0; c < topoServer->nodes[CPU].count; c++) {
    SDCCLCHECK(
        sdcclTopoSetPaths(topoServer->nodes[CPU].nodes + c, topoServer));
  }

  // Set direct paths to GPUs.
  INFO(SDCCL_GRAPH, "Setting paths to APUs");
  for (int g = 0; g < topoServer->nodes[APU].count; g++) {
    SDCCLCHECK(
        sdcclTopoSetPaths(topoServer->nodes[APU].nodes + g, topoServer));
  }

  // Set direct paths to NICs.
  INFO(SDCCL_GRAPH, "Setting paths to NICs");
  for (int n = 0; n < topoServer->nodes[NET].count; n++) {
    INFO(SDCCL_GRAPH, "setting paths to net node [%d]", n);
    SDCCLCHECK(
        sdcclTopoSetPaths(topoServer->nodes[NET].nodes + n, topoServer));
  }

  // TODO: Update paths for NICs (no GPU Direct, PXN, ...)
  return sdcclSuccess;
}

static void printNodePaths(struct sdcclTopoServer *topoServer,
                           struct sdcclTopoNode *node) {
  const int linesize = 1024;
  char line[linesize];
#ifdef ENABLE_TRACE
  INFO(SDCCL_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type],
       SDCCL_TOPO_ID_SERVER_ID(node->id), SDCCL_TOPO_ID_LOCAL_ID(node->id));
#else
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type],
           SDCCL_TOPO_ID_SERVER_ID(node->id),
           SDCCL_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL)
      continue;
    for (int n = 0; n < topoServer->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i = 0; i < node->paths[t][n].count; i++) {
        struct sdcclTopoLink *link = node->paths[t][n].list[i];
        struct sdcclTopoNode *remNode = link->remNode;
        snprintf(line + offset, linesize - offset, "--%s(%g)->%s/%lx-%lx",
                 topoLinkTypeStr[link->type], link->bw,
                 topoNodeTypeStr[remNode->type],
                 SDCCL_TOPO_ID_SERVER_ID(remNode->id),
                 SDCCL_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(SDCCL_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      snprintf(line + offset, linesize - offset, "%s/%lx-%lx (%d/%.1f/%s) ",
               topoNodeTypeStr[t],
               SDCCL_TOPO_ID_SERVER_ID(system->nodes[t].nodes[n].id),
               SDCCL_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id),
               node->paths[t][n].count, node->paths[t][n].bw,
               topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

sdcclResult_t sdcclTopoPrintPaths(struct sdcclTopoServer *topoServer) {
  for (int i = 0; i < topoServer->nodes[APU].count; i++) {
    printNodePaths(topoServer, topoServer->nodes[APU].nodes + i);
  }
  for (int i = 0; i < topoServer->nodes[NET].count; i++) {
    printNodePaths(topoServer, topoServer->nodes[NET].nodes + i);
  }
  return sdcclSuccess;
}

void sdcclTopoFree(struct sdcclTopoServer *topoServer) {
  sdcclTopoRemovePaths(topoServer);
  free(topoServer);
}

void sdcclInterServerTopoFree(struct sdcclInterServerTopo *interServerTopo) {
  for (int i = 0; i < interServerTopo->numServers; i++) {
    sdcclTopoRemovePaths(interServerTopo->servers + i);
  }
  free(interServerTopo->servers);
  // free interserver routes
  for (auto localRankIter = interServerTopo->routeMap.begin();
       localRankIter != interServerTopo->routeMap.end(); ++localRankIter) {
    auto remoteRoutes = localRankIter->second;
    for (auto remoteRankIter = remoteRoutes.begin();
         remoteRankIter != remoteRoutes.end(); ++remoteRankIter) {
      free(remoteRankIter->second);
    }
  }
  delete interServerTopo;
}
