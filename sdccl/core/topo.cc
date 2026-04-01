/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "topo.h"
#include "bootstrap.h"
#include "comm.h"
#include "core.h"
#include "cpuset.h"
#include "graph.h"
#include "net.h"
#include "rapidxml.h"
#include "transport.h"
#include "xml.h"
#include <fcntl.h>
#include <fstream>
#include <map>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char *topoNodeTypeStr[] = {"APU", "PCI", "CCI", "CPU",
                                 "NIC", "NET", "HBD"};
const char *topoLinkTypeStr[] = {"LOC", "CCI", "",    "PCI", "",
                                 "",    "",    "SYS", "NET"};
const char *topoPathTypeStr[] = {"LOC", "CCI", "CCB", "PIX", "PXB",
                                 "PXN", "PHB", "SYS", "NET", "DIS"};

struct kvDict kvDictPciGen[] = {{"2.5 GT/s", 15},
                                {"5 GT/s", 30},
                                {"8 GT/s", 60},
                                {"16 GT/s", 120},
                                {"32 GT/s", 240}, /* Kernel 5.6 and earlier */
                                {"2.5 GT/s PCIe", 15},
                                {"5.0 GT/s PCIe", 30},
                                {"8.0 GT/s PCIe", 60},
                                {"16.0 GT/s PCIe", 120},
                                {"32.0 GT/s PCIe", 240},
                                {"64.0 GT/s PCIe", 480},
                                {NULL, 60 /* Default fallback */}};

sdcclResult_t sdcclTopoGetLocal(struct sdcclTopoServer *topoServer, int type,
                                  int index, int resultType, int **locals,
                                  int *localCount, int *pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  SDCCLCHECK(sdcclCalloc(locals, topoServer->nodes[resultType].count));
  struct sdcclTopoPath *paths =
      topoServer->nodes[type].nodes[index].paths[resultType];

  for (int i = 0; i < topoServer->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw ||
        (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType)
        *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType)
      (*locals)[count++] = i;
  }
  *localCount = count;
  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoGetInterCpuBw(struct sdcclTopoNode *cpu,
                                              float *bw) {
  *bw = LOC_BW;
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return sdcclSuccess;
  }
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return sdcclSuccess;
  }
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_X86 &&
      cpu->cpu.vendor == SDCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw = cpu->cpu.model == SDCCL_TOPO_CPU_TYPE_SKL ? SKL_QPI_BW : QPI_BW;
  }
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_X86 &&
      cpu->cpu.vendor == SDCCL_TOPO_CPU_VENDOR_AMD) {
    *bw = AMD_BW;
  }
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_X86 &&
      cpu->cpu.vendor == SDCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model == SDCCL_TOPO_CPU_TYPE_YONGFENG ? YONGFENG_ZPI_BW
                                                          : ZPI_BW;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetNode(struct sdcclTopoServer *topoServer,
                                 struct sdcclTopoNode **node, int type,
                                 uint64_t id) {
  for (int i = 0; i < topoServer->nodes[type].count; i++) {
    if (topoServer->nodes[type].nodes[i].id == id) {
      *node = topoServer->nodes[type].nodes + i;
      return sdcclSuccess;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoCreateNode(struct sdcclTopoServer *topoServer,
                                    struct sdcclTopoNode **node, int type,
                                    uint64_t id) {
  if (topoServer->nodes[type].count == SDCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return sdcclInternalError;
  }
  struct sdcclTopoNode *tempNode =
      topoServer->nodes[type].nodes + topoServer->nodes[type].count;
  topoServer->nodes[type].count++;
  tempNode->type = type;
  tempNode->id = id;
  if (type == APU) {
    tempNode->nlinks = 1;
    tempNode->links[0].type = LINK_LOC;
    tempNode->links[0].remNode = tempNode;
    tempNode->links[0].bw = LOC_BW; // TODO: local bw of different APUs might
                                    // differ, change this in the future
    tempNode->apu.dev = SDCCL_TOPO_UNDEF;
    tempNode->apu.rank = SDCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    tempNode->cpu.arch = SDCCL_TOPO_UNDEF;
    tempNode->cpu.vendor = SDCCL_TOPO_UNDEF;
    tempNode->cpu.model = SDCCL_TOPO_UNDEF;
  } else if (type == NET) {
    tempNode->net.guid = 0ULL;
    tempNode->net.port = SDCCL_TOPO_UNDEF;
    tempNode->net.bw = 0.0;
    tempNode->net.latency = 0.0;
  }
  *node = tempNode;
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoConnectNodes(struct sdcclTopoNode *node,
                                      struct sdcclTopoNode *remNode, int type,
                                      float bw) {
  struct sdcclTopoLink *link;
  // check if there's an existing link of this type between node and remNode
  for (link = node->links;
       link - node->links != SDCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type)
      break;
  }
  if (link - node->links == SDCCL_TOPO_MAX_LINKS) {
    WARN("ERROR: too many topo links (max %d)", SDCCL_TOPO_MAX_LINKS);
    return sdcclInternalError;
  }
  if (link->remNode == NULL)
    node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;
  // TODO: sort links in BW descending order when we have bw info
  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoIdToIndex(struct sdcclTopoServer *topoServer,
                                          int type, int64_t id, int *index) {
  *index = -1;
  for (int i = 0; i < topoServer->nodes[type].count; i++) {
    if (topoServer->nodes[type].nodes[i].id == id) {
      *index = i;
      return sdcclSuccess;
    }
  }
  return sdcclInternalError;
}

sdcclResult_t sdcclTopoRemoveNode(struct sdcclTopoServer *topoServer,
                                    int type, int index) {
  struct sdcclTopoNode *delNode = topoServer->nodes[type].nodes + index;
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    free(delNode->paths[t]);
    for (int n = 0; n < topoServer->nodes[t].count; n++) {
      struct sdcclTopoNode *node = topoServer->nodes[t].nodes + n;
      if (node == delNode)
        continue;
      for (int l = 0; l < node->nlinks; l++) {
        while (l < node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links + l, node->links + l + 1,
                  (node->nlinks - l - 1) * sizeof(struct sdcclTopoLink));
          node->nlinks--;
        }
        if (l < node->nlinks && node->links[l].remNode->type == type &&
            node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode + 1,
          (topoServer->nodes[type].count - index - 1) *
              sizeof(struct sdcclTopoNode));
  topoServer->nodes[type].count--;
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoConnectCpus(struct sdcclTopoServer *topoServer) {
  for (int i = 0; i < topoServer->nodes[CPU].count; i++) {
    struct sdcclTopoNode *cpu1 = topoServer->nodes[CPU].nodes + i;
    for (int j = 0; j < topoServer->nodes[CPU].count; j++) {
      struct sdcclTopoNode *cpu2 = topoServer->nodes[CPU].nodes + j;
      if (i == j || (SDCCL_TOPO_ID_SERVER_ID(cpu1->id) !=
                     SDCCL_TOPO_ID_SERVER_ID(cpu2->id))) {
        continue;
      }
      float bw;
      SDCCLCHECK(sdcclTopoGetInterCpuBw(cpu1, &bw));
      SDCCLCHECK(sdcclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return sdcclSuccess;
}

int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000)
    return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level * 0x1000))
    return 5;
  return 0;
}

sdcclResult_t
sdcclTopoFlattenBcmSwitches(struct sdcclTopoServer *topoServer) {
  sdcclResult_t ret = sdcclSuccess;
  for (int s = 0; s < topoServer->nodes[PCI].count; s++) {
    struct sdcclTopoNode *pciSwitch = topoServer->nodes[PCI].nodes + s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);
    // Flatten Gen4 PEX switches in base mode
    if (gen) {
      // Find sub switches with the same device ID.
      int64_t *subSwIds;
      SDCCLCHECK(sdcclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l = 0; l < pciSwitch->nlinks; l++) {
        struct sdcclTopoNode *sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen)
          continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links + l, pciSwitch->links + l + 1,
                (pciSwitch->nlinks - l - 1) * (sizeof(struct sdcclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links
        // by one.
        l--;
      }

      for (int s = 0; s < subs; s++) {
        // Find sub switch (topoServer->nodes[PCI].nodes is changing every time
        // we remove a node)
        int index;
        SDCCLCHECKGOTO(
            sdcclTopoIdToIndex(topoServer, PCI, subSwIds[s], &index), ret,
            fail);
        struct sdcclTopoNode *sub = topoServer->nodes[PCI].nodes + index;
        // Connect all sub PCI devices to the parent switch
        for (int l = 0; l < sub->nlinks; l++) {
          struct sdcclTopoNode *remNode = sub->links[l].remNode;
          if (remNode == pciSwitch)
            continue;
          // Add link from parent PCI switch -> PCI device
          if (pciSwitch->nlinks == SDCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", SDCCL_TOPO_MAX_LINKS);
            ret = sdcclInternalError;
            goto fail;
          }
          memcpy(pciSwitch->links + pciSwitch->nlinks, sub->links + l,
                 sizeof(struct sdcclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl = 0; rl < remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        SDCCLCHECKGOTO(sdcclTopoRemoveNode(topoServer, PCI, index), ret,
                        fail);
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      pciSwitch->pci.device |= 0xffff;
      free(subSwIds);
      // Restart, as topoServer->nodes[PCI].nodes has changed.
      s = 0;
      continue;
    fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

// sdcclResult_t getLocalNetCountByBw(struct sdcclTopoServer* system, int gpu,
// int *count) {
//   int localNetCount = 0, netCountByBw = 0;
//   int* localNets;
//   float totalNetBw = 0, gpuBw = 0;

//   for (int l=0; l<system->nodes[GPU].nodes[gpu].nlinks; l++) {
//     //assuming BW to CPU reflects the GPU bandwidth via P2P or C2C
//     //caveat, this could be wrong if there is a PCIe switch,
//     //and a narrower link to the CPU
//     if (system->nodes[GPU].nodes[gpu].links[l].remNode->type == CPU) {
//        gpuBw = system->nodes[GPU].nodes[gpu].links[l].bw;
//     }
//   }

//   SDCCLCHECK(sdcclTopoGetLocal(system, GPU, gpu, NET, &localNets,
//   &localNetCount, NULL)); for (int l=0; (l < localNetCount) && (totalNetBw <
//   gpuBw); l++, netCountByBw++) {
//      totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][localNets[l]].bw;
//   }
//   *count = netCountByBw;

//   free(localNets);
//   return sdcclSuccess;
// }

// a temprarory function to get the local net from topo xml file.
// devId: the device id of the GPU
// netName: the name of the net
// strlen: the length of the netName
static sdcclResult_t sdcclGetLocalNetFromXmlFile(int devId, char *netName,
                                                   int strlen) {
  sdcclResult_t ret = sdcclSuccess;
  sdcclXmlNode *node = NULL;
  int dev = -1;
  // step 1: parse the xml file and load it into sdcclXml struct
  struct sdcclXml *xml;
  const char *xmlTopoFile = sdcclGetEnv("SDCCL_TOPO_FILE");
  if (!xmlTopoFile) {
    INFO(SDCCL_ENV, "SDCCL_TOPO_FILE environment variable not set");
    return ret;
  }
  SDCCLCHECK(xmlAlloc(&xml, SDCCL_TOPO_XML_MAX_NODES));
  INFO(SDCCL_ENV, "SDCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
  SDCCLCHECKGOTO(sdcclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);

  // step 2: scan sdcclXml struct to find the netName for the given devId
  SDCCLCHECKGOTO(xmlFindTag(xml, "gpu", &node), ret, fail);
  while (node != NULL) {
    // find the gpu node with the right dev
    SDCCLCHECKGOTO(xmlGetAttrInt(node, "dev", &dev), ret, fail);
    if (dev == devId) {
      const char *str;
      SDCCLCHECKGOTO(xmlGetAttr(node, "net", &str), ret, fail);
      if (str != NULL) {
        INFO(SDCCL_GRAPH, "GPU %d use net %s specified in topo file %s", dev,
             str, xmlTopoFile);
        strncpy(netName, str, strlen - 1);
        netName[strlen - 1] = '\0';
        break;
      } else {
        WARN("GPU %d net attribute is not specified in topo file %s", dev,
             xmlTopoFile);
        ret = sdcclInternalError;
        goto fail;
      }
    }
    sdcclXmlNode *next = NULL;
    SDCCLCHECKGOTO(xmlFindNextTag(xml, "gpu", node, &next), ret, fail);
    node = next;
  }
  if (dev != devId) {
    // device not found
    WARN("GPU %d not found in topo file %s", devId, xmlTopoFile);
    ret = sdcclInternalError;
    goto fail;
  }
exit:
  free(xml);
  return ret;
fail:
  goto exit;
}

#define SDCCL_MAX_NET_NAME 128

sdcclResult_t sdcclGetLocalNetFromXml(struct sdcclXml *xml, int apu,
                                        char *name, int strlen) {
  struct sdcclXmlNode *apuNode = NULL;
  SDCCLCHECK(xmlGetApuByIndex(xml, apu, &apuNode));
  if (apuNode == NULL) {
    WARN("invalid apu index %d", apu);
    return sdcclInternalError;
  }
  struct sdcclXmlNode *netNode = NULL;
  // first try to find the closest net under one CPU node
  SDCCLCHECK(xmlFindClosestNetUnderCpu(xml, apuNode, &netNode));
  if (netNode == NULL) {
    // if there is no net node that share the same CPU ancestor node with the
    // APU try to find a net node from the server scope
    SDCCLCHECK(xmlFindClosestNetUnderServer(xml, apuNode, &netNode));
  }
  if (netNode != NULL) {
    // found a net node
    const char *str;
    SDCCLCHECK(xmlGetAttrStr(netNode, "name", &str)); // get net name
    strncpy(name, str, strlen);
    INFO(SDCCL_INIT, "local net for apu %d is %s", apu, name);
  }
  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoRankToIndex(struct sdcclTopoServer *topoServer,
                                            int rank, int *index) {
  *index = -1;
  for (int i = 0; i < topoServer->nodes[APU].count; i++) {
    if (topoServer->nodes[APU].nodes[i].apu.rank == rank) {
      *index = i;
      return sdcclSuccess;
    }
  }
  return sdcclInternalError;
}

static sdcclResult_t sdcclTopoGetLocal(struct sdcclTopoServer *topoServer,
                                         int type, int index, int resultType,
                                         int locals[SDCCL_TOPO_MAX_NODES],
                                         int *localCount, int *pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct sdcclTopoPath *paths =
      topoServer->nodes[type].nodes[index].paths[resultType];
  if (paths == NULL) {
    *localCount = 0;
    return sdcclSuccess;
  }
  for (int i = 0; i < topoServer->nodes[resultType].count; i++) {
    if (paths[i].bw > maxBw ||
        (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;
      minType = paths[i].type;
      if (pathType)
        *pathType = minType;
      count = 0;
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {
      if (count == SDCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in "
             "sdcclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             SDCCL_TOPO_MAX_NODES, resultType, index, type);
        return sdcclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetLocalNet(struct sdcclTopoServer *topoServer,
                                     int rank, int *netDev) {
  int apu;
  SDCCLCHECK(sdcclTopoRankToIndex(topoServer, rank, &apu));

  int localNets[SDCCL_TOPO_MAX_NODES];
  int localNetCount;
  SDCCLCHECK(sdcclTopoGetLocal(topoServer, APU, apu, NET, localNets,
                                 &localNetCount, NULL));
  if (localNetCount == 0) {
    WARN("Could not find any local path from apu %d to net", apu);
    return sdcclInternalError;
  }

  INFO(SDCCL_GRAPH, "found %d local nets for apu %d", localNetCount, apu);
  int net = topoServer->nodes[APU].nodes[apu].apu.dev;
  if (isPow2(localNetCount)) { // load balance across apus
    net = mirrorBits(net, localNetCount);
  }
  if (netDev) {
    *netDev =
        topoServer->nodes[NET].nodes[localNets[net % localNetCount]].net.dev;
    INFO(SDCCL_GRAPH, "local net for apu %d is %d", apu, *netDev);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetLocalNetNode(struct sdcclTopoServer *topoServer,
                                         int rank,
                                         struct sdcclTopoNode **netNode) {
  int apu;
  SDCCLCHECK(sdcclTopoRankToIndex(topoServer, rank, &apu));

  int localNets[SDCCL_TOPO_MAX_NODES];
  int localNetCount;
  SDCCLCHECK(sdcclTopoGetLocal(topoServer, APU, apu, NET, localNets,
                                 &localNetCount, NULL));
  if (localNetCount == 0) {
    WARN("Could not find any local path from apu %d to net", apu);
    return sdcclInternalError;
  }

  INFO(SDCCL_GRAPH, "found %d local nets for apu %d", localNetCount, apu);
  int net = topoServer->nodes[APU].nodes[apu].apu.dev;
  if (isPow2(localNetCount)) { // load balance across apus
    net = mirrorBits(net, localNetCount);
  }
  *netNode = &(topoServer->nodes[NET].nodes[localNets[net % localNetCount]]);
  return sdcclSuccess;
}

sdcclResult_t sdcclGetLocalNetFromGpu(int apu, int *dev,
                                        struct sdcclHeteroComm *comm) {
  char name[SDCCL_MAX_NET_NAME + 1] = {0};
  // first try getting local net from existing xml file
  SDCCLCHECK(sdcclGetLocalNetFromXmlFile(apu, name, SDCCL_MAX_NET_NAME + 1));
  if (strlen(name) != 0) {
    comm->netAdaptor->getDevFromName(name, dev);
  }

  if (strlen(name) == 0) {
    SDCCLCHECK(sdcclTopoGetLocalNet(comm->topoServer, comm->rank, dev));
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclGetNicDistance(struct sdcclTopoServer *topoServer,
                                    int rank,
                                    struct sdcclNicDistance *distInfo) {
  int netDev;
  SDCCLCHECK(sdcclTopoGetLocalNet(topoServer, rank, &netDev));
  int apuIdx;
  SDCCLCHECK(sdcclTopoRankToIndex(topoServer, rank, &apuIdx));
  struct sdcclTopoPath *paths =
      topoServer->nodes[APU].nodes[apuIdx].paths[NET];
  for (int i = 0; i < topoServer->nodes[NET].count; i++) {
    if (topoServer->nodes[NET].nodes[i].net.dev == netDev) {
      distInfo->distance = paths[i].type;
      distInfo->netGuid = topoServer->nodes[NET].nodes[i].net.guid;
      return sdcclSuccess;
    }
  }
  return sdcclInternalError;
}

/****************************/
/* External query functions */
/****************************/

// sdcclResult_t sdcclTopoCpuType(struct sdcclTopoServer* system, int* arch,
// int* vendor, int* model) {
//   *arch = system->nodes[CPU].nodes[0].cpu.arch;
//   *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
//   *model = system->nodes[CPU].nodes[0].cpu.model;
//   return sdcclSuccess;
// }

// SDCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

// sdcclResult_t sdcclTopoGetGpuCount(struct sdcclTopoServer* system, int*
// count) {
//   *count = system->nodes[GPU].count;
//   return sdcclSuccess;
// }

// sdcclResult_t sdcclTopoGetNetCount(struct sdcclTopoServer* system, int*
// count) {
//   *count = system->nodes[NET].count;
//   return sdcclSuccess;
// }

// sdcclResult_t sdcclTopoGetNvsCount(struct sdcclTopoServer* system, int*
// count) {
//   *count = system->nodes[NVS].count;
//   return sdcclSuccess;
// }

// sdcclResult_t sdcclTopoGetCompCap(struct sdcclTopoServer* system, int*
// ccMin, int* ccMax) {
//   if (system->nodes[GPU].count == 0) return sdcclInternalError;
//   int min, max;
//   min = max = system->nodes[GPU].nodes[0].apu.cudaCompCap;
//   for (int g=1; g<system->nodes[GPU].count; g++) {
//     min = std::min(min, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
//     max = std::max(max, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
//   }
//   if (ccMin) *ccMin = min;
//   if (ccMax) *ccMax = max;
//   return sdcclSuccess;
// }

// static sdcclResult_t sdcclTopoPrintRec(struct sdcclTopoNode* node, struct
// sdcclTopoNode* prevNode, char* line, int offset) {
//   if (node->type == GPU) {
//     sprintf(line+offset, "%s/%lx-%lx (%d)", topoNodeTypeStr[node->type],
//     SDCCL_TOPO_ID_SERVER_ID(node->id), SDCCL_TOPO_ID_LOCAL_ID(node->id),
//     node->apu.rank);
//   } else if (node->type == CPU) {
//     sprintf(line+offset, "%s/%lx-%lx (%d/%d/%d)",
//     topoNodeTypeStr[node->type], SDCCL_TOPO_ID_SERVER_ID(node->id),
//     SDCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor,
//     node->cpu.model);
//   } else if (node->type == PCI) {
//     sprintf(line+offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type],
//     SDCCL_TOPO_ID_SERVER_ID(node->id), SDCCL_TOPO_ID_LOCAL_ID(node->id),
//     node->pci.device);
//   } else {
//     sprintf(line+offset, "%s/%lx-%lx", topoNodeTypeStr[node->type],
//     SDCCL_TOPO_ID_SERVER_ID(node->id), SDCCL_TOPO_ID_LOCAL_ID(node->id));
//   }
//   INFO(SDCCL_GRAPH, "%s", line);
//   for (int i=0; i<offset; i++) line[i] = ' ';

//   for (int l=0; l<node->nlinks; l++) {
//     struct sdcclTopoLink* link = node->links+l;
//     if (link->type == LINK_LOC) continue;
//     if (link->type != LINK_PCI || link->remNode != prevNode) {
//       sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type],
//       link->bw); int nextOffset = strlen(line); if (link->type == LINK_PCI) {
//         SDCCLCHECK(sdcclTopoPrintRec(link->remNode, node, line,
//         nextOffset));
//       } else {
//         if (link->remNode->type == NET) {
//           sprintf(line+nextOffset, "%s/%lX (%lx/%d/%f)",
//           topoNodeTypeStr[link->remNode->type], link->remNode->id,
//           link->remNode->net.asic, link->remNode->net.port,
//           link->remNode->net.bw);
//         } else {
//           sprintf(line+nextOffset, "%s/%lX",
//           topoNodeTypeStr[link->remNode->type], link->remNode->id);
//         }
//         INFO(SDCCL_GRAPH, "%s", line);
//       }
//     }
//   }
//   return sdcclSuccess;
// }

// sdcclResult_t sdcclTopoPrint(struct sdcclTopoServer* s) {
//   INFO(SDCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw,
//   s->totalBw); char line[1024]; for (int n=0; n<s->nodes[CPU].count; n++)
//   SDCCLCHECK(sdcclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
//   INFO(SDCCL_GRAPH, "==========================================");
//   SDCCLCHECK(sdcclTopoPrintPaths(s));
//   return sdcclSuccess;
// }

// will remove this function when we finish the function that builds server topo
sdcclResult_t sdcclTopoGetXmlTopo(struct sdcclHeteroComm *comm,
                                    struct sdcclXml *xml) {
  // create root node if we didn't get topo from xml file
  if (xml->maxIndex == 0) {
    INFO(SDCCL_INIT, "creating root XML node");
    // Create top tag
    struct sdcclXmlNode *top;
    // TODO: change root node name from "system" to "root"
    SDCCLCHECK(xmlAddNode(xml, NULL, "system", &top));
    SDCCLCHECK(xmlSetAttrInt(top, "version", SDCCL_TOPO_XML_VERSION));
  }

  INFO(SDCCL_INIT, "start detecting APUs");
  for (int r = 0; r < comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash) {
      INFO(SDCCL_INIT, "preparing to detect APU for rank %d", r);
      char busId[SDCCL_DEVICE_PCI_BUSID_BUFFER_SIZE];
      INFO(SDCCL_INIT, "converting busId to string");
      SDCCLCHECK(int64ToBusId(comm->peerInfo[r].busId, busId));
      struct sdcclXmlNode *node;
      SDCCLCHECK(sdcclTopoFillApu(xml, busId, &node));
      if (node == NULL) {
        continue;
      }
      int devLogicalIdx = 0;
      deviceAdaptor->getDeviceByPciBusId(&devLogicalIdx, busId);
      SDCCLCHECK(xmlSetAttrInt(node, "dev", devLogicalIdx));
      SDCCLCHECK(xmlSetAttrInt(node, "rank", r));
    }
  }

  int netDevCount = 0;
  SDCCLCHECK(comm->netAdaptor->devices(&netDevCount));
  for (int n = 0; n < netDevCount; n++) {
    sdcclNetProperties_t props;
    SDCCLCHECK(comm->netAdaptor->getProperties(n, (void *)&props));
    struct sdcclXmlNode *netNode;
    SDCCLCHECK(sdcclTopoFillNet(xml, props.pciPath, props.name, &netNode));
    SDCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    SDCCLCHECK(xmlSetAttrInt(netNode, "speed", props.speed));
    SDCCLCHECK(xmlSetAttrFloat(netNode, "latency", props.latency));
    SDCCLCHECK(xmlSetAttrInt(netNode, "port", props.port));
    SDCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    SDCCLCHECK(xmlSetAttrInt(netNode, "maxConn", props.maxComms));
  }

  if (comm->rank == 0) {
    const char *xmlTopoFile = sdcclGetEnv("SDCCL_TOPO_DUMP_FILE");
    INFO(SDCCL_ENV, "SDCCL_TOPO_DUMP_FILE is %s", xmlTopoFile);
    if (xmlTopoFile && comm->rank == 0) {
      INFO(SDCCL_INIT, "start dumping topo to xml file");
      SDCCLCHECK(sdcclTopoDumpXmlToFile(xmlTopoFile, xml));
    }
  }
  return sdcclSuccess;
}

struct kvDict kvDictCpuArch[] = {{"x86_64", SDCCL_TOPO_CPU_ARCH_X86},
                                 {"arm64", SDCCL_TOPO_CPU_ARCH_ARM},
                                 {"ppc64", SDCCL_TOPO_CPU_ARCH_POWER},
                                 {NULL, 0}};
struct kvDict kvDictCpuVendor[] = {
    {"GenuineIntel", SDCCL_TOPO_CPU_VENDOR_INTEL},
    {"AuthenticAMD", SDCCL_TOPO_CPU_VENDOR_AMD},
    {"CentaurHauls", SDCCL_TOPO_CPU_VENDOR_ZHAOXIN},
    {"  Shanghai  ", SDCCL_TOPO_CPU_VENDOR_ZHAOXIN},
    {NULL, 0}};

sdcclResult_t sdcclGetServerId(struct sdcclTopoServer *topoServer,
                                 struct sdcclXmlNode *xmlCpu,
                                 int *serverIdPtr) {
  const char *hostHashStr;
  SDCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int serverId;
  for (serverId = 0; serverId < topoServer->nHosts; serverId++) {
    if (topoServer->hostHashes[serverId] == hostHash) {
      break;
    }
  }
  // if current host hash hasn't been seen before, this is a new host
  if (serverId == topoServer->nHosts) {
    topoServer->hostHashes[topoServer->nHosts++] = hostHash;
  }
  *serverIdPtr = serverId;
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoAddNet(struct sdcclXmlNode *xmlNet,
                                struct sdcclTopoServer *topoServer,
                                struct sdcclTopoNode *nic, int serverId) {
  int dev;
  SDCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  struct sdcclTopoNode *net;
  SDCCLCHECK(sdcclTopoCreateNode(topoServer, &net, NET,
                                   SDCCL_TOPO_ID(serverId, dev)));
  net->net.dev = dev;
  int mbps;
  // SDCCLCHECK(xmlGetAttrLong(xmlNet, "guid", &net->net.guid));
  const char *str;
  SDCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  if (str) {
    sscanf(str, "0x%lx", &net->net.guid);
  } else {
    net->net.guid = dev;
  }
  INFO(SDCCL_GRAPH, "ADDING NET: net %d guid %lx", dev, net->net.guid);
  SDCCLCHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
  if (mbps <= 0) {
    mbps = 10000;
  }
  net->net.bw = mbps / 8000.0;
  SDCCLCHECK(xmlGetAttrFloat(xmlNet, "latency", &net->net.latency));
  SDCCLCHECK(xmlGetAttrInt(xmlNet, "port", &net->net.port));
  SDCCLCHECK(xmlGetAttrInt(xmlNet, "maxConn", &net->net.maxConn));

  SDCCLCHECK(sdcclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  SDCCLCHECK(sdcclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoAddNic(struct sdcclXmlNode *xmlNic,
                                struct sdcclTopoServer *topoServer,
                                struct sdcclTopoNode *nic, int serverId) {
  for (int s = 0; s < xmlNic->nSubs; s++) {
    struct sdcclXmlNode *xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0)
      continue;
    int index;
    SDCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    if (index == -1)
      continue;
    SDCCLCHECK(sdcclTopoAddNet(xmlNet, topoServer, nic, serverId));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoAddApu(struct sdcclXmlNode *xmlApu,
                                struct sdcclTopoServer *topoServer,
                                struct sdcclTopoNode *apu) {
  // we add attributes of the current apu here
  // right now we only have the device logic index of the apu, add more info in
  // the future
  SDCCLCHECK(xmlGetAttrInt(xmlApu, "dev", &apu->apu.dev));
  SDCCLCHECK(xmlGetAttrInt(xmlApu, "rank", &apu->apu.rank));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoAddPci(struct sdcclXmlNode *xmlPci,
                                struct sdcclTopoServer *topoServer,
                                struct sdcclTopoNode *parent, int serverId) {
  const char *str;

  // Assume default type is PCI
  int type = PCI;

  int64_t busId;
  SDCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  SDCCLCHECK(busIdToInt64(str, &busId));

  struct sdcclTopoNode *node = NULL;
  struct sdcclXmlNode *xmlApu = NULL;
  // check if there is any APU attached to current pci device
  SDCCLCHECK(xmlGetSub(xmlPci, "apu", &xmlApu));
  if (xmlApu != NULL) {
    type = APU;
    // TODO: need to get apu rank info when building xml structure
    // get apu rank here
    SDCCLCHECK(sdcclTopoCreateNode(topoServer, &node, type,
                                     SDCCL_TOPO_ID(serverId, busId)));
    SDCCLCHECK(sdcclTopoAddApu(xmlApu, topoServer, node));
  }
  struct sdcclXmlNode *xmlNic = NULL;
  // check if there is any APU attached to current pci device
  SDCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    busId &= 0xfffffffffffffff0;
    struct sdcclTopoNode *nicNode = NULL;
    int64_t id = SDCCL_TOPO_ID(serverId, busId);
    SDCCLCHECK(sdcclTopoGetNode(topoServer, &nicNode, type, id));
    if (nicNode == NULL) {
      SDCCLCHECK(sdcclTopoCreateNode(topoServer, &nicNode, type, id));
      node = nicNode;
    }

    SDCCLCHECK(sdcclTopoAddNic(xmlNic, topoServer, nicNode, serverId));
  } else if (type == PCI) {
    SDCCLCHECK(sdcclTopoCreateNode(topoServer, &node, type,
                                     SDCCL_TOPO_ID(serverId, busId)));
    // the following block is essentially storing pci device info into a unint64
    // each of the four attributes is 16bit long
    SDCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str)
      node->pci.device +=
          strtol(str, NULL, 0)
          << 48; // magic number, see if we can make it a constant
    SDCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str)
      node->pci.device += strtol(str, NULL, 0) << 32;
    SDCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str)
      node->pci.device += strtol(str, NULL, 0) << 16;
    SDCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str)
      node->pci.device += strtol(str, NULL, 0);

    // recursively add sub pci devices
    for (int s = 0; s < xmlPci->nSubs; s++) {
      struct sdcclXmlNode *xmlSubPci = xmlPci->subs[s];
      SDCCLCHECK(sdcclTopoAddPci(xmlSubPci, topoServer, node, serverId));
    }
  }

  if (node) {
    int width, speed;
    SDCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    SDCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));
    if (width == 0)
      width = 16;
    SDCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen));
    SDCCLCHECK(
        sdcclTopoConnectNodes(node, parent, LINK_PCI, width * speed / 80.0));
    SDCCLCHECK(
        sdcclTopoConnectNodes(parent, node, LINK_PCI, width * speed / 80.0));
  }
  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoGetCpuArch(const char *archStr, int *ret) {
  SDCCLCHECK(kvConvertToInt(archStr, ret, kvDictCpuArch));
  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoGetCpuVendor(const char *vendorStr, int *ret) {
  SDCCLCHECK(kvConvertToInt(vendorStr, ret, kvDictCpuVendor));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoAddCpu(struct sdcclXmlNode *xmlCpu,
                                struct sdcclTopoServer *topoServer) {
  int numaId;
  SDCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  int serverId;
  SDCCLCHECK(sdcclGetServerId(topoServer, xmlCpu, &serverId));
  struct sdcclTopoNode *cpu;
  SDCCLCHECK(sdcclTopoCreateNode(topoServer, &cpu, CPU,
                                   SDCCL_TOPO_ID(serverId, numaId)));
  const char *str;
  SDCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    SDCCLCHECK(sdcclStrToCpuset(str, &cpu->cpu.affinity));
  }

  SDCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  SDCCLCHECK(sdcclTopoGetCpuArch(str, &cpu->cpu.arch));
  if (cpu->cpu.arch == SDCCL_TOPO_CPU_ARCH_X86) {
    SDCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    SDCCLCHECK(sdcclTopoGetCpuVendor(str, &cpu->cpu.vendor));
    if (cpu->cpu.vendor == SDCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      SDCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      SDCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model = (familyId == 6 && modelId >= 0x55)
                           ? SDCCL_TOPO_CPU_TYPE_SKL
                           : SDCCL_TOPO_CPU_INTEL_BDW;
    } else if (cpu->cpu.vendor == SDCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      SDCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      SDCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B)
        cpu->cpu.model = SDCCL_TOPO_CPU_TYPE_YONGFENG;
    }
  }
  for (int s = 0; s < xmlCpu->nSubs; s++) {
    struct sdcclXmlNode *node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0)
      SDCCLCHECK(sdcclTopoAddPci(node, topoServer, cpu, serverId));
    if (strcmp(node->name, "nic") == 0) {
      struct sdcclTopoNode *nic = NULL;
      SDCCLCHECK(sdcclTopoGetNode(topoServer, &nic, NIC, 0));
      if (nic == NULL) {
        SDCCLCHECK(sdcclTopoCreateNode(topoServer, &nic, NIC,
                                         SDCCL_TOPO_ID(serverId, 0)));
        SDCCLCHECK(sdcclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        SDCCLCHECK(sdcclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      SDCCLCHECK(sdcclTopoAddNic(node, topoServer, nic, serverId));
    }
  }
  return sdcclSuccess;
}

sdcclResult_t
sdcclTopoGetServerTopoFromXml(struct sdcclXml *xml,
                               struct sdcclTopoServer **topoServer,
                               const uint64_t localHostHash) {
  SDCCLCHECK(sdcclCalloc(topoServer, 1));
  struct sdcclTopoServer *server = *topoServer;
  // get root node from xml
  struct sdcclXmlNode *topNode;
  SDCCLCHECK(xmlFindTag(xml, "system", &topNode));
  for (int s = 0; s < topNode->nSubs; s++) {
    struct sdcclXmlNode *node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0)
      SDCCLCHECK(sdcclTopoAddCpu(node, *topoServer));
  }
  // get the correct serverId for current server
  for (int serverId = 0; serverId < server->nHosts; serverId++) {
    if (server->hostHashes[serverId] == localHostHash) {
      server->serverId = serverId;
    }
  }

  // TODO: add CCI links, connect cpu nodes etc.
  SDCCLCHECK(sdcclTopoFlattenBcmSwitches(*topoServer));
  SDCCLCHECK(sdcclTopoConnectCpus(*topoServer));

  return sdcclSuccess;
}

static sdcclResult_t sdcclTopoPrintRec(struct sdcclTopoNode *node,
                                         struct sdcclTopoNode *prevNode,
                                         char *line, int offset) {
  if (node->type == APU) {
    // TODO: add rank info
    sprintf(line + offset, "Node [%s/%lx-%lx (%d)]",
            topoNodeTypeStr[node->type], SDCCL_TOPO_ID_SERVER_ID(node->id),
            SDCCL_TOPO_ID_LOCAL_ID(node->id), node->apu.rank);
  } else if (node->type == CPU) {
    sprintf(line + offset, "Node [%s/%lx-%lx (%d/%d/%d)]",
            topoNodeTypeStr[node->type], SDCCL_TOPO_ID_SERVER_ID(node->id),
            SDCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor,
            node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line + offset, "Node [%s/%lx-%lx (%lx)]",
            topoNodeTypeStr[node->type], SDCCL_TOPO_ID_SERVER_ID(node->id),
            SDCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else {
    sprintf(line + offset, "Node [%s/%lx-%lx]", topoNodeTypeStr[node->type],
            SDCCL_TOPO_ID_SERVER_ID(node->id),
            SDCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(SDCCL_GRAPH, "%s", line);
  for (int i = 0; i < offset; i++)
    line[i] = ' ';

  for (int l = 0; l < node->nlinks; l++) {
    struct sdcclTopoLink *link = node->links + l;
    if (link->type == LINK_LOC)
      continue;
    if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line + offset, "+ Link[%s/%2.1f] - ", topoLinkTypeStr[link->type],
              link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        SDCCLCHECK(sdcclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line + nextOffset, "Node [%s/%lx (%lx/%d/%f)]",
                  topoNodeTypeStr[link->remNode->type], link->remNode->id,
                  link->remNode->net.guid, link->remNode->net.port,
                  link->remNode->net.bw);
        } else {
          sprintf(line + nextOffset, "Node [%s/%lx]",
                  topoNodeTypeStr[link->remNode->type], link->remNode->id);
        }
        INFO(SDCCL_GRAPH, "%s", line);
      }
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoPrint(struct sdcclTopoServer *topoServer) {
  char line[1024];
  // start printing topology from CPU nodes
  INFO(SDCCL_INIT, "start printing server topology");
  for (int n = 0; n < topoServer->nodes[CPU].count; n++) {
    SDCCLCHECK(
        sdcclTopoPrintRec(topoServer->nodes[CPU].nodes + n, NULL, line, 0));
  }
  INFO(SDCCL_GRAPH, "==========================================");
  SDCCLCHECK(sdcclTopoPrintPaths(topoServer));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetServerTopo(struct sdcclHeteroComm *comm,
                                       struct sdcclTopoServer **topoServer) {
  // TODO: first try to acquire topo from xml file
  struct sdcclXml *xml;
  INFO(SDCCL_INIT, "allocing sdcclXml");
  SDCCLCHECK(xmlAlloc(&xml, SDCCL_TOPO_XML_MAX_NODES));

  SDCCLCHECK(sdcclTopoGetXmlTopo(comm, xml));
  INFO(SDCCL_INIT, "start converting xml to serverTopo");
  uint64_t localHostHash = comm->peerInfo[comm->rank].hostHash -
                           comm->commHash; // do not consider commHash here
  SDCCLCHECK(sdcclTopoGetServerTopoFromXml(xml, topoServer, localHostHash));

  free(xml);
  return sdcclSuccess;
}

static sdcclResult_t flattenLink(struct sdcclTopoServer *topoServer,
                                  struct sdcclTopoLink *link,
                                  struct flatTopoLink *flatLink) {
  flatLink->type = link->type;
  flatLink->bw = link->bw;
  sdcclTopoNode *remNode = link->remNode;
  int remNodeIdx;
  SDCCLCHECK(
      sdcclTopoIdToIndex(topoServer, remNode->type, remNode->id, &remNodeIdx));
  flatLink->remNodeIdx = remNodeIdx;
  flatLink->remNodeType = remNode->type;
  return sdcclSuccess;
}

static sdcclResult_t unflattenLink(struct sdcclTopoServer *topoServer,
                                    struct sdcclTopoLink *link,
                                    struct flatTopoLink *flatLink) {
  link->type = flatLink->type;
  link->bw = flatLink->bw;
  int remNodeIdx = flatLink->remNodeIdx;
  int remNodeType = flatLink->remNodeType;
  sdcclTopoNode *remNode = &(topoServer->nodes[remNodeType].nodes[remNodeIdx]);
  link->remNode = remNode;
  return sdcclSuccess;
}

static sdcclResult_t flattenNode(struct sdcclTopoServer *topoServer,
                                  struct sdcclTopoNode *node,
                                  struct flatTopoNode *flatNode) {
  flatNode->type = node->type;
  flatNode->id = node->id;
  flatNode->nlinks = node->nlinks;
  if (node->type == APU) {
    flatNode->apu.dev = node->apu.dev;
    flatNode->apu.rank = node->apu.rank;
    flatNode->apu.vendor = node->apu.vendor;
  } else if (node->type == CPU) {
    flatNode->cpu.arch = node->cpu.arch;
    flatNode->cpu.vendor = node->cpu.vendor;
    flatNode->cpu.model = node->cpu.model;
  } else if (node->type == PCI) {
    flatNode->pci.device = node->pci.device;
  } else if (node->type == NET) {
    flatNode->net.dev = node->net.dev;
    // flatNode->net.asic = node->net.asic;
    flatNode->net.guid = node->net.guid;
    flatNode->net.port = node->net.port;
    flatNode->net.bw = node->net.bw;
    flatNode->net.latency = node->net.latency;
    flatNode->net.maxConn = node->net.maxConn;
  }
  return sdcclSuccess;
}

static sdcclResult_t unflattenNode(struct sdcclTopoServer *topoServer,
                                    struct sdcclTopoNode *node,
                                    struct flatTopoNode *flatNode) {
  node->type = flatNode->type;
  node->id = flatNode->id;
  node->nlinks = flatNode->nlinks;
  if (node->type == APU) {
    node->apu.dev = flatNode->apu.dev;
    node->apu.rank = flatNode->apu.rank;
    node->apu.vendor = flatNode->apu.vendor;
  } else if (node->type == CPU) {
    node->cpu.arch = flatNode->cpu.arch;
    node->cpu.vendor = flatNode->cpu.vendor;
    node->cpu.model = flatNode->cpu.model;
  } else if (node->type == PCI) {
    node->pci.device = flatNode->pci.device;
  } else if (node->type == NET) {
    node->net.dev = flatNode->net.dev;
    node->net.guid = flatNode->net.guid;
    node->net.port = flatNode->net.port;
    node->net.bw = flatNode->net.bw;
    node->net.latency = flatNode->net.latency;
    node->net.maxConn = flatNode->net.maxConn;
  }
  return sdcclSuccess;
}

static sdcclResult_t flattenNodeSet(struct sdcclTopoServer *topoServer,
                                     struct sdcclTopoNodeSet *nodeSet,
                                     struct flatTopoNodeSet *flatNodeSet) {
  flatNodeSet->count = nodeSet->count;
  for (int n = 0; n < flatNodeSet->count; n++) {
    SDCCLCHECK(
        flattenNode(topoServer, &nodeSet->nodes[n], &flatNodeSet->nodes[n]));
  }
  return sdcclSuccess;
}

static sdcclResult_t unflattenNodeSet(struct sdcclTopoServer *topoServer,
                                       struct sdcclTopoNodeSet *nodeSet,
                                       struct flatTopoNodeSet *flatNodeSet) {
  nodeSet->count = flatNodeSet->count;
  for (int n = 0; n < nodeSet->count; n++) {
    SDCCLCHECK(
        unflattenNode(topoServer, &nodeSet->nodes[n], &flatNodeSet->nodes[n]));
  }
  return sdcclSuccess;
}

static sdcclResult_t flattenTopoServer(struct sdcclTopoServer *topoServer,
                                        struct flatTopoServer *flatTopo) {
  flatTopo->serverId = topoServer->serverId;
  INFO(SDCCL_GRAPH, "FLATTEN_SERVER: serverId = [%d]", flatTopo->serverId);
  flatTopo->nHosts = topoServer->nHosts;
  INFO(SDCCL_GRAPH, "FLATTEN_SERVER: nHosts = [%d]", flatTopo->nHosts);
  for (int h = 0; h < topoServer->nHosts; h++) {
    flatTopo->hostHashes[h] = topoServer->hostHashes[h];
  }

  // flatten node set
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    INFO(SDCCL_GRAPH, "FLATTEN_SERVER: start flattening node set of type [%d]",
         t);
    SDCCLCHECK(
        flattenNodeSet(topoServer, &topoServer->nodes[t], &flatTopo->nodes[t]));
  }
  // need to flatten all nodes first before flattening links
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    for (int n = 0; n < topoServer->nodes[t].count; n++) {
      for (int l = 0; l < topoServer->nodes[t].nodes[n].nlinks; l++) {
        struct sdcclTopoLink *link = &topoServer->nodes[t].nodes[n].links[l];
        struct flatTopoLink *flatLink = &flatTopo->nodes[t].nodes[n].links[l];
        SDCCLCHECK(flattenLink(topoServer, link, flatLink));
      }
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t unflattenTopoServer(struct sdcclTopoServer *topoServer,
                                          struct flatTopoServer *flatTopo) {
  topoServer->serverId = flatTopo->serverId;
  topoServer->nHosts = flatTopo->nHosts;
  INFO(SDCCL_GRAPH, "UNFLATTEN_SERVER: assigning host hashes");
  for (int h = 0; h < topoServer->nHosts; h++) {
    topoServer->hostHashes[h] = flatTopo->hostHashes[h];
  }

  // unflatten node set
  INFO(SDCCL_GRAPH, "UNFLATTEN_SERVER: start unflattening node set");
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    SDCCLCHECK(unflattenNodeSet(topoServer, &topoServer->nodes[t],
                                 &flatTopo->nodes[t]));
  }

  // need to unflatten all nodes first before flattening links
  INFO(SDCCL_GRAPH, "UNFLATTEN_SERVER: start unflattening links");
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    for (int n = 0; n < flatTopo->nodes[t].count; n++) {
      for (int l = 0; l < flatTopo->nodes[t].nodes[n].nlinks; l++) {
        struct sdcclTopoLink *link = &topoServer->nodes[t].nodes[n].links[l];
        struct flatTopoLink *flatLink = &flatTopo->nodes[t].nodes[n].links[l];
        SDCCLCHECK(unflattenLink(topoServer, link, flatLink));
      }
    }
  }

  return sdcclSuccess;
}

static sdcclResult_t
sdcclTopoReorderServerId(struct flatTopoServer *flatTopoServer, int nRanks) {
  // get all host hashes
  std::map<uint64_t, int> hostHashToServerId;
  int serverId = 0;
  int nHosts = 0;
  for (int i = 0; i < nRanks; i++) {
    // get host hash of server
    uint64_t hostHash =
        flatTopoServer[i].hostHashes[flatTopoServer[i].serverId];
    auto it = hostHashToServerId.find(hostHash);
    if (it == hostHashToServerId.end()) {
      // assign new serverId
      flatTopoServer[i].serverId = serverId;
      // if we haven't seen this host hash before, add it to the map
      hostHashToServerId[hostHash] = serverId;
      serverId++;
      nHosts++;
    } else {
      // if we have seen this host hash before, reorder serverId
      flatTopoServer[i].serverId = it->second;
    }
  }
  for (int i = 0; i < nRanks; i++) {
    // clear original host hash array
    memset(flatTopoServer[i].hostHashes, 0,
           sizeof(uint64_t) * SDCCL_TOPO_MAX_NODES);
    flatTopoServer[i].nHosts = nHosts;
    for (auto it = hostHashToServerId.begin(); it != hostHashToServerId.end();
         ++it) {
      // reorder host hashes
      flatTopoServer[i].hostHashes[it->second] = it->first;
    }
  }
  return sdcclSuccess;
}

// modify nodeIds based on new serverId
static sdcclResult_t sdcclModifyNodeIds(struct sdcclTopoServer *topoServer,
                                          uint64_t serverId) {
  for (int t = 0; t < SDCCL_TOPO_NODE_TYPES; t++) {
    for (int n = 0; n < topoServer->nodes[t].count; n++) {
      auto localId = SDCCL_TOPO_ID_LOCAL_ID(topoServer->nodes[t].nodes[n].id);
      topoServer->nodes[t].nodes[n].id = SDCCL_TOPO_ID(serverId, localId);
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t
fillNetToServerMap(struct sdcclInterServerTopo *interServerTopo,
                   struct sdcclTopoServer *topoServer) {
  struct sdcclTopoServer *server;
  for (int i = 0; i < interServerTopo->numServers; i++) {
    server =
        i == topoServer->serverId ? topoServer : interServerTopo->servers + i;
    for (int n = 0; n < server->nodes[NET].count; n++) {
      INFO(SDCCL_GRAPH,
           "FILL_NET_TO_SERVER_MAP: net guid = [%lx], serverId = [%d]",
           server->nodes[NET].nodes[n].net.guid, i);
      interServerTopo->netToServerMap[server->nodes[NET].nodes[n].net.guid] = i;
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t
getNetNodeFromServers(struct sdcclInterServerTopo *interServerTopo,
                      struct sdcclTopoServer *topoServer, uint64_t guid,
                      sdcclTopoNode **net) {
  int serverId = interServerTopo->netToServerMap.at(guid);
  struct sdcclTopoServer *server = serverId == topoServer->serverId
                                        ? topoServer
                                        : interServerTopo->servers + serverId;
  for (int n = 0; n < server->nodes[NET].count; n++) {
    if (server->nodes[NET].nodes[n].net.guid == guid) {
      *net = server->nodes[NET].nodes + n;
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t getEffectiveBw(struct sdcclInterServerRoute *route,
                                     float *bw) {
  float minBw = std::min(route->localNic->net.bw, route->remoteNic->net.bw);
  for (int i = 0; i < route->switchCount; i++) {
    sdcclSwitch *interSwitch = route->switchInfos + i;
    if (interSwitch->isTop) {
      minBw = std::min(minBw, interSwitch->downBw);
      continue;
    }
    float effBw =
        std::min(interSwitch->downBw, interSwitch->upBw * interSwitch->upLink /
                                          interSwitch->downLink);
    minBw = std::min(minBw, effBw);
  }
  *bw = minBw;
  return sdcclSuccess;
}

static sdcclResult_t
sdcclGetInterServerRouteFromFile(const char *xmlFile,
                                  struct sdcclInterServerTopo *interServerTopo,
                                  struct sdcclTopoServer *topoServer) {
  // Read the XML file
  std::ifstream file(xmlFile);
  if (!file.is_open()) {
    WARN("Unable to open file %s", xmlFile);
    return sdcclInternalError;
  }

  // Read file contents into a string
  std::string xmlContent((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  file.close();

  // Parse the XML
  rapidxml::xml_document<> doc;
  // Make a copy of the string since rapidxml will modify it during parsing
  std::vector<char> xmlCopy(xmlContent.begin(), xmlContent.end());
  xmlCopy.push_back('\0'); // Add null terminator

  doc.parse<0>(&xmlCopy[0]);

  rapidxml::xml_node<> *rootNode = doc.first_node("interserver_route");
  if (!rootNode) {
    WARN("No root node found in interserver_route XML");
    return sdcclInternalError;
  }

  rapidxml::xml_node<> *nicPairsNode = rootNode->first_node("nic_pairs");
  if (!nicPairsNode) {
    WARN("No nic_pairs node found in interserver_route XML");
    return sdcclInternalError;
  }

  for (rapidxml::xml_node<> *pairNode = nicPairsNode->first_node("pair");
       pairNode; pairNode = pairNode->next_sibling("pair")) {
    rapidxml::xml_node<> *nic1Node = pairNode->first_node("nic1");
    rapidxml::xml_node<> *nic2Node = pairNode->first_node("nic2");
    if (!nic1Node || !nic2Node) {
      WARN("Missing nic1 or nic2 node in pair");
      return sdcclInternalError;
    }
    rapidxml::xml_attribute<> *guidNic1 = nic1Node->first_attribute("guid");
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: guidNic1 = %s", guidNic1->value());
    rapidxml::xml_attribute<> *guidNic2 = nic2Node->first_attribute("guid");
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: guidNic2 = %s", guidNic2->value());
    // get the actual net node
    sdcclTopoNode *net1 = nullptr, *net2 = nullptr;
    int serverId1 =
        interServerTopo->netToServerMap.at(strtoul(guidNic1->value(), NULL, 0));
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: serverId1 = %d", serverId1);
    int serverId2 =
        interServerTopo->netToServerMap.at(strtoul(guidNic2->value(), NULL, 0));
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: serverId2 = %d", serverId2);

    struct sdcclInterServerRoute *route;
    struct sdcclInterServerRoute *reverseRoute;
    SDCCLCHECK(
        sdcclCalloc(&route, 1)); // remember to free this when destroying comm
    SDCCLCHECK(sdcclCalloc(&reverseRoute, 1));
    SDCCLCHECK(getNetNodeFromServers(interServerTopo, topoServer,
                                      strtoul(guidNic1->value(), NULL, 0),
                                      &net1));
    SDCCLCHECK(getNetNodeFromServers(interServerTopo, topoServer,
                                      strtoul(guidNic2->value(), NULL, 0),
                                      &net2));
    route->localNic = net1;
    route->remoteNic = net2;
    reverseRoute->localNic = net2;
    reverseRoute->remoteNic = net1;

    // parse interswitch
    rapidxml::xml_node<> *interSwitchNode = pairNode->first_node("interSwitch");
    if (!interSwitchNode) {
      WARN("No interSwitch node found in pair");
      return sdcclInternalError;
    }
    rapidxml::xml_attribute<> *countAttr =
        interSwitchNode->first_attribute("count");
    if (!countAttr) {
      WARN("No count attribute found in interSwitch");
      return sdcclInternalError;
    }
    route->switchCount = strtol(countAttr->value(), NULL, 0);
    reverseRoute->switchCount = route->switchCount;
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: switchCount = %d",
         route->switchCount);
    int switchIdx = 0;
    for (rapidxml::xml_node<> *switchNode =
             interSwitchNode->first_node("switch");
         switchNode;
         switchNode = switchNode->next_sibling("switch"), switchIdx++) {
      sdcclSwitch *interSwitch = route->switchInfos + switchIdx;
      // we don't record interSwitch info for reverseRoute to save space
      // also, interswitch info is only used to compute route bandwidth
      rapidxml::xml_attribute<> *downBwAttr =
          switchNode->first_attribute("downBw");
      rapidxml::xml_attribute<> *upBwAttr = switchNode->first_attribute("upBw");
      rapidxml::xml_attribute<> *upLinkAttr =
          switchNode->first_attribute("upLink");
      rapidxml::xml_attribute<> *downLinkAttr =
          switchNode->first_attribute("downLink");
      rapidxml::xml_attribute<> *isTopAttr =
          switchNode->first_attribute("isTop");
      interSwitch->downBw = strtof(downBwAttr->value(), NULL);
      interSwitch->upBw = strtof(upBwAttr->value(), NULL);
      interSwitch->isTop = strtol(isTopAttr->value(), NULL, 0);
      interSwitch->upLink = strtol(upLinkAttr->value(), NULL,
                                   0); // used to compute oversubscription ratio
      interSwitch->downLink =
          strtol(downLinkAttr->value(), NULL,
                 0); // used to compute oversubscription ratio
      INFO(SDCCL_GRAPH,
           "INTERSERVER_ROUTE: interSwitch[%d]: downBw = %f, upBw = %f, isTop "
           "= %d, upLink = %d, downLink = %d",
           switchIdx, interSwitch->downBw, interSwitch->upBw,
           interSwitch->isTop, interSwitch->upLink, interSwitch->downLink);
    }
    // get effective bw
    float effectiveBw;
    SDCCLCHECK(getEffectiveBw(route, &effectiveBw));
    route->interBw = effectiveBw;
    reverseRoute->interBw = effectiveBw;
    INFO(SDCCL_GRAPH, "INTERSERVER_ROUTE: effectiveBw = %f", effectiveBw);
    interServerTopo
        ->routeMap[route->localNic->net.guid][route->remoteNic->net.guid] =
        route;
    interServerTopo->routeMap[reverseRoute->localNic->net.guid]
                             [reverseRoute->remoteNic->net.guid] = reverseRoute;
  }
  return sdcclSuccess;
}

sdcclResult_t
sdcclGetInterServerTopo(struct sdcclHeteroComm *comm,
                         struct sdcclInterServerTopo **interServerTopo,
                         struct sdcclTopoServer *topoServer) {
  auto ret = sdcclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  uint64_t currRankHostHash = topoServer->hostHashes[topoServer->serverId];
  // SDCCLCHECK(sdcclCalloc(interServerTopo, 1));
  *interServerTopo = new sdcclInterServerTopo(); // remember to delete this
                                                  // when destroying comm
  sdcclInterServerTopo *interServer = *interServerTopo;
  flatTopoServer *flatServerData;
  SDCCLCHECK(sdcclCalloc(&flatServerData, nRanks));
  // we need to flatten topoServer first to remove all pointer types in the
  // structure before copying and trasferring it to other ranks
  SDCCLCHECK(flattenTopoServer(topoServer, flatServerData + rank));
  SDCCLCHECK(bootstrapAllGather(comm->bootstrap, (void *)flatServerData,
                                 sizeof(flatTopoServer)));
  SDCCLCHECK(bootstrapBarrier(comm->bootstrap, rank, nRanks, 0));

  // reorder serverId
  SDCCLCHECK(sdcclTopoReorderServerId(flatServerData, nRanks));

  // get unique flatServers
  std::map<int, flatTopoServer *> flatServerMap;
  flatServerMap[flatServerData[0].serverId] = &flatServerData[0];
  int serverCount = 1;
  for (int i = 1; i < nRanks; i++) {
    auto it = flatServerMap.find(flatServerData[i].serverId);
    if (it != flatServerMap.end()) {
      continue;
    }
    flatServerMap[flatServerData[i].serverId] = &flatServerData[i];
    serverCount++;
  }
  // unflatten the flatServers to topoServers
  sdcclTopoServer *topoServers;
  SDCCLCHECK(sdcclCalloc(&topoServers, serverCount));
  int i = 0;
  for (auto it = flatServerMap.begin(); it != flatServerMap.end(); ++it, i++) {
    flatTopoServer *server = it->second;
    if (server->hostHashes[server->serverId] == currRankHostHash) {
      // this is the current server, no need to flatten, but neet to change
      // serverId, and node ids
      topoServer->serverId = server->serverId;
      topoServer->nHosts = server->nHosts;
      memcpy(topoServer->hostHashes, server->hostHashes,
             sizeof(uint64_t) * SDCCL_TOPO_MAX_NODES);
      SDCCLCHECK(sdcclModifyNodeIds(topoServer, server->serverId));
      continue;
    }
    SDCCLCHECK(unflattenTopoServer(topoServers + i, server));
    SDCCLCHECK(sdcclModifyNodeIds(topoServers + i, server->serverId));
    // reconstruct paths because we didn't send path info in allgather
    SDCCLCHECK(sdcclTopoComputePaths(topoServers + i, comm));
  }
  interServer->numServers = serverCount;
  INFO(SDCCL_GRAPH, "INTERSERVER_TOPO: numServers = %d", serverCount);
  interServer->servers = topoServers;
  // populate entries of netToServerIdMap
  SDCCLCHECK(fillNetToServerMap(interServer, topoServer));

  // verify final topoServers
  // if (rank == 0) {
  //   for (int i = 0; i < serverCount; i++) {
  //     if (topoServer->serverId == i) {
  //       SDCCLCHECK(sdcclTopoPrint(topoServer));
  //     } else {
  //       SDCCLCHECK(sdcclTopoPrint(topoServers + i));
  //     }
  //   }
  // }
  const char *interserverFile = sdcclGetEnv("SDCCL_INTERSERVER_ROUTE_FILE");
  if (!interserverFile) {
    INFO(SDCCL_ENV, "SDCCL_INTERSERVER_ROUTE_FILE is not set");
    goto exit; // TODO: need to find a way to determine interserver bw if no
               // file is provided
  }
  // parse the interserver route file
  SDCCLCHECK(sdcclGetInterServerRouteFromFile(interserverFile, interServer,
                                                topoServer));

  // record all net guid and serverId mappings
exit:
  free(flatServerData);
  return ret;
}

sdcclResult_t
sdcclTopoGetServerFromRank(int rank, struct sdcclInterServerTopo *interServer,
                            struct sdcclTopoServer *currServer,
                            struct sdcclTopoServer **retServer) {
  for (int i = 0; i < interServer->numServers; i++) {
    struct sdcclTopoServer *server =
        i == currServer->serverId ? currServer : interServer->servers + i;
    for (int n = 0; n < server->nodes[APU].count; n++) {
      if (server->nodes[APU].nodes[n].apu.rank == rank) {
        *retServer = server;
        return sdcclSuccess;
      }
    }
  }
  return sdcclInternalError;
}
