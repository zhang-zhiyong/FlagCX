#include "xml.h"
#include "core.h"
#include "sdccl_common.h"
#include <ctype.h>
#include <fcntl.h>
#include <float.h>
#include <map>
#include <queue>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <cpuid.h>
#endif

/*******************/
/* XML File Parser */
/*******************/
sdcclResult_t xmlGetChar(FILE *file, char *c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return sdcclInternalError;
  }
  return sdcclSuccess;
}

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char *dst, const char *src, const size_t size) {
  for (int i = 0; i < size; i++)
    dst[i] = tolower(src[i]);
}
static sdcclResult_t getPciPath(const char *busId, char **path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath + sizeof("/sys/class/pci_bus/") - 1, busId,
              BUSID_REDUCED_SIZE - 1);
  memcpylower(busPath + sizeof("/sys/class/pci_bus/0000:00/../../") - 1, busId,
              BUSID_SIZE - 1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

sdcclResult_t xmlGetValue(FILE *file, char *value, char *last) {
  char c;
  SDCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      SDCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return sdcclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return sdcclInternalError;
#endif
  }
  int o = 0;
  do {
    SDCCLCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o - 1] = '\0';
  SDCCLCHECK(xmlGetChar(file, last));
  return sdcclSuccess;
}

sdcclResult_t xmlGetToken(FILE *file, char *name, char *value, char *last) {
  char c;
  char *ptr = name;
  int o = 0;
  do {
    SDCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return sdcclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN - 1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return sdcclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o - 1] = '\0';
  *last = c;
  return sdcclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c)                                                     \
  do {                                                                         \
    s[0] = s[1];                                                               \
    s[1] = s[2];                                                               \
    s[2] = c;                                                                  \
  } while (0)
sdcclResult_t xmlSkipComment(FILE *file, char *start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i = 0; i < strlen(start); i++)
    SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return sdcclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return sdcclSuccess;
}

sdcclResult_t xmlGetNode(FILE *file, struct sdcclXmlNode *node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0)
      return sdcclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return sdcclInternalError;
  }
  // Read XML element name
  SDCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    SDCCLCHECK(xmlSkipComment(file, node->name + 3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    SDCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c,
           node->name);
      return sdcclInternalError;
    }
    return sdcclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    SDCCLCHECK(
        xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(SDCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)",
           MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an
      // extra one.
    } else
      a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    SDCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return sdcclInternalError;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetStrFromSys(const char *path, const char *fileName,
                                       char *strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE *file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue + offset, 1, MAX_STR_LEN - offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(SDCCL_GRAPH, "Topology detection : could not read %s, ignoring",
         filePath);
  } else {
    strValue[offset - 1] = '\0';
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoSetAttrFromSys(struct sdcclXmlNode *pciNode,
                                        const char *path, const char *fileName,
                                        const char *attrName) {
  char strValue[MAX_STR_LEN];
  SDCCLCHECK(sdcclTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') {
    SDCCLCHECK(xmlSetAttr(pciNode, attrName, strValue));
  }
  INFO(SDCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName,
       strValue);
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetPciNode(struct sdcclXml *xml, const char *busId,
                                    struct sdcclXmlNode **pciNode) {
  SDCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    SDCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
    SDCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  }

  return sdcclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) {
  return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
          (c >= 'A' && c <= 'F'));
}
int checkBDFFormat(char *bdf) {
  if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.')
    return 0;
  if (isHex(bdf[0]) == 0 || isHex(bdf[1] == 0) || isHex(bdf[2] == 0) ||
      isHex(bdf[3] == 0) || isHex(bdf[5] == 0) || isHex(bdf[6] == 0) ||
      isHex(bdf[8] == 0) || isHex(bdf[9] == 0) || isHex(bdf[11] == 0))
    return 0;
  return 1;
}

// TODO: it would be better if we have a device handle and can call APIs to get
// Apu information using that device handle
sdcclResult_t sdcclTopoGetXmlFromApu(struct sdcclXmlNode *pciNode,
                                       struct sdcclXml *xml,
                                       struct sdcclXmlNode **apuNodeRet) {
  struct sdcclXmlNode *apuNode = NULL;
  SDCCLCHECK(xmlGetSub(pciNode, "apu", &apuNode));
  if (apuNode == NULL) {
    SDCCLCHECK(xmlAddNode(xml, pciNode, "apu", &apuNode));
  }
  // TODO: maybe add vendor information to the xml node in the future
  *apuNodeRet = apuNode;
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetXmlFromCpu(struct sdcclXmlNode *cpuNode,
                                       struct sdcclXml *xml) {
  int index;
  SDCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char *numaId;
    SDCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return sdcclInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    SDCCLCHECK(
        sdcclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  SDCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    SDCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    SDCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    SDCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  SDCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    unsigned unused;
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    SDCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  SDCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId : 4;
        unsigned modelId : 4;
        unsigned familyId : 4;
        unsigned processorType : 2;
        unsigned resv0 : 2;
        unsigned extModelId : 4;
        unsigned extFamilyId : 8;
        unsigned resv1 : 4;
      };
      uint32_t val;
    } cpuid1;
    unsigned unused;
    __cpuid(1, cpuid1.val, unused, unused, unused);
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    SDCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    SDCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetXmlFromSys(struct sdcclXmlNode *pciNode,
                                       struct sdcclXml *xml) {
  const char *busId;
  SDCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char *path = NULL;
  getPciPath(busId, &path);

  if (path) {
    SDCCLCHECK(sdcclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  SDCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path)
      sdcclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  SDCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path)
      sdcclTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  SDCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path)
      sdcclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor",
                               "subsystem_vendor");
  }
  SDCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path)
      sdcclTopoSetAttrFromSys(pciNode, path, "subsystem_device",
                               "subsystem_device");
  }
  sdcclTopoSetAttrFromSys(pciNode, path, "max_link_speed", "link_speed");
  sdcclTopoSetAttrFromSys(pciNode, path, "max_link_width", "link_width");

  struct sdcclXmlNode *parent = pciNode->parent;
  if (parent == NULL) {
    // try to find the parent along the pci path
    if (path) {
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      SDCCLCHECK(sdcclTopoGetStrFromSys(path, "numa_node", numaIdStr));

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper
      // PCI switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      for (parentOffset = strlen(path) - 1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') {
          slashCount++;
          path[parentOffset] = '\0';
          int start = parentOffset - 1;
          while (start > 0 && path[start] != '/')
            start--;
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path + start + 1) == 0) {
            // This a CPU root complex. Create a CPU tag and stop there.
            struct sdcclXmlNode *topNode;
            SDCCLCHECK(xmlFindTag(xml, "system", &topNode));
            SDCCLCHECK(
                xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
            if (parent == NULL) {
              SDCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
              SDCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
              SDCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
            }
          } else if (slashCount == 2) {
            // Continue on the upper PCI switch
            for (int i = strlen(path) - 1; i > 0; i--) {
              if (path[i] == '/') {
                SDCCLCHECK(
                    xmlFindTagKv(xml, "pci", &parent, "busid", path + i + 1));
                if (parent == NULL) {
                  SDCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                  SDCCLCHECK(xmlSetAttr(parent, "busid", path + i + 1));
                }
                break;
              }
            }
          }
        }
        if (parent)
          break;
      }
    } else {
      // No information on /sys, attach GPU to unknown CPU
      SDCCLCHECK(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"));
      if (parent == NULL) {
        struct sdcclXmlNode *topNode;
        SDCCLCHECK(xmlFindTag(xml, "system", &topNode));
        SDCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
        SDCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
        SDCCLCHECK(xmlSetAttr(parent, "numaid", "-1"));
        SDCCLCHECK(sdcclTopoGetXmlFromCpu(parent, xml));
      }
    }
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    int subIndex = parent->nSubs;
    const char *newBusId;
    SDCCLCHECK(xmlGetAttrStr(pciNode, "busid", &newBusId));
    for (int s = 0; s < parent->nSubs; s++) {
      const char *busId;
      SDCCLCHECK(xmlGetAttr(parent->subs[s], "busid", &busId));
      if (busId != NULL && strcmp(newBusId, busId) < 0) {
        subIndex = s;
        break;
      }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      return sdcclInternalError;
    }
    for (int s = parent->nSubs; s > subIndex; s--)
      parent->subs[s] = parent->subs[s - 1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    SDCCLCHECK(sdcclTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    SDCCLCHECK(sdcclTopoGetXmlFromCpu(parent, xml));
  }
  free(path);
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoFillApu(struct sdcclXml *xml, const char *busId,
                                 struct sdcclXmlNode **gpuNode) {
  struct sdcclXmlNode *pciNode;
  INFO(SDCCL_INIT, "creating xml pci node for busId [%s]", busId);
  SDCCLCHECK(sdcclTopoGetPciNode(xml, busId, &pciNode));
  SDCCLCHECK(sdcclTopoGetXmlFromSys(pciNode, xml));
  INFO(SDCCL_INIT, "creating xml apu node for busId [%s]", busId);
  SDCCLCHECK(sdcclTopoGetXmlFromApu(pciNode, xml, gpuNode));
  return sdcclSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
sdcclResult_t sdcclTopoGetSubsystem(const char *sysPath, char *subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char *path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--)
      ;
    strcpy(subSys, path + offset + 1);
    free(path);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoFillNet(struct sdcclXml *xml, const char *pciPath,
                                 const char *netName,
                                 struct sdcclXmlNode **netNode) {
  SDCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  if (*netNode != NULL)
    return sdcclSuccess;

  const char *pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    INFO(SDCCL_INIT, "gettting subsystem for pciPath [%s]", pciSysPath);
    SDCCLCHECK(sdcclTopoGetSubsystem(pciSysPath, subSystem));
    if (strcmp(subSystem, "pci") != 0) {
      INFO(SDCCL_GRAPH,
           "Topology detection: network path %s is not a PCI device (%s). "
           "Attaching to first CPU",
           pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct sdcclXmlNode *parent = NULL;
  if (pciSysPath) {
    INFO(SDCCL_INIT, "getting parent pci node for nic");
    int offset;
    for (offset = strlen(pciSysPath) - 1; pciSysPath[offset] != '/'; offset--)
      ;
    char busId[SDCCL_DEVICE_PCI_BUSID_BUFFER_SIZE];
    strcpy(busId, pciSysPath + offset + 1);
    INFO(SDCCL_INIT, "busId for parent pci node is [%s]", busId);
    SDCCLCHECK(sdcclTopoGetPciNode(xml, busId, &parent));
    SDCCLCHECK(sdcclTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    SDCCLCHECK(xmlFindTag(xml, "cpu", &parent));
  }

  struct sdcclXmlNode *nicNode = NULL;
  SDCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    SDCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  SDCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  SDCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoDumpXmlRec(int indent, FILE *file,
                                    struct sdcclXmlNode *node) {
  for (int i = 0; i < indent; i++)
    fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a = 0; a < node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s = 0; s < node->nSubs; s++) {
      SDCCLCHECK(sdcclTopoDumpXmlRec(indent + 2, file, node->subs[s]));
    }
    for (int i = 0; i < indent; i++)
      fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return sdcclSuccess;
}

typedef sdcclResult_t (*xmlHandlerFunc_t)(FILE *, struct sdcclXml *,
                                           struct sdcclXmlNode *);

struct xmlHandler {
  const char *name;
  xmlHandlerFunc_t func;
};

sdcclResult_t xmlLoadSub(FILE *file, struct sdcclXml *xml,
                          struct sdcclXmlNode *head,
                          struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE)
    return sdcclSuccess;
  while (1) {
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return sdcclInternalError;
    }
    struct sdcclXmlNode *node = xml->nodes + xml->maxIndex;
    memset(node, 0, sizeof(struct sdcclXmlNode));
    SDCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return sdcclInternalError;
      } else {
        // All done
        return sdcclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return sdcclInternalError;
      }
      return sdcclSuccess;
    }
    int found = 0;
    for (int h = 0; h < nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) {
          if (head->nSubs == MAX_SUBS) {
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return sdcclInternalError;
          }
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        SDCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers)
        INFO(SDCCL_GRAPH, "Ignoring element %s", node->name);
      SDCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/
sdcclResult_t sdcclTopoXmlLoadGpu(FILE *file, struct sdcclXml *xml,
                                    struct sdcclXmlNode *head) {
  SDCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoXmlLoadSystem(FILE *file, struct sdcclXml *xml,
                                       struct sdcclXmlNode *head) {
  int version;
  SDCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != SDCCL_TOPO_XML_VERSION) {
    WARN("XML Topology has wrong version %d, %d needed", version,
         SDCCL_TOPO_XML_VERSION);
    return sdcclInvalidUsage;
  }
  const char *name;
  SDCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL)
    INFO(SDCCL_GRAPH, "Loading topology %s", name);
  else
    INFO(SDCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = {{"gpu", sdcclTopoXmlLoadGpu}};
  SDCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoGetXmlFromFile(const char *xmlTopoFile,
                                        struct sdcclXml *xml, int warn) {
  FILE *file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    if (warn) {
      WARN("Could not open XML topology file %s : %s", xmlTopoFile,
           strerror(errno));
    }
    return sdcclSuccess;
  }
  INFO(SDCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = {{"system", sdcclTopoXmlLoadSystem}};
  xml->maxIndex = 0;
  SDCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return sdcclSuccess;
}

sdcclResult_t sdcclTopoDumpXmlToFile(const char *xmlTopoFile,
                                       struct sdcclXml *xml) {
  // TODO: clear file content if file is not empty
  FILE *file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return sdcclSuccess;
  }
  SDCCLCHECK(sdcclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return sdcclSuccess;
}

sdcclResult_t xmlGetApuByIndex(struct sdcclXml *xml, int apu,
                                struct sdcclXmlNode **apuNode) {
  // iterate through all nodes in xml and find the apuNode with logical index ==
  // apu
  for (int i = 0; i < xml->maxIndex; i++) {
    struct sdcclXmlNode *n = xml->nodes + i;
    if (strcmp(n->name, "apu") == 0) {
      int value = 0;
      SDCCLCHECK(xmlGetAttrInt(n, "dev", &value));
      if (value == apu) {
        *apuNode = n;
        return sdcclSuccess;
      }
    }
  }

  return sdcclSuccess;
}

sdcclResult_t xmlFindClosestNetUnderCpu(struct sdcclXml *xml,
                                         struct sdcclXmlNode *apuNode,
                                         struct sdcclXmlNode **retNet) {
  INFO(SDCCL_INIT, "searching for local net node under one cpu node");
  std::queue<struct sdcclXmlNode *> nodeQueue;
  std::map<struct sdcclXmlNode *, bool> visited;
  nodeQueue.push(apuNode);
  visited[apuNode] = true;
  while (!nodeQueue.empty()) {
    struct sdcclXmlNode *node = nodeQueue.front();
    nodeQueue.pop();
    // INFO(SDCCL_INIT, "node name = %s", node->name);
    if (strcmp(node->name, "system") == 0) {
      // do not go through root node, we are searching under one cpu node
      continue;
    }
    if (strcmp(node->name, "net") == 0) {
      // found a net node
      *retNet = node;
      break;
    }
    // push parent if parent is not visited
    if (node->parent && !visited[node->parent]) {
      nodeQueue.push(node->parent);
      visited[node->parent] = true;
    }
    // push children if children are not visited
    for (int i = 0; i < node->nSubs; i++) {
      if (!visited[node->subs[i]]) {
        nodeQueue.push(node->subs[i]);
        visited[node->subs[i]] = true;
      }
    }
  }
  return sdcclSuccess;
}

sdcclResult_t xmlFindClosestNetUnderServer(struct sdcclXml *xml,
                                            struct sdcclXmlNode *apuNode,
                                            struct sdcclXmlNode **retNet) {
  INFO(SDCCL_INIT, "searching for local net node under one server");
  std::queue<struct sdcclXmlNode *> nodeQueue;
  std::map<struct sdcclXmlNode *, bool> visited;
  nodeQueue.push(apuNode);
  visited[apuNode] = true;
  while (!nodeQueue.empty()) {
    struct sdcclXmlNode *node = nodeQueue.front();
    nodeQueue.pop();
    if (strcmp(node->name, "net") == 0) {
      // found a net node
      *retNet = node;
      break;
    }
    // push parent if parent is not visited
    if (node->parent && !visited[node->parent]) {
      nodeQueue.push(node->parent);
      visited[node->parent] = true;
    }
    // push children if children are not visited
    for (int i = 0; i < node->nSubs; i++) {
      if (!visited[node->subs[i]]) {
        nodeQueue.push(node->subs[i]);
        visited[node->subs[i]] = true;
      }
    }
  }
  return sdcclSuccess;
}