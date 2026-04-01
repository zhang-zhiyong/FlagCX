/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "core.h"
#include "sdccl_common.h"
#include "sdccl_net.h"
#include "ib_common.h"
#include "ib_retrans.h"
#include "ibvwrap.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "timer.h"
#include "utils.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

char sdcclIbIfName[MAX_IF_NAME_SIZE + 1];
union sdcclSocketAddress sdcclIbIfAddr;

SDCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
SDCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
SDCCL_PARAM(IbTimeout, "IB_TIMEOUT", 18);
SDCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
SDCCL_PARAM(IbPkey, "IB_PKEY", 0);
SDCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
SDCCL_PARAM(IbSl, "IB_SL", 0);
SDCCL_PARAM(IbTc, "IB_TC", 0);
SDCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
SDCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
SDCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
SDCCL_PARAM(IbDisable, "IB_DISABLE", 0);
SDCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
SDCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);
SDCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

int sdcclNMergedIbDevs = -1;
int sdcclNIbDevs = -1;

// Define global arrays
struct sdcclIbMergedDev sdcclIbMergedDevs[MAX_IB_VDEVS];
struct sdcclIbDev sdcclIbDevs[MAX_IB_DEVS];
pthread_mutex_t sdcclIbLock = PTHREAD_MUTEX_INITIALIZER;
int sdcclIbRelaxedOrderingEnabled = 0;

pthread_t sdcclIbAsyncThread;
void *sdcclIbAsyncThreadMain(void *args) {
  struct sdcclIbDev *dev = (struct sdcclIbDev *)args;
  while (1) {
    struct ibv_async_event event;
    if (sdcclSuccess != sdcclWrapIbvGetAsyncEvent(dev->context, &event)) {
      break;
    }
    char *str;
    if (sdcclSuccess != sdcclWrapIbvEventTypeStr(&str, event.event_type)) {
      break;
    }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : %s:%d Got async event : %s", dev->devName, dev->portNum,
           str);
    if (sdcclSuccess != sdcclWrapIbvAckAsyncEvent(&event)) {
      break;
    }
  }
  return NULL;
}

sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char *env = sdcclGetEnv("SDCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(SDCCL_ENV, "SDCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

void *envIbAddrRange(sa_family_t af, int *mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char *env = sdcclGetEnv("SDCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(SDCCL_ENV, "SDCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = {0};
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/") + 1;
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr - 1) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    WARN("NET/IB: Ip address '%s' is invalid for family %s, ignoring address",
         addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
         *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
         *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

sa_family_t getGidAddrFamily(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) |
                       (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
      (a->s6_addr32[0] == htonl(0xff0e0000) &&
       ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

bool matchGidAddrPrefix(sa_family_t af, void *prefix, int prefixlen,
                        union ibv_gid *gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;
  ;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

bool configuredGid(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) ||
      ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

bool linkLocalGid(union ibv_gid *gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

bool validGid(union ibv_gid *gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

sdcclResult_t sdcclIbRoceGetVersionNum(const char *deviceName, int portNum,
                                         int gidIndex, int *version) {
  char gidRoceVerStr[16] = {0};
  char roceTypePath[PATH_MAX] = {0};
  sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
          deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    return sdcclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    return sdcclSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
        strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclUpdateGidIndex(struct ibv_context *context,
                                    uint8_t portNum, sa_family_t af,
                                    void *prefix, int prefixlen, int roceVer,
                                    int gidIndexCandidate, int *gidIndex) {
  union ibv_gid gid, gidCandidate;
  SDCCLCHECK(sdcclWrapIbvQueryGid(context, portNum, *gidIndex, &gid));
  SDCCLCHECK(sdcclWrapIbvQueryGid(context, portNum, gidIndexCandidate,
                                    &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet =
      matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam &&
      gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) ||
        !gidCandidateMatchSubnet) {
      return sdcclSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate;
    const char *deviceName = sdcclWrapIbvGetDeviceName(context->device);
    SDCCLCHECK(sdcclIbRoceGetVersionNum(deviceName, portNum, *gidIndex,
                                          &gidRoceVerNum));
    SDCCLCHECK(sdcclIbRoceGetVersionNum(
        deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) &&
        gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIbGetGidIndex(struct ibv_context *context, uint8_t portNum,
                                   int gidTblLen, int *gidIndex) {
  *gidIndex = sdcclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return sdcclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = sdcclParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    SDCCLCHECK(sdcclUpdateGidIndex(context, portNum, userAddrFamily, prefix,
                                     prefixlen, userRoceVersion, gidIndexNext,
                                     gidIndex));
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIbGetPciPath(char *devName, char **path, int *realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char *p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p) - 1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (sdcclParamIbMergeVfs())
      p[strlen(p) - 3] = p[strlen(p) - 4] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d = 0; d < sdcclNIbDevs; d++) {
      if (strcmp(p, sdcclIbDevs[d].pciPath) == 0)
        (*realPort)++;
    }
  }
  *path = p;
  return sdcclSuccess;
}

int ibvWidths[] = {1, 4, 8, 12, 2};
int ibvSpeeds[] = {2500,  /* SDR */
                   5000,  /* DDR */
                   10000, /* QDR */
                   10000, /* QDR */
                   14000, /* FDR */
                   25000, /* EDR */
                   50000, /* HDR */
                   100000 /* NDR */};

int firstBitSet(int val, int max) {
  int i = 0;
  while (i < max && ((val & (1 << i)) == 0))
    i++;
  return i;
}
int sdcclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths) / sizeof(int) - 1)];
}
int sdcclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds) / sizeof(int) - 1)];
}

int sdcclIbRelaxedOrderingCapable(void) {
  int roMode = sdcclParamIbPciRelaxedOrdering();
  sdcclResult_t r = sdcclInternalError;
  if (roMode == 1 || roMode == 2) {
    // Query IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
    r = sdcclWrapIbvRegMrIova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return r == sdcclInternalError ? 0 : 1;
}

int sdcclIbFindMatchingDev(int dev) {
  for (int i = 0; i < sdcclNMergedIbDevs; i++) {
    if (sdcclIbMergedDevs[i].ndevs < SDCCL_IB_MAX_DEVS_PER_NIC) {
      int compareDev = sdcclIbMergedDevs[i].devs[0];
      if (strcmp(sdcclIbDevs[dev].pciPath, sdcclIbDevs[compareDev].pciPath) ==
              0 &&
          (sdcclIbDevs[dev].guid == sdcclIbDevs[compareDev].guid) &&
          (sdcclIbDevs[dev].link == sdcclIbDevs[compareDev].link)) {
        TRACE(SDCCL_NET,
              "NET/IB: Matched name1=%s pciPath1=%s guid1=0x%lx link1=%u "
              "name2=%s pciPath2=%s guid2=0x%lx link2=%u",
              sdcclIbDevs[dev].devName, sdcclIbDevs[dev].pciPath,
              sdcclIbDevs[dev].guid, sdcclIbDevs[dev].link,
              sdcclIbDevs[compareDev].devName,
              sdcclIbDevs[compareDev].pciPath, sdcclIbDevs[compareDev].guid,
              sdcclIbDevs[compareDev].link);
        return i;
      }
    }
  }

  return sdcclNMergedIbDevs;
}

sdcclResult_t sdcclIbInit() {
  sdcclResult_t ret;
  if (sdcclParamIbDisable())
    return sdcclInternalError;
  static int shownIbHcaEnv = 0;
  if (sdcclWrapIbvSymbols() != sdcclSuccess) {
    return sdcclInternalError;
  }

  if (sdcclNIbDevs == -1) {
    pthread_mutex_lock(&sdcclIbLock);
    sdcclWrapIbvForkInit();
    if (sdcclNIbDevs == -1) {
      sdcclNIbDevs = 0;
      sdcclNMergedIbDevs = 0;
      if (sdcclFindInterfaces(sdcclIbIfName, &sdcclIbIfAddr,
                               MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = sdcclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device **devices;

      // Check if user defined which IB device:port to use
      char *userIbEnv = getenv("SDCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0)
        INFO(SDCCL_NET | SDCCL_ENV, "SDCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot)
        userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact)
        userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (sdcclSuccess != sdcclWrapIbvGetDeviceList(&devices, &nIbDevs)) {
        ret = sdcclInternalError;
        goto fail;
      }

      for (int d = 0; d < nIbDevs && sdcclNIbDevs < MAX_IB_DEVS; d++) {
        struct ibv_context *context;
        if (sdcclSuccess != sdcclWrapIbvOpenDevice(&context, devices[d]) ||
            context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (sdcclSuccess != sdcclWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (sdcclSuccess != sdcclWrapIbvCloseDevice(context)) {
            ret = sdcclInternalError;
            goto fail;
          }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (sdcclSuccess !=
              sdcclWrapIbvQueryPort(context, port_num, &portAttr)) {
            WARN("NET/IB : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE)
            continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
              portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
            continue;

          // check against user specified HCAs/ports
          if (!(matchIfList(devices[d]->name, port_num, userIfs, nUserIfs,
                            searchExact) ^
                searchNot)) {
            continue;
          }
          pthread_mutex_init(&sdcclIbDevs[sdcclNIbDevs].lock, NULL);
          sdcclIbDevs[sdcclNIbDevs].device = d;
          sdcclIbDevs[sdcclNIbDevs].guid = devAttr.sys_image_guid;
          sdcclIbDevs[sdcclNIbDevs].portAttr = portAttr;
          sdcclIbDevs[sdcclNIbDevs].portNum = port_num;
          sdcclIbDevs[sdcclNIbDevs].link = portAttr.link_layer;
          sdcclIbDevs[sdcclNIbDevs].speed =
              sdcclIbSpeed(portAttr.active_speed) *
              sdcclIbWidth(portAttr.active_width);
          sdcclIbDevs[sdcclNIbDevs].context = context;
          sdcclIbDevs[sdcclNIbDevs].pdRefs = 0;
          sdcclIbDevs[sdcclNIbDevs].pd = NULL;
          strncpy(sdcclIbDevs[sdcclNIbDevs].devName, devices[d]->name,
                  MAXNAMESIZE);
          SDCCLCHECK(
              sdcclIbGetPciPath(sdcclIbDevs[sdcclNIbDevs].devName,
                                 &sdcclIbDevs[sdcclNIbDevs].pciPath,
                                 &sdcclIbDevs[sdcclNIbDevs].realPort));
          sdcclIbDevs[sdcclNIbDevs].maxQp = devAttr.max_qp;
          sdcclIbDevs[sdcclNIbDevs].mrCache.capacity = 0;
          sdcclIbDevs[sdcclNIbDevs].mrCache.population = 0;
          sdcclIbDevs[sdcclNIbDevs].mrCache.slots = NULL;

          // Enable ADAPTIVE_ROUTING by default on IB networks
          // But allow it to be overloaded by an env parameter
          sdcclIbDevs[sdcclNIbDevs].ar =
              (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (sdcclParamIbAdaptiveRouting() != -2)
            sdcclIbDevs[sdcclNIbDevs].ar = sdcclParamIbAdaptiveRouting();

          TRACE(SDCCL_NET,
                "NET/IB: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d",
                d, devices[d]->name, devices[d]->dev_name,
                sdcclIbDevs[sdcclNIbDevs].portNum,
                portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                 : "RoCE",
                sdcclIbDevs[sdcclNIbDevs].speed, context,
                sdcclIbDevs[sdcclNIbDevs].pciPath,
                sdcclIbDevs[sdcclNIbDevs].ar);

          pthread_create(&sdcclIbAsyncThread, NULL, sdcclIbAsyncThreadMain,
                         sdcclIbDevs + sdcclNIbDevs);
          sdcclSetThreadName(sdcclIbAsyncThread, "SDCCL IbAsync %2d",
                              sdcclNIbDevs);
          pthread_detach(sdcclIbAsyncThread); // will not be pthread_join()'d

          int mergedDev = sdcclNMergedIbDevs;
          if (sdcclParamIbMergeNics()) {
            mergedDev = sdcclIbFindMatchingDev(sdcclNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if
          // there's only one dev inside)
          if (mergedDev == sdcclNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IB device
            sdcclIbMergedDevs[mergedDev].ndevs = 1;
            sdcclIbMergedDevs[mergedDev].devs[0] = sdcclNIbDevs;
            sdcclNMergedIbDevs++;
            strncpy(sdcclIbMergedDevs[mergedDev].devName,
                    sdcclIbDevs[sdcclNIbDevs].devName, MAXNAMESIZE);
            // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IB device
            int ndevs = sdcclIbMergedDevs[mergedDev].ndevs;
            sdcclIbMergedDevs[mergedDev].devs[ndevs] = sdcclNIbDevs;
            sdcclIbMergedDevs[mergedDev].ndevs++;
            snprintf(sdcclIbMergedDevs[mergedDev].devName +
                         strlen(sdcclIbMergedDevs[mergedDev].devName),
                     MAXNAMESIZE + 1, "+%s",
                     sdcclIbDevs[sdcclNIbDevs].devName);
          }

          // Aggregate speed
          sdcclIbMergedDevs[mergedDev].speed +=
              sdcclIbDevs[sdcclNIbDevs].speed;
          sdcclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && sdcclSuccess != sdcclWrapIbvCloseDevice(context)) {
          ret = sdcclInternalError;
          goto fail;
        }
      }
      if (nIbDevs && (sdcclSuccess != sdcclWrapIbvFreeDeviceList(devices))) {
        ret = sdcclInternalError;
        goto fail;
      };
    }
    if (sdcclNIbDevs == 0) {
      INFO(SDCCL_INIT | SDCCL_NET, "NET/IB : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      sdcclIbRelaxedOrderingEnabled = sdcclIbRelaxedOrderingCapable();
      for (int d = 0; d < sdcclNMergedIbDevs; d++) {
        struct sdcclIbMergedDev *mergedDev = sdcclIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibDev = mergedDev->devs[i];
            snprintf(
                line + strlen(line), 2047 - strlen(line), "[%d] %s:%d/%s%s",
                ibDev, sdcclIbDevs[ibDev].devName, sdcclIbDevs[ibDev].portNum,
                sdcclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                      : "RoCE",
                // Insert comma to delineate
                i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line + strlen(line), 2047 - strlen(line), "}");
        } else {
          int ibDev = mergedDev->devs[0];
          snprintf(
              line + strlen(line), 2047 - strlen(line), " [%d]%s:%d/%s", ibDev,
              sdcclIbDevs[ibDev].devName, sdcclIbDevs[ibDev].portNum,
              sdcclIbDevs[ibDev].link == IBV_LINK_LAYER_INFINIBAND ? "IB"
                                                                    : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN + 1];
      INFO(SDCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line,
           sdcclIbRelaxedOrderingEnabled ? "[RO]" : "", sdcclIbIfName,
           sdcclSocketToString(&sdcclIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&sdcclIbLock);
  }
  return sdcclSuccess;
fail:
  pthread_mutex_unlock(&sdcclIbLock);
  return ret;
}

const char *reqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet"};

static void sdcclIbAddEvent(struct sdcclIbRequest *req, int devIndex,
                             struct sdcclIbNetCommDevBase *base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

sdcclResult_t sdcclIbInitCommDevBase(int ibDevN,
                                       struct sdcclIbNetCommDevBase *base) {
  base->ibDevN = ibDevN;
  sdcclIbDev *ibDev = sdcclIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == ibDev->pdRefs++) {
    sdcclResult_t res;
    SDCCLCHECKGOTO(sdcclWrapIbvAllocPd(&ibDev->pd, ibDev->context), res,
                    failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibDev->lock);
      return res;
    }
  }
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for
  // the Recv).
  SDCCLCHECK(sdcclWrapIbvCreateCq(
      &base->cq, ibDev->context, 2 * MAX_REQUESTS * sdcclParamIbQpsPerConn(),
      NULL, NULL, 0));

  return sdcclSuccess;
}

sdcclResult_t sdcclIbDestroyBase(struct sdcclIbNetCommDevBase *base) {
  sdcclResult_t res;

  // Poll any remaining completions before destroying CQ
  if (base->cq) {
    struct ibv_wc wcs[64];
    int nCqe = 0;
    // Poll multiple times to drain all pending completions
    for (int i = 0; i < 16; i++) {
      sdcclWrapIbvPollCq(base->cq, 64, wcs, &nCqe);
      if (nCqe == 0)
        break;
    }
  }

  SDCCLCHECK(sdcclWrapIbvDestroyCq(base->cq));

  pthread_mutex_lock(&sdcclIbDevs[base->ibDevN].lock);
  if (0 == --sdcclIbDevs[base->ibDevN].pdRefs) {
    sdcclResult_t pd_result =
        sdcclWrapIbvDeallocPd(sdcclIbDevs[base->ibDevN].pd);
    if (pd_result != sdcclSuccess) {
      // Log but don't fail - PD deallocation errors are often non-fatal
      // (e.g., "Device or resource busy" when there are still resources using
      // the PD)
      if (sdcclDebugNoWarn == 0)
        INFO(SDCCL_ALL,
             "Failed to deallocate PD: %d (non-fatal, may have remaining "
             "resources)",
             pd_result);
      res = sdcclSuccess; // Continue cleanup even if PD deallocation fails
    } else {
      res = sdcclSuccess;
    }
  } else {
    res = sdcclSuccess;
  }
  pthread_mutex_unlock(&sdcclIbDevs[base->ibDevN].lock);
  return res;
}

sdcclResult_t sdcclIbCreateQp(uint8_t ib_port,
                                struct sdcclIbNetCommDevBase *base,
                                int accessFlags, struct sdcclIbQp *qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2 * MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data =
      sdcclParamIbUseInline() ? sizeof(struct sdcclIbSendFifo) : 0;
  SDCCLCHECK(sdcclWrapIbvCreateQp(&qp->qp, base->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = sdcclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = accessFlags;
  SDCCLCHECK(sdcclWrapIbvModifyQp(qp->qp, &qpAttr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbRtrQp(struct ibv_qp *qp, uint8_t sGidIndex,
                             uint32_t dest_qp_num,
                             struct sdcclIbDevInfo *info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->linkLayer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = sdcclParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = sdcclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ibPort;
  SDCCLCHECK(sdcclWrapIbvModifyQp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbRtsQp(struct ibv_qp *qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = sdcclParamIbTimeout();
  qpAttr.retry_cnt = sdcclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  SDCCLCHECK(sdcclWrapIbvModifyQp(
      qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbListen(int dev, void *opaqueHandle, void **listenComm) {
  struct sdcclIbListenComm *comm;
  SDCCLCHECK(sdcclCalloc(&comm, 1));
  struct sdcclIbHandle *handle = (struct sdcclIbHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct sdcclIbHandle));
  comm->dev = dev;
  handle->magic = SDCCL_SOCKET_MAGIC;
  SDCCLCHECK(sdcclSocketInit(&comm->sock, &sdcclIbIfAddr, handle->magic,
                               sdcclSocketTypeNetIb, NULL, 1));
  SDCCLCHECK(sdcclSocketListen(&comm->sock));
  SDCCLCHECK(sdcclSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbConnect(int dev, void *opaqueHandle, void **sendComm) {
  struct sdcclIbHandle *handle = (struct sdcclIbHandle *)opaqueHandle;
  struct sdcclIbCommStage *stage = &handle->stage;
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)stage->comm;
  int ready;
  *sendComm = NULL;

  if (stage->state == sdcclIbCommStateConnect)
    goto ib_connect_check;
  if (stage->state == sdcclIbCommStateSend)
    goto ib_send;
  if (stage->state == sdcclIbCommStateConnecting)
    goto ib_connect;
  if (stage->state == sdcclIbCommStateConnected)
    goto ib_send_ready;
  if (stage->state != sdcclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return sdcclInternalError;
  }

  SDCCLCHECK(sdcclIbMalloc((void **)&comm, sizeof(struct sdcclIbSendComm)));
  SDCCLCHECK(sdcclSocketInit(&comm->base.sock, &handle->connectAddr,
                               handle->magic, sdcclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = sdcclIbCommStateConnect;
  SDCCLCHECK(sdcclSocketConnect(&comm->base.sock));

ib_connect_check:
  /* since sdcclSocketConnect is async, we must check if connection is complete
   */
  SDCCLCHECK(sdcclSocketReady(&comm->base.sock, &ready));
  if (!ready)
    return sdcclSuccess;

  // IB Setup
  struct sdcclIbMergedDev *mergedDev;
  mergedDev = sdcclIbMergedDevs + dev;
  comm->base.ndevs = mergedDev->ndevs;
  comm->base.nqps = sdcclParamIbQpsPerConn() *
                    comm->base.ndevs; // We must have at least 1 qp per-device
  comm->base.isSend = true;

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDevN = mergedDev->devs[i];
    SDCCLCHECK(sdcclIbInitCommDevBase(ibDevN, &comm->devs[i].base));
    comm->ar = comm->ar &&
               sdcclIbDevs[dev]
                   .ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  struct sdcclIbConnectionMetadata meta;
  meta.ndevs = comm->base.ndevs;

  // IBRC retransmission: default disabled, can be enabled via
  // SDCCL_IB_RETRANS_ENABLE=1
  meta.retransEnabled = sdcclParamIbRetransEnable() ? 1 : 0;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    sdcclIbSendCommDev *commDev = comm->devs + devIndex;
    sdcclIbDev *ibDev = sdcclIbDevs + commDev->base.ibDevN;
    SDCCLCHECK(
        sdcclIbCreateQp(ibDev->portNum, &commDev->base,
                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC,
                         comm->base.qps + q));
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // Query ece capabilities (enhanced connection establishment)
    SDCCLCHECK(sdcclWrapIbvQueryEce(comm->base.qps[q].qp, &meta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));
    devIndex = (devIndex + 1) % comm->base.ndevs;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    sdcclIbSendCommDev *commDev = comm->devs + i;
    sdcclIbDev *ibDev = sdcclIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    sdcclIbDevInfo *devInfo = meta.devs + i;
    devInfo->ibPort = ibDev->portNum;
    devInfo->mtu = ibDev->portAttr.active_mtu;
    devInfo->lid = ibDev->portAttr.lid;

    // Prepare my fifo
    SDCCLCHECK(
        sdcclWrapIbvRegMr(&commDev->fifoMr, commDev->base.pd, comm->fifo,
                           sizeof(struct sdcclIbSendFifo) * MAX_REQUESTS *
                               SDCCL_NET_IB_MAX_RECVS,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ));
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // Prepare putSignalScratchpad (for RDMA Atomic result)
    SDCCLCHECK(sdcclWrapIbvRegMr(&commDev->putSignalScratchpadMr,
                                   commDev->base.pd, &comm->putSignalScratchpad,
                                   sizeof(comm->putSignalScratchpad),
                                   IBV_ACCESS_LOCAL_WRITE));

    // RoCE support
    devInfo->linkLayer = commDev->base.gidInfo.linkLayer =
        ibDev->portAttr.link_layer;
    if (devInfo->linkLayer == IBV_LINK_LAYER_ETHERNET) {
      SDCCLCHECK(sdcclIbGetGidIndex(ibDev->context, ibDev->portNum,
                                      ibDev->portAttr.gid_tbl_len,
                                      &commDev->base.gidInfo.localGidIndex));
      SDCCLCHECK(sdcclWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                        commDev->base.gidInfo.localGidIndex,
                                        &commDev->base.gidInfo.localGid));
      devInfo->spn = commDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo->iid = commDev->base.gidInfo.localGid.global.interface_id;
    } else {
      commDev->base.gidInfo.localGidIndex = 0;
      memset(&commDev->base.gidInfo.localGid, 0, sizeof(union ibv_gid));
    }

    // Create control QP for retransmission if enabled
    if (meta.retransEnabled) {
      SDCCLCHECK(sdcclIbCreateCtrlQp(ibDev->context, commDev->base.pd,
                                       ibDev->portNum, &commDev->ctrlQp));
      meta.ctrlQpn[i] = commDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibDev->portAttr.lid;
      meta.ctrlGid[i] = commDev->base.gidInfo.localGid;

      size_t ackBufSize =
          (sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING) *
          SDCCL_IB_ACK_BUF_COUNT;
      commDev->ackBuffer = malloc(ackBufSize);
      SDCCLCHECK(sdcclWrapIbvRegMr(&commDev->ackMr, commDev->base.pd,
                                     commDev->ackBuffer, ackBufSize,
                                     IBV_ACCESS_LOCAL_WRITE));

      TRACE(SDCCL_NET,
            "Send: Created control QP for dev %d: qpn=%u, link_layer=%d, "
            "lid=%u, gid=%lx:%lx",
            i, commDev->ctrlQp.qp->qp_num, devInfo->linkLayer, meta.ctrlLid[i],
            (unsigned long)meta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)meta.ctrlGid[i].global.interface_id);
    }

    if (devInfo->linkLayer == IBV_LINK_LAYER_INFINIBAND) { // IB
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(SDCCL_NET,
               "NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d "
               "fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "SDCCL MergedDev" : "SDCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, devInfo->lid, devInfo->fifoRkey,
               commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(SDCCL_NET,
               "NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d "
               "query_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
               "comp_mask=0x%x} GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "SDCCL MergedDev" : "SDCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, meta.qpInfo[q].eceSupported,
               meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options,
               meta.qpInfo[q].ece.comp_mask,
               (int64_t)commDev->base.gidInfo.localGidIndex, devInfo->spn,
               devInfo->iid, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    }
  }
  meta.fifoAddr = (uint64_t)comm->fifo;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = sdcclIbCommStateSend;
  stage->offset = 0;
  SDCCLCHECK(sdcclIbMalloc((void **)&stage->buffer, sizeof(meta)));

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_SEND, &comm->base.sock,
                                   stage->buffer, sizeof(meta),
                                   &stage->offset));
  if (stage->offset != sizeof(meta))
    return sdcclSuccess;

  stage->state = sdcclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct sdcclIbConnectionMetadata remMeta;
  SDCCLCHECK(
      sdcclSocketProgress(SDCCL_SOCKET_RECV, &comm->base.sock, stage->buffer,
                           sizeof(sdcclIbConnectionMetadata), &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return sdcclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(sdcclIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;
  if (comm->base.nRemDevs != comm->base.ndevs) {
    mergedDev = sdcclIbMergedDevs + dev;
    WARN("NET/IB : Local mergedDev=%s has a different number of devices=%d as "
         "remoteDev=%s nRemDevs=%d",
         mergedDev->devName, comm->base.ndevs, remMeta.devName,
         comm->base.nRemDevs);
  }

  int linkLayer;
  linkLayer = remMeta.devs[0].linkLayer;
  for (int i = 1; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].linkLayer != linkLayer) {
      WARN("NET/IB : Can't merge net devices with different linkLayer. i=%d "
           "remMeta.ndevs=%d linkLayer=%d rem_linkLayer=%d",
           i, remMeta.ndevs, linkLayer, remMeta.devs[i].linkLayer);
      return sdcclInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id =
        comm->base.remDevs[i].iid;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix =
        comm->base.remDevs[i].spn;

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    SDCCLCHECK(
        sdcclWrapIbvRegMr(comm->remSizesFifo.mrs + i, comm->devs[i].base.pd,
                           &comm->remSizesFifo.elems,
                           sizeof(int) * MAX_REQUESTS * SDCCL_NET_IB_MAX_RECVS,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ));
  }
  comm->base.nRemDevs = remMeta.ndevs;

  // Initialize retransmission state
  SDCCLCHECK(sdcclIbRetransInit(&comm->retrans));
  comm->lastTimeoutCheckUs = 0;
  comm->outstandingSends = 0;
  comm->outstandingRetrans = 0;
  comm->maxOutstanding = sdcclParamIbMaxOutstanding();

  if (comm->retrans.enabled) {
    INFO(SDCCL_NET,
         "NET/IBRC Sender: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         comm->retrans.minRtoUs, comm->retrans.maxRetry,
         comm->retrans.ackInterval);
  } else {
    INFO(SDCCL_NET, "NET/IBRC Sender: Retransmission DISABLED");
  }

  for (int q = 0; q < comm->base.nqps; q++) {
    struct sdcclIbQpInfo *remQpInfo = remMeta.qpInfo + q;
    struct sdcclIbDevInfo *remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    sdcclIbSendCommDev *commDev = comm->devs + devIndex;
    uint8_t gidIndex = commDev->base.gidInfo.localGidIndex;

    struct ibv_qp *qp = comm->base.qps[q].qp;
    if (remQpInfo->eceSupported)
      SDCCLCHECK(
          sdcclWrapIbvSetEce(qp, &remQpInfo->ece, &remQpInfo->eceSupported));

    SDCCLCHECK(sdcclIbRtrQp(qp, gidIndex, remQpInfo->qpn, remDevInfo));
    SDCCLCHECK(sdcclIbRtsQp(qp));
  }

  if (linkLayer == IBV_LINK_LAYER_ETHERNET) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct sdcclIbQp *qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct sdcclIbDev *ibDev = sdcclIbDevs + ibDevN;
      INFO(SDCCL_NET,
           "NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, "
           "vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
           ibDevN, ibDev->portNum, remMeta.qpInfo[q].qpn,
           remMeta.qpInfo[q].eceSupported, remMeta.qpInfo[q].ece.vendor_id,
           remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  // Setup control QP for retransmission if enabled
  if (comm->retrans.enabled && remMeta.retransEnabled) {
    bool allAhSuccess = true;

    for (int i = 0; i < comm->base.ndevs; i++) {
      sdcclIbSendCommDev *commDev = &comm->devs[i];
      sdcclIbDev *ibDev = sdcclIbDevs + commDev->base.ibDevN;

      TRACE(SDCCL_NET,
            "Send: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibDev->portAttr.link_layer);

      sdcclResult_t ah_result = sdcclIbSetupCtrlQpConnection(
          ibDev->context, commDev->base.pd, &commDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibDev->portNum, ibDev->portAttr.link_layer,
          commDev->base.gidInfo.localGidIndex);

      if (ah_result != sdcclSuccess || !commDev->ctrlQp.ah) {
        allAhSuccess = false;
        break;
      }

      size_t bufEntrySize =
          sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING;
      for (int r = 0; r < 32; r++) {
        struct ibv_sge sge;
        sge.addr = (uint64_t)((char *)commDev->ackBuffer + r * bufEntrySize);
        sge.length = bufEntrySize;
        sge.lkey = commDev->ackMr->lkey;

        struct ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.wr_id = r;
        recv_wr.next = NULL;
        recv_wr.sg_list = &sge;
        recv_wr.num_sge = 1;

        struct ibv_recv_wr *bad_wr;
        SDCCLCHECK(
            sdcclWrapIbvPostRecv(commDev->ctrlQp.qp, &recv_wr, &bad_wr));
      }

      TRACE(SDCCL_NET,
            "Control QP ready for dev %d: local_qpn=%u, remote_qpn=%u, posted "
            "32 recv WRs",
            i, commDev->ctrlQp.qp->qp_num, remMeta.ctrlQpn[i]);
    }

    if (!allAhSuccess) {
      comm->retrans.enabled = 0;

      for (int i = 0; i < comm->base.ndevs; i++) {
        if (comm->devs[i].ackMr)
          sdcclWrapIbvDeregMr(comm->devs[i].ackMr);
        if (comm->devs[i].ackBuffer)
          free(comm->devs[i].ackBuffer);
        sdcclIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
      }
    }

    if (allAhSuccess && comm->retrans.enabled) {
      sdcclResult_t mr_result = sdcclWrapIbvRegMr(
          &comm->retransHdrMr, comm->devs[0].base.pd, comm->retransHdrPool,
          sizeof(comm->retransHdrPool), IBV_ACCESS_LOCAL_WRITE);

      if (mr_result != sdcclSuccess || !comm->retransHdrMr) {
        WARN("Failed to register retrans_hdr_mr, disabling retransmission");
        comm->retrans.enabled = 0;
        // Clean up already created resources
        for (int i = 0; i < comm->base.ndevs; i++) {
          if (comm->devs[i].ackMr)
            sdcclWrapIbvDeregMr(comm->devs[i].ackMr);
          if (comm->devs[i].ackBuffer)
            free(comm->devs[i].ackBuffer);
          sdcclIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
        }
      } else {
        TRACE(
            SDCCL_NET,
            "Sender: Initialized SEND retransmission (header pool MR created)");
      }
    }
  }

  comm->base.ready = 1;
  stage->state = sdcclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_SEND, &comm->base.sock,
                                   &comm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return sdcclSuccess;

  free(stage->buffer);
  stage->state = sdcclIbCommStateStart;

  *sendComm = comm;
  return sdcclSuccess;
}

SDCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

sdcclResult_t sdcclIbAccept(void *listenComm, void **recvComm) {
  struct sdcclIbListenComm *lComm = (struct sdcclIbListenComm *)listenComm;
  struct sdcclIbCommStage *stage = &lComm->stage;
  struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)stage->comm;
  int ready;
  *recvComm = NULL;
  // Pre-declare variables because of goto
  struct ibv_srq *srq = NULL;

  if (stage->state == sdcclIbCommStateAccept)
    goto ib_accept_check;
  if (stage->state == sdcclIbCommStateRecv)
    goto ib_recv;
  if (stage->state == sdcclIbCommStateSend)
    goto ib_send;
  if (stage->state == sdcclIbCommStatePendingReady)
    goto ib_recv_ready;
  if (stage->state != sdcclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return sdcclInternalError;
  }

  SDCCLCHECK(sdcclIbMalloc((void **)&rComm, sizeof(struct sdcclIbRecvComm)));
  stage->comm = rComm;
  stage->state = sdcclIbCommStateAccept;
  SDCCLCHECK(sdcclSocketInit(&rComm->base.sock));
  SDCCLCHECK(sdcclSocketAccept(&rComm->base.sock, &lComm->sock));

ib_accept_check:
  SDCCLCHECK(sdcclSocketReady(&rComm->base.sock, &ready));
  if (!ready)
    return sdcclSuccess;

  struct sdcclIbConnectionMetadata remMeta;
  stage->state = sdcclIbCommStateRecv;
  stage->offset = 0;
  SDCCLCHECK(sdcclIbMalloc((void **)&stage->buffer, sizeof(remMeta)));

ib_recv:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, &rComm->base.sock,
                                   stage->buffer, sizeof(remMeta),
                                   &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return sdcclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct sdcclIbConnectionMetadata));

  // IB setup
  // Pre-declare variables because of goto
  struct sdcclIbMergedDev *mergedDev;
  struct sdcclIbDev *ibDev;
  int ibDevN;
  struct sdcclIbRecvCommDev *rCommDev;
  struct sdcclIbDevInfo *remDevInfo;
  struct sdcclIbQp *qp;

  mergedDev = sdcclIbMergedDevs + lComm->dev;
  rComm->base.ndevs = mergedDev->ndevs;
  rComm->base.nqps = sdcclParamIbQpsPerConn() *
                     rComm->base.ndevs; // We must have at least 1 qp per-device
  rComm->base.isSend = false;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.ndevs) {
    WARN("NET/IB : Local mergedDev %s has a different number of devices=%d as "
         "remote %s %d",
         mergedDev->devName, rComm->base.ndevs, remMeta.devName,
         rComm->base.nRemDevs);
  }

  // Metadata to send back to requestor (sender)
  struct sdcclIbConnectionMetadata meta;
  memset(&meta, 0,
         sizeof(meta)); // Initialize meta, including meta.retransEnabled = 0

  // IBRC retransmission: default disabled, can be enabled via
  // SDCCL_IB_RETRANS_ENABLE=1
  meta.retransEnabled = sdcclParamIbRetransEnable() ? 1 : 0;

  // Create SRQ if retransmission is enabled
  if (remMeta.retransEnabled && meta.retransEnabled) {
    ibDevN = mergedDev->devs[0];
    ibDev = sdcclIbDevs + ibDevN;

    sdcclResult_t srq_result = sdcclIbCreateSrq(
        ibDev->context, rComm->devs[0].base.pd, &rComm->srqMgr);
    if (srq_result == sdcclSuccess) {
      srq = (struct ibv_srq *)rComm->srqMgr.srq;
      TRACE(SDCCL_NET, "Receiver: Created SRQ for retransmission: srq=%p",
            srq);
    } else {
      INFO(SDCCL_NET,
           "Receiver: Failed to create SRQ (result=%d), disabling "
           "retransmission",
           srq_result);
      meta.retransEnabled = 0;
      srq = NULL;
    }
  }

  for (int i = 0; i < rComm->base.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = mergedDev->devs[i];
    SDCCLCHECK(sdcclIbInitCommDevBase(ibDevN, &rCommDev->base));
    ibDev = sdcclIbDevs + ibDevN;
    SDCCLCHECK(sdcclIbGetGidIndex(ibDev->context, ibDev->portNum,
                                    ibDev->portAttr.gid_tbl_len,
                                    &rCommDev->base.gidInfo.localGidIndex));
    SDCCLCHECK(sdcclWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                      rCommDev->base.gidInfo.localGidIndex,
                                      &rCommDev->base.gidInfo.localGid));
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id =
        rComm->base.remDevs[i].iid;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix =
        rComm->base.remDevs[i].spn;
  }

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIdx;
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIdx = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIdx;
    qp = rComm->base.qps + q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIdx;

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = sdcclIbDevs + ibDevN;
    SDCCLCHECK(sdcclIbCreateQp(ibDev->portNum, &rCommDev->base,
                                 IBV_ACCESS_REMOTE_WRITE |
                                     IBV_ACCESS_REMOTE_ATOMIC |
                                     IBV_ACCESS_REMOTE_READ,
                                 qp));
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].eceSupported) {
      SDCCLCHECK(sdcclWrapIbvSetEce(qp->qp, &remMeta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));

      // Query the reduced ece for this QP (matching enhancements between the
      // requestor and the responder) Store this in our own qpInfo for returning
      // to the requestor
      if (meta.qpInfo[q].eceSupported)
        SDCCLCHECK(sdcclWrapIbvQueryEce(qp->qp, &meta.qpInfo[q].ece,
                                          &meta.qpInfo[q].eceSupported));
    }

    SDCCLCHECK(sdcclIbRtrQp(qp->qp, rCommDev->base.gidInfo.localGidIndex,
                              remMeta.qpInfo[q].qpn, remDevInfo));
    SDCCLCHECK(sdcclIbRtsQp(qp->qp));
  }

  rComm->flushEnabled = 1;

  for (int i = 0; i < mergedDev->ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rCommDev->base.ibDevN;
    ibDev = sdcclIbDevs + ibDevN;

    // Retain remote fifo info and prepare my RDMA ops
    rCommDev->fifoRkey = remMeta.devs[i].fifoRkey;
    rComm->remFifo.addr = remMeta.fifoAddr;
    SDCCLCHECK(sdcclWrapIbvRegMr(
        &rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems,
        sizeof(struct sdcclIbSendFifo) * MAX_REQUESTS *
            SDCCL_NET_IB_MAX_RECVS,
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_READ));
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (sdcclParamIbUseInline())
      rComm->remFifo.flags = IBV_SEND_INLINE;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      SDCCLCHECK(sdcclWrapIbvRegMr(&rCommDev->gpuFlush.hostMr,
                                     rCommDev->base.pd, &rComm->gpuFlushHostMem,
                                     sizeof(int), IBV_ACCESS_LOCAL_WRITE));
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      SDCCLCHECK(
          sdcclIbCreateQp(ibDev->portNum, &rCommDev->base,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ,
                           &rCommDev->gpuFlush.qp));
      struct sdcclIbDevInfo devInfo;
      devInfo.lid = ibDev->portAttr.lid;
      devInfo.linkLayer = ibDev->portAttr.link_layer;
      devInfo.ibPort = ibDev->portNum;
      devInfo.spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.iid = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu = ibDev->portAttr.active_mtu;
      SDCCLCHECK(sdcclIbRtrQp(rCommDev->gpuFlush.qp.qp,
                                rCommDev->base.gidInfo.localGidIndex,
                                rCommDev->gpuFlush.qp.qp->qp_num, &devInfo));
      SDCCLCHECK(sdcclIbRtsQp(rCommDev->gpuFlush.qp.qp));
    }

    // Fill Handle
    meta.devs[i].lid = ibDev->portAttr.lid;
    meta.devs[i].linkLayer = rCommDev->base.gidInfo.linkLayer =
        ibDev->portAttr.link_layer;
    meta.devs[i].ibPort = ibDev->portNum;
    meta.devs[i].spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].iid = rCommDev->base.gidInfo.localGid.global.interface_id;

    // Adjust the MTU
    remMeta.devs[i].mtu =
        (enum ibv_mtu)std::min(remMeta.devs[i].mtu, ibDev->portAttr.active_mtu);
    meta.devs[i].mtu = remMeta.devs[i].mtu;

    // Prepare sizes fifo
    SDCCLCHECK(sdcclWrapIbvRegMr(
        &rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo,
        sizeof(int) * MAX_REQUESTS * SDCCL_NET_IB_MAX_RECVS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;

    // Create control QP for retransmission if enabled
    if (remMeta.retransEnabled && meta.retransEnabled) {
      SDCCLCHECK(sdcclIbCreateCtrlQp(ibDev->context, rCommDev->base.pd,
                                       ibDev->portNum, &rCommDev->ctrlQp));
      meta.ctrlQpn[i] = rCommDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibDev->portAttr.lid;
      meta.ctrlGid[i] = rCommDev->base.gidInfo.localGid;

      TRACE(SDCCL_NET,
            "Receiver: Control QP created for dev %d, qpn=%u, lid=%u", i,
            meta.ctrlQpn[i], meta.ctrlLid[i]);

      size_t ackBufSize =
          (sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING) *
          SDCCL_IB_ACK_BUF_COUNT;
      rCommDev->ackBuffer = malloc(ackBufSize);
      SDCCLCHECK(sdcclWrapIbvRegMr(&rCommDev->ackMr, rCommDev->base.pd,
                                     rCommDev->ackBuffer, ackBufSize,
                                     IBV_ACCESS_LOCAL_WRITE));

      TRACE(SDCCL_NET,
            "Recv: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibDev->portAttr.link_layer);

      sdcclResult_t ah_result = sdcclIbSetupCtrlQpConnection(
          ibDev->context, rCommDev->base.pd, &rCommDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibDev->portNum, ibDev->portAttr.link_layer,
          rCommDev->base.gidInfo.localGidIndex);

      if (ah_result != sdcclSuccess || !rCommDev->ctrlQp.ah) {
        INFO(SDCCL_NET,
             "Receiver Control QP setup failed for dev %d, disabling "
             "retransmission",
             i);
        meta.retransEnabled = 0;

        if (rCommDev->ackMr)
          sdcclWrapIbvDeregMr(rCommDev->ackMr);
        if (rCommDev->ackBuffer)
          free(rCommDev->ackBuffer);
        sdcclIbDestroyCtrlQp(&rCommDev->ctrlQp);
      } else {
        TRACE(SDCCL_NET,
              "Receiver Control QP successfully initialized for dev %d (ah=%p)",
              i, rCommDev->ctrlQp.ah);
      }
    }
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = sdcclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer)
    free(stage->buffer);
  SDCCLCHECK(sdcclIbMalloc((void **)&stage->buffer,
                             sizeof(struct sdcclIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct sdcclIbConnectionMetadata));

ib_send:
  SDCCLCHECK(sdcclSocketProgress(
      SDCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer,
      sizeof(struct sdcclIbConnectionMetadata), &stage->offset));
  if (stage->offset < sizeof(struct sdcclIbConnectionMetadata))
    return sdcclSuccess;

  stage->offset = 0;
  stage->state = sdcclIbCommStatePendingReady;

ib_recv_ready:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, &rComm->base.sock,
                                   &rComm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return sdcclSuccess;

  // Initialize retransmission state
  SDCCLCHECK(sdcclIbRetransInit(&rComm->retrans));

  if (rComm->retrans.enabled) {
    INFO(SDCCL_NET,
         "NET/IBRC Receiver: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         rComm->retrans.minRtoUs, rComm->retrans.maxRetry,
         rComm->retrans.ackInterval);
  } else {
    INFO(SDCCL_NET, "NET/IBRC Receiver: Retransmission DISABLED");
  }

  // Initialize SRQ with recv buffers if enabled
  if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
    rComm->srqMgr.postSrqCount = SDCCL_IB_SRQ_SIZE;

    // Post in batches until all are posted
    while (rComm->srqMgr.postSrqCount > 0) {
      SDCCLCHECK(sdcclIbSrqPostRecv(&rComm->srqMgr, SDCCL_IB_ACK_BUF_COUNT));
    }

    INFO(SDCCL_INIT | SDCCL_NET,
         "NET/IBRC Receiver: Posted %d recv WRs to SRQ for retransmission",
         SDCCL_IB_SRQ_SIZE);
  }

  free(stage->buffer);
  *recvComm = rComm;

  /* reset lComm stage */
  stage->state = sdcclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbGetRequest(struct sdcclIbNetCommBase *base,
                                  struct sdcclIbRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct sdcclIbRequest *r = base->reqs + i;
    if (r->type == SDCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->devBases[0] = NULL;
      r->devBases[1] = NULL;
      r->events[0] = r->events[1] = 0;
      *req = r;
      return sdcclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return sdcclInternalError;
}

sdcclResult_t sdcclIbFreeRequest(struct sdcclIbRequest *r) {
  r->type = SDCCL_NET_IB_REQ_UNUSED;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbTest(void *request, int *done, int *size);

sdcclResult_t sdcclIbRegMrDmaBufInternal(sdcclIbNetCommDevBase *base,
                                           void *data, size_t size, int type,
                                           uint64_t offset, int fd, int mrFlags,
                                           ibv_mr **mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0)
    pageSize = sysconf(_SC_PAGESIZE);
  struct sdcclIbMrCache *cache = &sdcclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  sdcclResult_t res;
  pthread_mutex_lock(&sdcclIbDevs[base->ibDevN].lock);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population ||
        addr < cache->slots[slot].addr) {         // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        SDCCLCHECKGOTO(
            sdcclRealloc(&cache->slots, cache->population, cache->capacity),
            res, returning);
      }
      // Deregister / register
      struct ibv_mr *mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      if (sdcclIbRelaxedOrderingEnabled &&
          !(mrFlags & SDCCL_NET_MR_FLAG_FORCE_SO))
        flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        SDCCLCHECKGOTO(sdcclWrapIbvRegDmabufMr(&mr, base->pd, offset,
                                                 pages * pageSize, addr, fd,
                                                 flags),
                        res, returning);
      } else {
        void *cpuptr = NULL;
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMmap(&cpuptr, (void *)addr, pages * pageSize);
        }
        if (sdcclIbRelaxedOrderingEnabled &&
            !(mrFlags & SDCCL_NET_MR_FLAG_FORCE_SO)) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING
          // support
          SDCCLCHECKGOTO(
              sdcclWrapIbvRegMrIova2(&mr, base->pd,
                                      cpuptr == NULL ? (void *)addr : cpuptr,
                                      pages * pageSize, addr, flags),
              res, returning);
        } else {
          SDCCLCHECKGOTO(
              sdcclWrapIbvRegMr(&mr, base->pd,
                                 cpuptr == NULL ? (void *)addr : cpuptr,
                                 pages * pageSize, flags),
              res, returning);
        }
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMunmap(cpuptr, pages * pageSize);
        }
      }
      TRACE(SDCCL_INIT | SDCCL_NET,
            "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d",
            (unsigned long)addr, (long long)pages * pageSize, mr->rkey,
            mr->lkey, fd);
      if (slot != cache->population)
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(struct sdcclIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      res = sdcclSuccess;
      goto returning;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <=
                   cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      res = sdcclSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&sdcclIbDevs[base->ibDevN].lock);
  return res;
}

struct sdcclIbNetCommDevBase *
sdcclIbGetNetCommDevBase(sdcclIbNetCommBase *base, int devIndex) {
  if (base->isSend) {
    struct sdcclIbSendComm *sComm = (struct sdcclIbSendComm *)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
sdcclResult_t sdcclIbRegMrDmaBuf(void *comm, void *data, size_t size,
                                   int type, uint64_t offset, int fd,
                                   int mrFlags, void **mhandle) {
  assert(size > 0);
  struct sdcclIbNetCommBase *base = (struct sdcclIbNetCommBase *)comm;
  struct sdcclIbMrHandle *mhandleWrapper =
      (struct sdcclIbMrHandle *)malloc(sizeof(struct sdcclIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    // Each sdcclIbNetCommDevBase is at different offset in send and recv
    // netComms
    struct sdcclIbNetCommDevBase *devComm = sdcclIbGetNetCommDevBase(base, i);
    SDCCLCHECK(sdcclIbRegMrDmaBufInternal(devComm, data, size, type, offset,
                                            fd, mrFlags,
                                            mhandleWrapper->mrs + i));
  }
  *mhandle = (void *)mhandleWrapper;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbRegMr(void *comm, void *data, size_t size, int type,
                             int mrFlags, void **mhandle) {
  return sdcclIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mrFlags,
                             mhandle);
}

sdcclResult_t sdcclIbDeregMrInternal(sdcclIbNetCommDevBase *base,
                                       ibv_mr *mhandle) {
  struct sdcclIbMrCache *cache = &sdcclIbDevs[base->ibDevN].mrCache;
  sdcclResult_t res;
  pthread_mutex_lock(&sdcclIbDevs[base->ibDevN].lock);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population],
                sizeof(struct sdcclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        SDCCLCHECKGOTO(sdcclWrapIbvDeregMr(mhandle), res, returning);
      }
      res = sdcclSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle,
       cache->population);
  res = sdcclInternalError;
returning:
  pthread_mutex_unlock(&sdcclIbDevs[base->ibDevN].lock);
  return res;
}

sdcclResult_t sdcclIbDeregMr(void *comm, void *mhandle) {
  struct sdcclIbMrHandle *mhandleWrapper = (struct sdcclIbMrHandle *)mhandle;
  struct sdcclIbNetCommBase *base = (struct sdcclIbNetCommBase *)comm;
  for (int i = 0; i < base->ndevs; i++) {
    // Each sdcclIbNetCommDevBase is at different offset in send and recv
    // netComms
    struct sdcclIbNetCommDevBase *devComm = sdcclIbGetNetCommDevBase(base, i);
    SDCCLCHECK(sdcclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return sdcclSuccess;
}

SDCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

sdcclResult_t sdcclIbMultiSend(struct sdcclIbSendComm *comm, int slot) {
  struct sdcclIbRequest **reqs = comm->fifoReqs[slot];
  volatile struct sdcclIbSendFifo *slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > SDCCL_NET_IB_MAX_RECVS)
    return sdcclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) {
    struct ibv_send_wr *wr = comm->wrs + r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge *sge = comm->sges + r;
    sge->addr = (uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r * 8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  uint32_t seq = 0;
  if (nreqs == 1) {
    if (comm->retrans.enabled) {
      seq = comm->retrans.sendSeq;
      comm->retrans.sendSeq = (comm->retrans.sendSeq + 1) & 0xFFFF;
      immData = sdcclIbEncodeImmData(seq, reqs[0]->send.size);
      TRACE(SDCCL_NET,
            "Sending packet with SEQ: seq=%u, size=%u, immData=0x%x", seq,
            reqs[0]->send.size, immData);
    } else {
      immData = reqs[0]->send.size;
    }
  } else {
    int *sizes = comm->remSizesFifo.elems[slot];
    for (int r = 0; r < nreqs; r++)
      sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs * sizeof(int);
  }

  struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
  if (nreqs > 1 ||
      (comm->ar && reqs[0]->send.size > sdcclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr =
          comm->remSizesFifo.addr +
          slot * SDCCL_NET_IB_MAX_RECVS * sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128
  // protocols still work
  const int align = 128;
  int nqps = sdcclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    sdcclIbQp *qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    for (int r = 0; r < nreqs; r++) {
      // Track this event for completion
      // sdcclIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length =
          std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges + r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    struct ibv_send_wr *bad_wr;
    SDCCLCHECK(sdcclWrapIbvPostSend(qp->qp, comm->wrs, &bad_wr));

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }

  // Add packet to retransmission buffer if enabled
  if (comm->retrans.enabled && nreqs == 1) {
    SDCCLCHECK(sdcclIbRetransAddPacket(
        &comm->retrans, seq, reqs[0]->send.size, reqs[0]->send.data,
        slots[0].addr, // remote_addr
        reqs[0]->send.lkeys, (uint32_t *)slots[0].rkeys));
  }

  comm->outstandingSends++;

  return sdcclSuccess;
}

sdcclResult_t sdcclIbIsend(void *sendComm, void *data, size_t size, int tag,
                             void *mhandle, void *phandle, void **request) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: sdcclIbIsend() called when comm->base.ready == 0");
    return sdcclInternalError;
  }
  if (comm->base.ready == 0) {
    *request = NULL;
    return sdcclSuccess;
  }

  struct sdcclIbMrHandle *mhandleWrapper = (struct sdcclIbMrHandle *)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct sdcclIbSendFifo *slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct sdcclIbRequest **reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead + 1;
  if (slots[0].idx != idx) {
    *request = NULL;
    return sdcclSuccess;
  }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r = 1; r < nreqs; r++)
    while (slots[r].idx != idx)
      ;
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads
                        // below
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag)
      continue;

    if (size > slots[r].size)
      size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union sdcclSocketAddress addr;
      sdcclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: "
           "size %ld addr %lx rkeys[0]=%x",
           r, nreqs, tag, sdcclSocketToString(&addr, line), slots[r].size,
           slots[r].addr, slots[r].rkeys[0]);
      return sdcclInternalError;
    }

    struct sdcclIbRequest *req;
    SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
    req->type = SDCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents =
        sdcclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      sdcclIbQp *qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      sdcclIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same
      // set of QPs inside sdcclIbMultiSend()
      qpIndex = (qpIndex + 1) % comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r] == NULL)
        return sdcclSuccess;
    }

    TIME_START(0);
    SDCCLCHECK(sdcclIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and
    // sanity checks
    memset((void *)slots, 0, sizeof(struct sdcclIbSendFifo));
    memset(reqs, 0, SDCCL_NET_IB_MAX_RECVS * sizeof(struct sdcclIbRequest *));
    comm->fifoHead++;
    TIME_STOP(0);
    return sdcclSuccess;
  }

  *request = NULL;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbPostFifo(struct sdcclIbRecvComm *comm, int n,
                                void **data, size_t *sizes, int *tags,
                                void **mhandles, struct sdcclIbRequest *req) {
  return sdcclIbCommonPostFifo(comm, n, data, sizes, tags, mhandles, req,
                                sdcclIbAddEvent);
}

sdcclResult_t sdcclIbIrecv(void *recvComm, int n, void **data, size_t *sizes,
                             int *tags, void **mhandles, void **phandles,
                             void **request) {
  struct sdcclIbRecvComm *comm = (struct sdcclIbRecvComm *)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IB: sdcclIbIrecv() called when comm->base.ready == 0");
    return sdcclInternalError;
  }
  if (comm->base.ready == 0) {
    *request = NULL;
    return sdcclSuccess;
  }
  if (n > SDCCL_NET_IB_MAX_RECVS)
    return sdcclInternalError;

  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;

  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps =
      sdcclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Post recvs
  struct ibv_recv_wr *bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct sdcclIbQp *qp = comm->base.qps + comm->base.qpIndex;
    sdcclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    SDCCLCHECK(sdcclWrapIbvPostRecv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  SDCCLCHECK(sdcclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbIflush(void *recvComm, int n, void **data, int *sizes,
                              void **mhandles, void **request) {
  struct sdcclIbRecvComm *comm = (struct sdcclIbRecvComm *)recvComm;
  int last = -1;
  for (int i = 0; i < n; i++)
    if (sizes[i])
      last = i;
  if (comm->flushEnabled == 0 || last == -1)
    return sdcclSuccess;

  // Only flush once using the last non-zero receive
  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct sdcclIbMrHandle *mhandle = (struct sdcclIbMrHandle *)mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr *bad_wr;
    SDCCLCHECK(
        sdcclWrapIbvPostSend(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    sdcclIbAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return sdcclSuccess;
}

static sdcclResult_t sdcclIbrcTestPreCheck(struct sdcclIbRequest *r) {
  if (!r)
    return sdcclInternalError;

  if (r->type == SDCCL_NET_IB_REQ_SEND && r->base->isSend) {
    struct sdcclIbSendComm *sComm = (struct sdcclIbSendComm *)r->base;
    if (sComm->retrans.enabled) {
      // Check if control QP is still valid before accessing it
      bool ctrlQpValid = false;
      for (int i = 0; i < sComm->base.ndevs; i++) {
        if (sComm->devs[i].ctrlQp.qp && sComm->devs[i].ctrlQp.cq) {
          ctrlQpValid = true;
          break;
        }
      }

      if (ctrlQpValid) {
        for (int i = 0; i < sComm->base.ndevs; i++) {
          // Only poll if control QP is still valid
          if (sComm->devs[i].ctrlQp.qp && sComm->devs[i].ctrlQp.cq) {
            for (int p = 0; p < 4; p++) {
              sdcclResult_t poll_result =
                  sdcclIbRetransRecvAckViaUd(sComm, i);
              if (poll_result != sdcclSuccess)
                break;
            }
          }
        }

        uint64_t now_us = sdcclIbGetTimeUs();
        const uint64_t CHECK_INTERVAL_US = 1000;
        if (now_us - sComm->lastTimeoutCheckUs >= CHECK_INTERVAL_US) {
          // Don't use SDCCLCHECK here - retransmission errors shouldn't block
          // operation completion
          sdcclResult_t retrans_result =
              sdcclIbRetransCheckTimeout(&sComm->retrans, sComm);
          if (retrans_result != sdcclSuccess &&
              retrans_result != sdcclInProgress) {
            // Log error but don't fail the operation
            if (sdcclDebugNoWarn == 0)
              INFO(SDCCL_ALL,
                   "%s:%d -> %d (retransmission check failed, continuing)",
                   __FILE__, __LINE__, retrans_result);
          }
          sComm->lastTimeoutCheckUs = now_us;
        }
      }
    }
  }

  if (r->type == SDCCL_NET_IB_REQ_RECV && !r->base->isSend) {
    struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)r->base;
    if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
      const int kPostThreshold = 16;
      if (rComm->srqMgr.postSrqCount >= kPostThreshold) {
        SDCCLCHECK(
            sdcclIbSrqPostRecv(&rComm->srqMgr, SDCCL_IB_ACK_BUF_COUNT));
      }
    }
  }

  return sdcclSuccess;
}

static sdcclResult_t sdcclIbrcProcessWc(struct sdcclIbRequest *r,
                                          struct ibv_wc *wc, int devIndex,
                                          bool *handled) {
  if (!r || !wc || !handled)
    return sdcclInternalError;

  *handled = false;

  if (r->type == SDCCL_NET_IB_REQ_RECV && !r->base->isSend) {
    struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)r->base;

    if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL &&
        wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      uint32_t seq, size;
      sdcclIbDecodeImmData(wc->imm_data, &seq, &size);
      r->recv.sizes[0] = size;

      struct sdcclIbAckMsg ack_msg = {0};
      int shouldAck = 0;
      SDCCLCHECK(sdcclIbRetransRecvPacket(&rComm->retrans, seq, &ack_msg,
                                            &shouldAck));

      if (shouldAck) {
        sdcclResult_t ack_result =
            sdcclIbRetransSendAckViaUd(rComm, &ack_msg, devIndex);
        if (ack_result != sdcclSuccess) {
          TRACE(SDCCL_NET, "Failed to send ACK for seq=%u (result=%d)", seq,
                ack_result);
        } else {
          TRACE(SDCCL_NET, "Sent ACK for seq=%u, ack_seq=%u", seq,
                ack_msg.ackSeq);
        }
      } else {
        TRACE(SDCCL_NET, "No ACK needed for seq=%u (expect=%u)", seq,
              rComm->retrans.recvSeq);
      }
    } else if (!rComm->retrans.enabled &&
               wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
      r->recv.sizes[0] = wc->imm_data;
    }
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIbTest(void *request, int *done, int *sizes) {
  static const struct sdcclIbCommonTestOps kIbrcTestOps = {
      .component = "NET/IBRC",
      .pre_check = sdcclIbrcTestPreCheck,
      .process_wc = sdcclIbrcProcessWc};
  return sdcclIbCommonTestDataQp((struct sdcclIbRequest *)request, done,
                                  sizes, &kIbrcTestOps);
}

sdcclResult_t sdcclIbCloseSend(void *sendComm) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  if (comm) {
    SDCCLCHECK(sdcclSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int nCqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          sdcclWrapIbvPollCq(commDev->base.cq, 64, wcs, &nCqe);
          if (nCqe == 0)
            break;
        }
      }
    }

    // 清理重传使用的资源：即使 runtime 期间禁用了 retrans，也可能已经
    // 分配了 MR/ctrl QP，必须在 PD 释放前全部销毁
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (commDev->ackMr) {
        sdcclWrapIbvDeregMr(commDev->ackMr);
        commDev->ackMr = NULL;
      }
      if (commDev->ackBuffer) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
      sdcclIbDestroyCtrlQp(&commDev->ctrlQp);
      if (i == 0 && comm->retransHdrMr) {
        sdcclWrapIbvDeregMr(comm->retransHdrMr);
        comm->retransHdrMr = NULL;
      }
    }
    sdcclIbRetransDestroy(&comm->retrans);

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        SDCCLCHECK(sdcclWrapIbvDestroyQp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (commDev->fifoMr != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->fifoMr));
      if (commDev->putSignalScratchpadMr != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->putSignalScratchpadMr));
      if (comm->remSizesFifo.mrs[i] != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(comm->remSizesFifo.mrs[i]));
      SDCCLCHECK(sdcclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return sdcclSuccess;
}

sdcclResult_t sdcclIbCloseRecv(void *recvComm) {
  struct sdcclIbRecvComm *comm = (struct sdcclIbRecvComm *)recvComm;
  if (comm) {
    SDCCLCHECK(sdcclSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbRecvCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int nCqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          sdcclWrapIbvPollCq(commDev->base.cq, 64, wcs, &nCqe);
          if (nCqe == 0)
            break;
        }
      }
    }

    // Clean up retransmission resources regardless of enable flag
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbRecvCommDev *commDev = comm->devs + i;
      if (commDev->ackMr) {
        sdcclWrapIbvDeregMr(commDev->ackMr);
        commDev->ackMr = NULL;
      }
      if (commDev->ackBuffer) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
      sdcclIbDestroyCtrlQp(&commDev->ctrlQp);
    }

    if (comm->srqMgr.srq != NULL) {
      sdcclIbDestroySrq(&comm->srqMgr);
    }
    sdcclIbRetransDestroy(&comm->retrans);

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        SDCCLCHECK(sdcclWrapIbvDestroyQp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbRecvCommDev *commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL)
          SDCCLCHECK(sdcclWrapIbvDestroyQp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL)
          SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->gpuFlush.hostMr));
      }
      if (commDev->fifoMr != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->fifoMr));
      if (commDev->sizesFifoMr != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->sizesFifoMr));
      SDCCLCHECK(sdcclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclIbCloseListen(void *listenComm) {
  struct sdcclIbListenComm *comm = (struct sdcclIbListenComm *)listenComm;
  if (comm) {
    SDCCLCHECK(sdcclSocketClose(&comm->sock));
    free(comm);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclIbGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < sdcclNMergedIbDevs; i++) {
    if (strcmp(sdcclIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return sdcclSuccess;
    }
  }
  return sdcclSystemError;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// sdcclSuccess : GDR works
// sdcclSystemError : no module or module loaded but not supported by GPU
sdcclResult_t sdcclIbGdrSupport() {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    // Check for the nv_peer_mem module being loaded
    moduleLoaded =
        ((access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) &&
         // Also support the new nvidia-peermem module
         (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) ==
          -1))
            ? 0
            : 1;
  }
  if (moduleLoaded == 0)
    return sdcclSystemError;
  return sdcclSuccess;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// sdcclSuccess : DMA-BUF support is available
// sdcclSystemError : DMA-BUF is not supported by the kernel
sdcclResult_t sdcclIbDmaBufSupport(int dev) {
  static int dmaBufSupported = -1;
  if (dmaBufSupported == -1) {
    sdcclResult_t res;
    struct ibv_pd *pd;
    struct ibv_context *ctx;
    struct sdcclIbMergedDev *mergedDev = sdcclIbMergedDevs + dev;

    // Test each dev
    for (int i = 0; i < mergedDev->ndevs; i++) {
      int ibDev = mergedDev->devs[i];
      ctx = sdcclIbDevs[ibDev].context;
      SDCCLCHECKGOTO(sdcclWrapIbvAllocPd(&pd, ctx), res, failure);
      // Test kernel DMA-BUF support with a dummy call (fd=-1)
      (void)sdcclWrapDirectIbvRegDmabufMr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                           0ULL /*iova*/, -1 /*fd*/,
                                           0 /*flags*/);
      // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
      // supported (EBADF otherwise)
      dmaBufSupported =
          (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
      SDCCLCHECKGOTO(sdcclWrapIbvDeallocPd(pd), res, failure);
    }
  }
  if (dmaBufSupported == 0)
    return sdcclSystemError;
  return sdcclSuccess;
failure:
  dmaBufSupported = 0;
  return sdcclSystemError;
}

sdcclResult_t sdcclIbDevices(int *ndev) {
  *ndev = sdcclNMergedIbDevs;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbGetProperties(int dev, void *props) {
  struct sdcclIbMergedDev *mergedDev = sdcclIbMergedDevs + dev;
  sdcclNetProperties_t *properties = (sdcclNetProperties_t *)props;

  properties->name = mergedDev->devName;
  properties->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device (should be the
  // same)
  struct sdcclIbDev *ibDev = sdcclIbDevs + mergedDev->devs[0];
  properties->pciPath = ibDev->pciPath;
  properties->guid = ibDev->guid;
  properties->ptrSupport = SDCCL_PTR_HOST;

  if (sdcclIbGdrSupport() == sdcclSuccess) {
    properties->ptrSupport |= SDCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  properties->regIsGlobal = 1;
  if (sdcclIbDmaBufSupport(dev) == sdcclSuccess) {
    properties->ptrSupport |= SDCCL_PTR_DMABUF;
  }
  properties->latency = 0; // Not set
  properties->port = ibDev->portNum + ibDev->realPort;
  properties->maxComms = ibDev->maxQp;
  properties->maxRecvs = SDCCL_NET_IB_MAX_RECVS;
  properties->netDeviceType = SDCCL_NET_DEVICE_HOST;
  properties->netDeviceVersion = SDCCL_NET_DEVICE_INVALID_VERSION;
  return sdcclSuccess;
}
sdcclResult_t sdcclIbIput(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                            size_t size, int srcRank, int dstRank,
                            void **srcHandles, void **dstHandles,
                            void **request) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  struct sdcclOneSideHandleInfo *srcInfo =
      (struct sdcclOneSideHandleInfo *)srcHandles;
  struct sdcclOneSideHandleInfo *dstInfo =
      (struct sdcclOneSideHandleInfo *)dstHandles;

  struct sdcclIbQp *qp = &comm->base.qps[0];
  void *srcPtr = (void *)(srcInfo->baseVas[srcRank] + srcOff);
  void *dstPtr = (void *)(dstInfo->baseVas[dstRank] + dstOff);
  int lkey = srcInfo->lkeys[srcRank];
  int rkey = dstInfo->rkeys[dstRank];
  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_IPUT;
  req->sock = &comm->base.sock;
  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = req - comm->base.reqs;
  wr.next = NULL;
  wr.wr.rdma.remote_addr = (uint64_t)dstPtr;
  wr.wr.rdma.rkey = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr; // Local buffer address
  sge.length = (uint32_t)size;  // ibv_sge::length is 32-bit
  if ((size_t)sge.length != size) {
    WARN("sdcclIbIput: transfer size %zu exceeds ibv_sge 32-bit limit", size);
    sdcclIbFreeRequest(req);
    return sdcclInternalError;
  }
  sge.lkey = lkey; // Local key

  struct ibv_send_wr *bad_wr;
  SDCCLCHECK(sdcclWrapIbvPostSend(qp->qp, &wr, &bad_wr));
  sdcclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbIget(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                            size_t size, int srcRank, int dstRank,
                            void **srcHandles, void **dstHandles,
                            void **request) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  struct sdcclOneSideHandleInfo *srcInfo =
      (struct sdcclOneSideHandleInfo *)srcHandles;
  struct sdcclOneSideHandleInfo *dstInfo =
      (struct sdcclOneSideHandleInfo *)dstHandles;

  struct sdcclIbQp *qp = &comm->base.qps[0];
  // For RDMA READ: remote_addr is the source (remote peer), sge is the local
  // destination
  void *srcPtr = (void *)(srcInfo->baseVas[srcRank] + srcOff);
  void *dstPtr = (void *)(dstInfo->baseVas[dstRank] + dstOff);
  int rkey = srcInfo->rkeys[srcRank]; // remote key for the source buffer
  int lkey = dstInfo->lkeys[dstRank]; // local key for the destination buffer
  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_IGET;
  req->sock = &comm->base.sock;
  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = req - comm->base.reqs;
  wr.next = NULL;
  wr.wr.rdma.remote_addr = (uint64_t)srcPtr; // remote source address
  wr.wr.rdma.rkey = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)dstPtr; // local destination address
  sge.length = (uint32_t)size;
  if ((size_t)sge.length != size) {
    WARN("sdcclIbIget: transfer size %zu exceeds ibv_sge 32-bit limit", size);
    sdcclIbFreeRequest(req);
    return sdcclInternalError;
  }
  sge.lkey = lkey;

  struct ibv_send_wr *bad_wr;
  SDCCLCHECK(sdcclWrapIbvPostSend(qp->qp, &wr, &bad_wr));
  sdcclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbIputSignal(void *sendComm, uint64_t srcOff,
                                  uint64_t dstOff, size_t size, int srcRank,
                                  int dstRank, void **srcHandles,
                                  void **dstHandles, uint64_t signalOff,
                                  void **signalHandles, uint64_t signalValue,
                                  void **request) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  struct sdcclOneSideHandleInfo *srcInfo =
      (struct sdcclOneSideHandleInfo *)srcHandles;
  struct sdcclOneSideHandleInfo *dstInfo =
      (struct sdcclOneSideHandleInfo *)dstHandles;
  struct sdcclOneSideHandleInfo *signalInfo =
      (struct sdcclOneSideHandleInfo *)signalHandles;
  if (signalInfo == NULL || signalInfo->baseVas == NULL) {
    WARN("sdcclIbIputSignal: signalHandles is NULL or uninitialized");
    return sdcclInternalError;
  }

  struct sdcclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;
  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_IPUT;
  req->sock = &comm->base.sock;
  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // wr[0]: RDMA WRITE (data) — no CQE, chained to signal
  if (size > 0 && srcInfo != NULL && dstInfo != NULL) {
    void *srcPtr = (void *)(srcInfo->baseVas[srcRank] + srcOff);
    void *dstPtr = (void *)(dstInfo->baseVas[dstRank] + dstOff);
    uint32_t lkey = srcInfo->lkeys[srcRank];
    uint32_t rkey = dstInfo->rkeys[dstRank];

    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].send_flags = 0; // No CQE — only signal gets CQE
    wr[0].wr_id = req - comm->base.reqs;
    wr[0].next = &wr[1]; // Chain to signal
    wr[0].wr.rdma.remote_addr = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;
    sge[0].length = (uint32_t)size;
    if ((size_t)sge[0].length != size) {
      WARN("sdcclIbIputSignal: transfer size %zu exceeds ibv_sge 32-bit limit",
           size);
      sdcclIbFreeRequest(req);
      return sdcclInternalError;
    }
    sge[0].lkey = lkey;
  }

  // wr[1]: ATOMIC FETCH_AND_ADD (signal) — IBV_SEND_SIGNALED
  void *signalPtr = (void *)(signalInfo->baseVas[dstRank] + signalOff);
  uint32_t signalRkey = signalInfo->rkeys[dstRank];

  wr[1].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags = IBV_SEND_SIGNALED;
  wr[1].wr_id = req - comm->base.reqs;
  wr[1].next = NULL;
  wr[1].wr.atomic.remote_addr = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add = signalValue;
  wr[1].wr.atomic.rkey = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Post chained (data+signal) or signal-only
  struct ibv_send_wr *bad_wr;
  bool chainData = (size > 0 && srcInfo != NULL && dstInfo != NULL);
  SDCCLCHECK(
      sdcclWrapIbvPostSend(qp->qp, chainData ? &wr[0] : &wr[1], &bad_wr));
  sdcclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);

  *request = req;
  return sdcclSuccess;
}

// Adapter wrapper functions

struct sdcclNetAdaptor sdcclNetIb = {
    // Basic functions
    "IB", sdcclIbInit, sdcclIbDevices, sdcclIbGetProperties,

    // Setup functions
    sdcclIbListen, sdcclIbConnect, sdcclIbAccept, sdcclIbCloseSend,
    sdcclIbCloseRecv, sdcclIbCloseListen,

    // Memory region functions
    sdcclIbRegMr, sdcclIbRegMrDmaBuf, sdcclIbDeregMr,

    // Two-sided functions
    sdcclIbIsend, sdcclIbIrecv, sdcclIbIflush, sdcclIbTest,

    // One-sided functions
    sdcclIbIput, sdcclIbIget, sdcclIbIputSignal,

    // Device name lookup
    sdcclIbGetDevFromName};
