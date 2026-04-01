/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef USE_UCX

#include "ucx_adaptor.h"
#include "adaptor.h"
#include "check.h"
#include "comm.h"
#include "core.h"
#include "debug.h"
#include "sdccl.h"
#include "ib_common.h"
#include "ibvwrap.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "utils.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucp/api/ucp.h>
#include <unistd.h>

#define SDCCL_IB_LLSTR(link_layer)                                            \
  ((link_layer) == IBV_LINK_LAYER_INFINIBAND ? "IB" : "ETH")
#define SDCCL_STATIC_ASSERT(condition, message)                               \
  static_assert(condition, message)

// Additional global variables
static pthread_mutex_t sdccl_p2p_lock = PTHREAD_MUTEX_INITIALIZER;
static int sdcclIbGdrModuleLoaded = 0;
static struct { pthread_once_t once; } onces[MAX_IB_DEVS];

sdcclResult_t sdcclIbMakeVDeviceInternal(int *d,
                                           sdcclNetVDeviceProps_t *props,
                                           int sdcclNIbDevs,
                                           int *sdcclNMergedIbDevs) {
  if ((sdcclParamIbMergeNics() == 0) && props->ndevs > 1) {
    INFO(SDCCL_NET, "NET/IB : Skipping makeVDevice, sdccl_IB_MERGE_NICS=0");
    return sdcclInvalidUsage;
  }

  if (props->ndevs == 0) {
    WARN("NET/IB : Can't make virtual NIC with 0 devices");
    return sdcclInvalidUsage;
  }

  if (*sdcclNMergedIbDevs == MAX_IB_DEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_DEVS);
    return sdcclInvalidUsage;
  }

  // Always count up number of merged devices
  sdcclIbMergedDev *mDev = sdcclIbMergedDevs + *sdcclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    sdcclIbDev *dev = sdcclIbDevs + props->devs[i];
    if (mDev->vProps.ndevs == SDCCL_IB_MAX_DEVS_PER_NIC)
      return sdcclInvalidUsage; // SDCCL_IB_MAX_DEVS_PER_NIC
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName),
               sizeof(mDev->devName) - strlen(mDev->devName), "+%s",
               dev->devName);
      // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
    }
  }

  // Check link layers
  sdcclIbDev *dev0 = sdcclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= sdcclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i],
           sdcclNIbDevs);
      return sdcclInvalidUsage;
    }
    sdcclIbDev *dev = sdcclIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and "
           "[%d]%s:%d/%s. Try selecting NICs of only one link type using "
           "sdccl_IB_HCA",
           props->devs[0], dev0->devName, dev0->portNum, "IB", props->devs[i],
           dev->devName, dev->portNum, "IB");
      return sdcclInvalidUsage;
    }
  }

  *d = *sdcclNMergedIbDevs;
  (*sdcclNMergedIbDevs)++;

  INFO(SDCCL_NET,
       "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d,
       mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return sdcclSuccess;
}
static int sdcclIbMatchVfPath(char *path1, char *path2) {
  // Merge multi-port NICs into the same PCI device
  if (sdcclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1) - 4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1) - 1) == 0;
  }
}
static void sdcclIbStatsFatalError(struct sdcclIbStats *stat) {
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static void sdcclIbQpFatalError(struct ibv_qp *qp) {
  sdcclIbStatsFatalError((struct sdcclIbStats *)qp->qp_context);
}
static void sdcclIbCqFatalError(struct ibv_cq *cq) {
  sdcclIbStatsFatalError((struct sdcclIbStats *)cq->cq_context);
}
static void sdcclIbDevFatalError(struct sdcclIbDev *dev) {
  sdcclIbStatsFatalError(&dev->stats);
}
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  sdcclIbGdrModuleLoaded =
      KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
      KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
      KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

/* for data direct nic, the device name is ends with suffix '_dma`.
 * remove this suffix before passing name to device */
void plugin_get_device_name(const char *input, char *output,
                            size_t outputSize) {
  const char *suffix = "_dma";
  size_t inputLen = strlen(input);
  size_t suffixLen = strlen(suffix);

  if (inputLen >= suffixLen &&
      strcmp(input + inputLen - suffixLen, suffix) == 0) {
    size_t newLen = inputLen - suffixLen;
    if (newLen >= outputSize) {
      newLen = outputSize - 1;
    }
    memcpy(output, input, newLen);
    output[newLen] = '\0';
  } else {
    strncpy(output, input, outputSize - 1);
    output[outputSize - 1] = '\0';
  }
}
// Missing arrays and constants
static int ibv_widths[] = {1, 2, 4, 8, 12};
static int ibv_speeds[] = {2500,  5000,  10000,  14000,
                           25000, 50000, 100000, 200000};

// Helper function
static int first_bit_set(int value, int max_bits) {
  for (int i = 0; i <= max_bits; i++) {
    if (value & (1 << i))
      return i;
  }
  return max_bits;
}

// Function implementations from ucx_adaptor.cc
int sdccl_p2p_ib_width(int width) {
  return ibv_widths[first_bit_set(width, sizeof(ibv_widths) / sizeof(int) - 1)];
}

int sdccl_p2p_ib_speed(int speed) {
  return ibv_speeds[first_bit_set(speed, sizeof(ibv_speeds) / sizeof(int) - 1)];
}

sdcclResult_t sdcclIbStatsInit(struct sdcclIbStats *stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return sdcclSuccess;
}

sdcclResult_t sdcclP2pIbPciPath(sdcclIbDev *devs, int num_devs,
                                  char *devName, char **path, int *realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char *p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s", devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p) - 1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (sdcclParamIbMergeVfs())
      p[strlen(p) - 3] = p[strlen(p) - 4] = '0';
    // Keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d = 0; d < num_devs; d++) {
      if (sdcclIbMatchVfPath(p, sdcclIbDevs[d].pciPath))
        (*realPort)++;
    }
    *path = p;
  }
  return sdcclSuccess;
}

// TODO

// static void *sdcclIbAsyncThreadMain(void *args) {
//   struct sdcclIbDev *dev = (struct sdcclIbDev *)args;
//   while (1) {
//     struct ibv_async_event event;
//     if (sdcclSuccess != sdcclWrapIbvGetAsyncEvent(dev->context, &event)) {
//       break;
//     }
//     char *str;
//     struct ibv_cq *cq __attribute__((unused)) =
//         event.element.cq; // only valid if CQ error
//     struct ibv_qp *qp __attribute__((unused)) =
//         event.element.qp; // only valid if QP error
//     struct ibv_srq *srq __attribute__((unused)) =
//         event.element.srq; // only valid if SRQ error
//     if (sdcclSuccess != sdcclWrapIbvEventTypeStr(&str, event.event_type)) {
//       break;
//     }
//     switch (event.event_type) {
//       case IBV_EVENT_DEVICE_FATAL:
//         // the above is device fatal error
//         WARN("NET/IB : %s:%d async fatal event: %s", dev->devName,
//         dev->portNum,
//              str);
//         sdcclIbDevFatalError(dev);
//         break;
//       case IBV_EVENT_PORT_ACTIVE:
//       case IBV_EVENT_PORT_ERR:
//         WARN("NET/IB : %s:%d port event: %s", dev->devName, dev->portNum,
//         str); break;
//       case IBV_EVENT_CQ_ERR:
//         WARN("NET/IB : %s:%d CQ event: %s", dev->devName, dev->portNum, str);
//         break;
//       case IBV_EVENT_QP_FATAL:
//       case IBV_EVENT_QP_ACCESS_ERR:
//         WARN("NET/IB : %s:%d QP event: %s", dev->devName, dev->portNum, str);
//         break;
//       case IBV_EVENT_SRQ_ERR:
//         WARN("NET/IB : %s:%d SRQ event: %s", dev->devName, dev->portNum,
//         str); break;
//       default:
//         WARN("NET/IB : %s:%d unknown event: %s", dev->devName, dev->portNum,
//              str);
//     }
//     if (sdcclSuccess != sdcclWrapIbvAckAsyncEvent(&event)) {
//       break;
//     }
//   }
//   return NULL;
// }
static int sdcclUcxRefCount = 0;

SDCCL_PARAM(UCXDisable, "UCX_DISABLE", 0);
/* Exclude cuda-related UCX transports */
SDCCL_PARAM(UCXCudaDisable, "UCX_CUDA_DISABLE", 1);

static const ucp_tag_t tag = 0x8a000000;
static const ucp_tag_t tagMask = (uint64_t)(-1);

sdcclResult_t sdcclUcxDevices(int *ndev) {
  *ndev = sdcclNIbDevs;
  return sdcclSuccess;
}

static __thread int
    sdcclUcxDmaSupportInitDev; // which device to init, must be thread local
static void sdcclUcxDmaBufSupportInitOnce() {
  sdcclResult_t res;
  int devFail = 0;

  // This is a physical device, not a virtual one, so select from ibDevs
  sdcclIbMergedDev *mergedDev =
      sdcclIbMergedDevs + sdcclUcxDmaSupportInitDev;
  sdcclIbDev *ibDev = sdcclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd *pd;
  struct ibv_context *ctx = ibDev->context;
  SDCCLCHECKGOTO(sdcclWrapIbvAllocPd(&pd, ctx), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)sdcclWrapDirectIbvRegMr(pd, 0ULL /*addr*/, 0ULL /*len*/, 0 /*access*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not
  // supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  SDCCLCHECKGOTO(sdcclWrapIbvDeallocPd(pd), res, failure);
  // stop the search and goto failure
  if (devFail)
    goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}
sdcclResult_t sdccl_p2p_dmabuf_support(int dev) {
  // init the device only once
  sdcclUcxDmaSupportInitDev = dev;
  pthread_once(&onces[dev].once, sdcclUcxDmaBufSupportInitOnce);
  sdcclIbMergedDev *mergedDev =
      sdcclIbMergedDevs + sdcclUcxDmaSupportInitDev;
  sdcclIbDev *ibDev = sdcclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1)
    return sdcclSuccess;
  return sdcclSystemError;
}
sdcclResult_t sdccl_p2p_gdr_support() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, ibGdrSupportInitOnce);
  if (!sdcclIbGdrModuleLoaded)
    return sdcclSystemError;
  return sdcclSuccess;
}

sdcclResult_t sdccl_p2p_ib_init(int *nDevs, int *nmDevs,
                                  sdcclIbDev *sdcclIbDevs,
                                  char *sdcclIbIfName,
                                  union sdcclSocketAddress *sdcclIbIfAddr,
                                  pthread_t *sdcclIbAsyncThread) {
  sdcclResult_t ret = sdcclSuccess;
  int sdcclNIbDevs = *nDevs;
  int sdcclNMergedIbDevs = *nmDevs;
  if (sdcclNIbDevs == -1) {
    for (int i = 0; i < MAX_IB_DEVS; i++)
      onces[i].once = PTHREAD_ONCE_INIT;
    pthread_mutex_lock(&sdccl_p2p_lock);
    sdcclWrapIbvForkInit();
    if (sdcclNIbDevs == -1) {
      int nIpIfs = 0;
      sdcclNIbDevs = 0;
      sdcclNMergedIbDevs = 0;
      nIpIfs = sdcclFindInterfaces(sdcclIbIfName, sdcclIbIfAddr,
                                    MAX_IF_NAME_SIZE, 1);
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = sdcclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device **devices;
      // Check if user defined which IB device:port to use
      const char *userIbEnv = sdcclGetEnv("SDCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot)
        userIbEnv++;
      int searchExact = userIbEnv && userIbEnv[0] == '=';
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
        enum sdcclIbProvider ibProvider = IB_PROVIDER_NONE;
        char dataDirectDevicePath[PATH_MAX];
        int dataDirectSupported = 0;
        int skipNetDevForDataDirect = 0;
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        if (sdcclSuccess != sdcclWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (sdcclSuccess != sdcclWrapIbvCloseDevice(context)) {
            ret = sdcclInternalError;
            goto fail;
          }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          for (int dataDirect = skipNetDevForDataDirect;
               dataDirect < 1 + dataDirectSupported; ++dataDirect) {
            struct ibv_port_attr portAttr;
            uint32_t portSpeed;
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
            sdcclIbDevs[sdcclNIbDevs].ibProvider = ibProvider;
            sdcclIbDevs[sdcclNIbDevs].guid = devAttr.sys_image_guid;
            sdcclIbDevs[sdcclNIbDevs].portAttr = portAttr;
            sdcclIbDevs[sdcclNIbDevs].portNum = port_num;
            sdcclIbDevs[sdcclNIbDevs].link = portAttr.link_layer;
#if HAVE_STRUCT_IBV_PORT_ATTR_ACTIVE_SPEED_EX
            portSpeed = portAttr.active_speed_ex ? portAttr.active_speed_ex
                                                 : portAttr.active_speed;
#else
            portSpeed = portAttr.active_speed;
#endif
            sdcclIbDevs[sdcclNIbDevs].speed =
                sdccl_p2p_ib_speed(portSpeed) *
                sdccl_p2p_ib_width(portAttr.active_width);
            sdcclIbDevs[sdcclNIbDevs].context = context;
            sdcclIbDevs[sdcclNIbDevs].pdRefs = 0;
            sdcclIbDevs[sdcclNIbDevs].pd = NULL;
            if (!dataDirect) {
              strncpy(sdcclIbDevs[sdcclNIbDevs].devName, devices[d]->name,
                      MAXNAMESIZE);
              SDCCLCHECKGOTO(
                  sdcclP2pIbPciPath(sdcclIbDevs, sdcclNIbDevs,
                                     sdcclIbDevs[sdcclNIbDevs].devName,
                                     &sdcclIbDevs[sdcclNIbDevs].pciPath,
                                     &sdcclIbDevs[sdcclNIbDevs].realPort),
                  ret, fail);
            } else {
              snprintf(sdcclIbDevs[sdcclNIbDevs].devName, MAXNAMESIZE,
                       "%.*s_dma", (int)(MAXNAMESIZE - 5), devices[d]->name);
              sdcclIbDevs[sdcclNIbDevs].pciPath = (char *)malloc(PATH_MAX);
              strncpy(sdcclIbDevs[sdcclNIbDevs].pciPath, dataDirectDevicePath,
                      PATH_MAX);
              sdcclIbDevs[sdcclNIbDevs].capsProvider.mlx5.dataDirect = 1;
            }
            sdcclIbDevs[sdcclNIbDevs].maxQp = devAttr.max_qp;
            sdcclIbDevs[sdcclNIbDevs].mrCache.capacity = 0;
            sdcclIbDevs[sdcclNIbDevs].mrCache.population = 0;
            sdcclIbDevs[sdcclNIbDevs].mrCache.slots = NULL;
            SDCCLCHECK(sdcclIbStatsInit(&sdcclIbDevs[sdcclNIbDevs].stats));

            // Enable ADAPTIVE_ROUTING by default on IB networks
            // But allow it to be overloaded by an env parameter
            sdcclIbDevs[sdcclNIbDevs].ar =
                (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
            if (sdcclParamIbAdaptiveRouting() != -2)
              sdcclIbDevs[sdcclNIbDevs].ar = sdcclParamIbAdaptiveRouting();

            TRACE(SDCCL_NET,
                  "NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p "
                  "pciPath=%s ar=%d",
                  d, devices[d]->name, devices[d]->dev_name,
                  sdcclIbDevs[sdcclNIbDevs].portNum,
                  SDCCL_IB_LLSTR(portAttr.link_layer),
                  ibProviderName[sdcclIbDevs[sdcclNIbDevs].ibProvider],
                  sdcclIbDevs[sdcclNIbDevs].speed, context,
                  sdcclIbDevs[sdcclNIbDevs].pciPath,
                  sdcclIbDevs[sdcclNIbDevs].ar);
            if (sdcclIbAsyncThread != NULL) {
              PTHREADCHECKGOTO(pthread_create(sdcclIbAsyncThread, NULL,
                                              sdcclIbAsyncThreadMain,
                                              sdcclIbDevs + sdcclNIbDevs),
                               "pthread_create", ret, fail);
              sdcclSetThreadName(*sdcclIbAsyncThread, "sdccl IbAsync %2d",
                                  sdcclNIbDevs);
              PTHREADCHECKGOTO(pthread_detach(*sdcclIbAsyncThread),
                               "pthread_detach", ret,
                               fail); // will not be pthread_join()'d
            }

            // Add this plain physical device to the list of virtual devices
            int vDev;
            sdcclNetVDeviceProps_t vProps = {0};
            vProps.ndevs = 1;
            vProps.devs[0] = sdcclNIbDevs;
            SDCCLCHECK(sdcclIbMakeVDeviceInternal(
                &vDev, &vProps, sdcclNIbDevs, &sdcclNMergedIbDevs));

            sdcclNIbDevs++;
            nPorts++;
          }
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
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
#ifdef HAVE_IB_PLUGIN
    sdcclIbRelaxedOrderingEnabled = sdcclIbRelaxedOrderingCapable();
#endif
    for (int d = 0; d < sdcclNIbDevs; d++) {
      snprintf(line + strlen(line), sizeof(line) - strlen(line),
               " [%d]%s:%d/%s", d, sdcclIbDevs[d].devName,
               sdcclIbDevs[d].portNum, SDCCL_IB_LLSTR(sdcclIbDevs[d].link));
    }
    char addrline[SOCKET_NAME_MAXLEN + 1];
    INFO(SDCCL_INIT | SDCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line,
#ifdef HAVE_IB_PLUGIN
         sdcclIbRelaxedOrderingEnabled ? "[RO]" : ""
#else
         ""
#endif
         ,
         sdcclIbIfName, sdcclSocketToString(sdcclIbIfAddr, addrline, 1));
    *nDevs = sdcclNIbDevs;
    *nmDevs = sdcclNMergedIbDevs;
    pthread_mutex_unlock(&sdccl_p2p_lock);
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&sdccl_p2p_lock);
  goto exit;
}
sdcclResult_t sdcclIbGetPhysProperties(int dev,
                                         sdcclNetProperties_t *props) {
  struct sdcclIbDev *ibDev = sdcclIbDevs + dev;
  pthread_mutex_lock(&ibDev->lock);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport = SDCCL_PTR_HOST;
  if (sdccl_p2p_gdr_support() == sdcclSuccess) {
    props->ptrSupport |= SDCCL_PTR_CUDA; // GDR support via nv_peermem
    INFO(SDCCL_NET,
         "NET/IB : GPU Direct RDMA (nvidia-peermem) enabled for HCA %d '%s",
         dev, ibDev->devName);
  }
  props->regIsGlobal = 1;
  if (sdccl_p2p_dmabuf_support(dev) == sdcclSuccess) {
    props->ptrSupport |= SDCCL_PTR_DMABUF; // GDR support via DMA-BUF
    INFO(SDCCL_NET, "NET/IB : GPU Direct RDMA (DMABUF) enabled for HCA %d '%s",
         dev, ibDev->devName);
  }

  props->latency = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;
  props->maxRecvs = SDCCL_NET_IB_MAX_RECVS;
  props->netDeviceType = SDCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = SDCCL_NET_DEVICE_INVALID_VERSION;
  pthread_mutex_unlock(&ibDev->lock);
  return sdcclSuccess;
}
sdcclResult_t sdccl_p2p_ib_get_properties(sdcclIbDev *devs,
                                            int sdcclNMergedIbDevs, int dev,
                                            sdcclNetProperties_t *props) {
  if (dev >= sdcclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been "
         "created",
         dev, sdcclNMergedIbDevs);
    return sdcclInvalidUsage;
  }
  struct sdcclIbMergedDev *mergedDev = sdcclIbMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the
  // same)
  SDCCLCHECK(sdcclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  return sdcclSuccess;
}
sdcclResult_t sdcclUcxGetProperties(int dev, void *props) {
  return sdccl_p2p_ib_get_properties(sdcclIbDevs, sdcclNMergedIbDevs, dev,
                                      (sdcclNetProperties_t *)props);
}

pthread_mutex_t sdcclUcxLock = PTHREAD_MUTEX_INITIALIZER;

static ucp_tag_t sdcclUcxWorkerTags[MAX_IB_DEVS];
static ucp_context_h sdcclUcxCtx[MAX_IB_DEVS];
static struct sdcclUcxWorker *sdcclUcxWorkers[MAX_IB_DEVS];
static int sdcclUcxWorkerCount = 0;

static void send_handler_nbx(void *request, ucs_status_t status,
                             void *userData) {
  int *pending = (int *)userData;

  assert(status == UCS_OK);
  assert(*pending > 0);
  (*pending)--;
  ucp_request_free(request);
}

static void recv_handler_nbx(void *request, ucs_status_t status,
                             const ucp_tag_recv_info_t *tagInfo,
                             void *userData) {
  send_handler_nbx(request, status, userData);
}

static union sdcclSocketAddress sdcclUcxIfAddr;
static char if_name[MAX_IF_NAME_SIZE];

static sdcclResult_t sdcclUcxConfigNoCuda(ucp_config_t *config) {
  char tmp[PATH_MAX];
  const char *sdcclUcxTls;
  ssize_t n;

  sdcclUcxTls = getenv("SDCCL_UCX_TLS");
  if (sdcclUcxTls == NULL) {
    sdcclUcxTls = getenv("UCX_TLS");
  }

  if (sdcclUcxTls == NULL) {
    sdcclUcxTls = "^cuda";
  } else if (sdcclUcxTls[0] == '^') {
    /* Negative expression, make sure to keep cuda excluded */
    n = snprintf(tmp, sizeof(tmp), "^cuda,%s", &sdcclUcxTls[1]);
    if (n >= sizeof(tmp)) {
      return sdcclInternalError;
    }

    sdcclUcxTls = tmp;
  } else {
    /* Positive expression cannot allow cuda-like transports */
    if ((strstr(sdcclUcxTls, "cuda") != NULL) ||
        (strstr(sdcclUcxTls, "gdr") != NULL)) {
      WARN("Cannot use cuda/gdr transports as part of specified UCX_TLS");
      return sdcclInternalError;
    }
  }

  UCXCHECK(ucp_config_modify(config, "TLS", sdcclUcxTls));
  UCXCHECK(ucp_config_modify(config, "RNDV_THRESH", "0"));
  UCXCHECK(ucp_config_modify(config, "RNDV_SCHEME", "get_zcopy"));
  UCXCHECK(
      ucp_config_modify(config, "MEMTYPE_REG_WHOLE_ALLOC_TYPES", "unknown"));
  return sdcclSuccess;
}

static sdcclResult_t sdcclUcxInitContext(ucp_context_h *ctx, int dev) {
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char sdcclUcxDevName[PATH_MAX];
  sdcclResult_t result;

  if (sdcclUcxCtx[dev] == NULL) {
    plugin_get_device_name(sdcclIbDevs[dev].devName, sdcclUcxDevName, 64);
    snprintf(sdcclUcxDevName + strlen(sdcclUcxDevName),
             PATH_MAX - strlen(sdcclUcxDevName), ":%d",
             sdcclIbDevs[dev].portNum);

    UCXCHECK(ucp_config_read("SDCCL", NULL, &config));
    UCXCHECK(ucp_config_modify(config, "NET_DEVICES", sdcclUcxDevName));

    if (sdcclParamUCXCudaDisable()) {
      result = sdcclUcxConfigNoCuda(config);
      if (result != sdcclSuccess) {
        return result;
      }
    }

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

    UCXCHECK(ucp_init(&ucp_params, config, &sdcclUcxCtx[dev]));
    ucp_config_release(config);
  }

  *ctx = sdcclUcxCtx[dev];

  return sdcclSuccess;
}

static sdcclResult_t sdcclUcxInitWorker(ucp_context_h ctx,
                                          ucp_worker_h *worker) {
  ucp_worker_params_t worker_params;
  ucp_worker_attr_t worker_attr;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));

  worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
  ucp_worker_query(*worker, &worker_attr);
  if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
    INFO(SDCCL_NET, "Thread mode multi is not supported");
  }

  return sdcclSuccess;
}

static sdcclResult_t sdcclUcxWorkerGetNetaddress(ucp_worker_h worker,
                                                   ucp_address_t **address,
                                                   size_t *addressLength) {
  ucp_worker_attr_t attr;

  attr.field_mask =
      UCP_WORKER_ATTR_FIELD_ADDRESS | UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = (ucp_address_t *)malloc(attr.address_length);
  if (address == NULL) {
    return sdcclSystemError;
  }

  memcpy(*address, attr.address, attr.address_length);
  *address_length = attr.address_length;
  free(attr.address);

  return sdcclSuccess;
}

static sdcclResult_t sdcclUcxGetCtxAndWorker(int dev, ucp_context_h *ctx,
                                               sdcclUcxWorker_t **ucxWorker,
                                               ucp_tag_t *newtag) {
  pthread_mutex_lock(&sdcclUcxLock);
  sdcclResult_t result;

  if (sdcclNIbDevs <= dev) {
    WARN("Device index is too large");
    goto err;
  }

  sdcclUcxWorker_t *w;
  for (w = sdcclUcxWorkers[dev]; w != NULL; w = w->next) {
    assert(w->dev == dev);
    if (w->thread == pthread_self()) {
      break;
    }
  }

  if (w == NULL) {
    w = (sdcclUcxWorker_t *)calloc(1, sizeof(*w));
    if (w == NULL) {
      WARN("Worker allocation failure");
      goto err;
    }

    w->dev = dev;
    w->thread = pthread_self();
    w->count = 0;

    result = sdcclUcxInitContext(&w->ctx, dev);
    if (result != sdcclSuccess) {
      return result;
    }
    sdcclUcxInitWorker(w->ctx, &w->worker);
    sdcclUcxWorkerCount++;

    w->next = sdcclUcxWorkers[dev];
    sdcclUcxWorkers[dev] = w;
  }

  *ctx = w->ctx;
  *ucxWorker = w;
  if (newtag != NULL) {
    *newtag = ++sdcclUcxWorkerTags[dev];
  }

  ucp_worker_progress(w->worker);
  w->count++;
  pthread_mutex_unlock(&sdcclUcxLock);
  return sdcclSuccess;

err:
  pthread_mutex_unlock(&sdcclUcxLock);
  return sdcclSystemError;
}

static sdcclResult_t sdcclUcxFreeWorker(sdcclUcxWorker_t *ucxWorker) {
  int dev, dummy, done = 0;
  struct sdcclUcxEpList *ep, *cur;
  struct sdcclUcxWorker *next;
  sdcclResult_t result;

  pthread_mutex_lock(&sdcclUcxLock);
  ucxWorker->count--;
  if (ucxWorker->count == 0) {
    sdcclUcxWorkerCount--;
    done = sdcclUcxWorkerCount == 0;
  }
  pthread_mutex_unlock(&sdcclUcxLock);

  if (!done) {
    return sdcclSuccess;
  }

  for (dev = 0; dev < sizeof(sdcclUcxWorkers) / sizeof(*sdcclUcxWorkers);
       dev++) {
    for (ucxWorker = sdcclUcxWorkers[dev]; ucxWorker != NULL;
         ucxWorker = next) {
      next = ucxWorker->next;
      assert(ucxWorker->count == 0);

      ep = ucxWorker->eps;
      while (ep) {
        cur = ep;
        result = sdcclSocketRecv(ep->sock, &dummy, sizeof(int));
        if (result != sdcclSuccess) {
          WARN("Failed to receive close for worker cleanup (res:%d)", result);
        }

        ep = ep->next;
        close(cur->sock->fd);
        free(cur);
      }
      ucp_worker_destroy(ucxWorker->worker);
      free(ucxWorker);
    }

    sdcclUcxWorkers[dev] = NULL;
    if (sdcclUcxCtx[dev]) {
      ucp_cleanup(sdcclUcxCtx[dev]);
      sdcclUcxCtx[dev] = NULL;
    }
  }

  return sdcclSuccess;
}

static sdcclResult_t sdcclUcxAddEp(sdcclUcxWorker_t *ucxWorker,
                                     struct sdcclSocket *sock) {
  struct sdcclUcxEpList *newEp =
      (struct sdcclUcxEpList *)malloc(sizeof(struct sdcclUcxEpList));
  if (newEp == NULL) {
    return sdcclSystemError;
  }

  newEp->sock = sock;
  newEp->next = ucxWorker->eps;
  ucxWorker->eps = newEp;
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxInit() {
  if (sdcclUcxRefCount++)
    return sdcclSuccess;
  if (sdcclParamUCXDisable())
    return sdcclInternalError;

  for (int i = 0;
       i < sizeof(sdcclUcxWorkerTags) / sizeof(*sdcclUcxWorkerTags); i++) {
    sdcclUcxWorkerTags[i] = tag;
  }

  // Initialize InfiniBand symbols first
  if (sdcclWrapIbvSymbols() != sdcclSuccess) {
    WARN("NET/UCX: Failed to initialize InfiniBand symbols");
    return sdcclInternalError;
  }

  return sdccl_p2p_ib_init(&sdcclNIbDevs, &sdcclNMergedIbDevs, sdcclIbDevs,
                            if_name, &sdcclUcxIfAddr, NULL);
}

sdcclResult_t sdcclUcxListen(int dev, void *handle, void **listen_comm) {
  sdcclUcxListenHandle_t *my_handle = (sdcclUcxListenHandle_t *)handle;
  sdcclUcxListenComm_t *comm =
      (sdcclUcxListenComm_t *)calloc(1, sizeof(*comm));

  SDCCL_STATIC_ASSERT(sizeof(sdcclUcxListenHandle_t) <
                           SDCCL_NET_HANDLE_MAXSIZE,
                       "UCX listen handle size too large");
  my_handle->magic = SDCCL_SOCKET_MAGIC;
  SDCCLCHECK(sdcclSocketInit(&comm->sock, &sdcclUcxIfAddr, my_handle->magic,
                               sdcclSocketTypeNetIb, NULL, 1));
  SDCCLCHECK(sdcclSocketListen(&comm->sock));
  SDCCLCHECK(sdcclSocketGetAddr(&comm->sock, &my_handle->connectAddr));
  SDCCLCHECK(
      sdcclUcxGetCtxAndWorker(dev, &comm->ctx, &comm->ucxWorker, &comm->tag));

  comm->dev = dev;
  my_handle->tag = comm->tag;
  *listen_comm = comm;

  return sdcclSuccess;
}

static void sdcclUcxRequestInit(sdcclUcxComm_t *comm) {
  static const int entries = sizeof(comm->reqs) / sizeof(*comm->reqs);

  comm->freeReq = NULL;
  for (int i = entries - 1; i >= 0; i--) {
    comm->reqs[i].comm = comm;
    comm->reqs[i].next = comm->freeReq;
    comm->freeReq = &comm->reqs[i];
  }
}

sdcclResult_t sdcclUcxConnect(int dev, void *handle, void **send_comm) {
  sdcclUcxListenHandle_t *recv_handle = (sdcclUcxListenHandle_t *)handle;
  struct sdcclUcxCommStage *stage = &recv_handle->stage;
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)stage->comm;
  ucp_address_t *my_addr;
  size_t localAddrLen;
  int ready;

  *send_comm = NULL;

  if (stage->state == sdcclUcxCommStateConnect)
    goto sdcclUcxConnectCheck;

  SDCCLCHECK(sdcclIbMalloc((void **)&comm, sizeof(sdcclUcxComm_t)));
  SDCCLCHECK(sdcclSocketInit(&comm->sock, &recv_handle->connectAddr,
                               recv_handle->magic, sdcclSocketTypeNetIb, NULL,
                               1));
  stage->comm = comm;
  stage->state = sdcclUcxCommStateConnect;
  SDCCLCHECK(sdcclSocketConnect(&comm->sock));
  sdcclUcxRequestInit(comm);

sdcclUcxConnectCheck:
  /* since sdcclSocketConnect is async, we must check if connection is complete
   */
  SDCCLCHECK(sdcclSocketReady(&comm->sock, &ready));
  if (!ready)
    return sdcclSuccess;

  SDCCLCHECK(
      sdcclUcxGetCtxAndWorker(dev, &comm->ctx, &comm->ucxWorker, &comm->ctag));
  comm->tag = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  SDCCLCHECK(sdcclUcxWorkerGetNetaddress(comm->ucxWorker->worker, &my_addr,
                                           &localAddrLen));
  SDCCLCHECK(sdcclUcxAddEp(comm->ucxWorker, &comm->sock));
  TRACE(SDCCL_NET, "NET/UCX: Worker address length: %zu", localAddrLen);

  SDCCLCHECK(sdcclSocketSend(&comm->sock, &localAddrLen, sizeof(size_t)));
  SDCCLCHECK(sdcclSocketSend(&comm->sock, my_addr, localAddrLen));
  SDCCLCHECK(sdcclSocketSend(&comm->sock, &comm->ctag, sizeof(ucp_tag_t)));

  *send_comm = comm;
  free(my_addr);
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxAccept(void *listen_comm, void **recv_comm) {
  sdcclUcxListenComm_t *l_comm = (sdcclUcxListenComm_t *)listen_comm;
  struct sdcclUcxCommStage *stage = &l_comm->stage;
  sdcclUcxComm_t *r_comm = (sdcclUcxComm_t *)stage->comm;
  size_t peerAddrLen;
  ucp_address_t *peer_addr;
  ucp_ep_params_t ep_params;
  int ready;

  *recv_comm = NULL;
  if (stage->state == sdcclUcxCommStateAccept)
    goto sdcclUcxAcceptCheck;

  SDCCLCHECK(sdcclIbMalloc((void **)&r_comm, sizeof(sdcclUcxComm_t)));
  stage->comm = r_comm;
  stage->state = sdcclUcxCommStateAccept;
  l_comm->sock.asyncFlag = 1;
  r_comm->sock.asyncFlag = 1;

  SDCCLCHECK(sdcclSocketInit(&r_comm->sock, NULL, SDCCL_SOCKET_MAGIC,
                               sdcclSocketTypeUnknown, NULL, 0));
  SDCCLCHECK(sdcclSocketAccept(&r_comm->sock, &l_comm->sock));
sdcclUcxAcceptCheck:
  SDCCLCHECK(sdcclSocketReady(&r_comm->sock, &ready));
  if (!ready)
    return sdcclSuccess;

  r_comm->ctx = l_comm->ctx;
  r_comm->ucxWorker = l_comm->ucxWorker;
  r_comm->tag = l_comm->tag;

  sdcclUcxRequestInit(r_comm);

  SDCCLCHECK(sdcclSocketRecv(&r_comm->sock, &peerAddrLen, sizeof(size_t)));
  peer_addr = (ucp_address_t *)malloc(peerAddrLen);
  if (peer_addr == NULL) {
    return sdcclSystemError;
  }

  SDCCLCHECK(sdcclSocketRecv(&r_comm->sock, peer_addr, peerAddrLen));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = peer_addr;
  UCXCHECK(ucp_ep_create(r_comm->ucxWorker->worker, &ep_params, &r_comm->ep));
  SDCCLCHECK(
      sdcclSocketRecv(&r_comm->sock, &r_comm->ctag, sizeof(ucp_tag_t)));

  r_comm->gpuFlush.enabled = (sdccl_p2p_gdr_support() == sdcclSuccess);
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t localAddrLen;

    SDCCLCHECK(sdcclUcxWorkerGetNetaddress(r_comm->ucxWorker->worker,
                                             &my_addr, &localAddrLen));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->ucxWorker->worker, &ep_params,
                           &r_comm->gpuFlush.flushEp));
    free(my_addr);
  }

  free(peer_addr);
  *recv_comm = r_comm;

  return sdcclSuccess;
}

#define REG_ALIGN (4096)
sdcclResult_t sdcclUcxRegMr(void *comm, void *data, size_t size, int type,
                              int mrFlags, void **mhandle) {
  (void)mrFlags;
  sdcclUcxCtx_t *ctx = (sdcclUcxCtx_t *)comm;
  uint64_t addr = (uint64_t)data;
  ucp_mem_map_params_t mmap_params;
  sdcclUcxMhandle_t *mh;
  uint64_t reg_addr, reg_size;
  size_t rkeyBufSize;
  void *rkeyBuf;

  SDCCLCHECK(sdcclIbMalloc((void **)&mh, sizeof(sdcclUcxMhandle_t)));
  reg_addr = addr & (~(REG_ALIGN - 1));
  reg_size = addr + size - reg_addr;
  reg_size = ROUNDUP(reg_size, REG_ALIGN);

  mmap_params.field_mask =
      UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address = (void *)reg_addr;
  mmap_params.length = reg_size;
  mh->memType =
      (type == SDCCL_PTR_HOST) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;
  mmap_params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
  mmap_params.memory_type = (ucs_memory_type_t)mh->memType;

  UCXCHECK(ucp_mem_map(ctx->sdcclUcxCtx, &mmap_params, &mh->ucpMemh));
  if (ctx->gpuFlush.enabled) {
    UCXCHECK(
        ucp_rkey_pack(ctx->sdcclUcxCtx, mh->ucpMemh, &rkeyBuf, &rkeyBufSize));
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flushEp, rkeyBuf, &mh->rkey));
    ucp_rkey_buffer_release(rkeyBuf);
  }

  *mhandle = mh;
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxDeregMr(void *comm, void *mhandle) {
  sdcclUcxCtx_t *ctx = (sdcclUcxCtx_t *)comm;
  sdcclUcxMhandle_t *mh = (sdcclUcxMhandle_t *)mhandle;

  if (ctx->gpuFlush.enabled) {
    ucp_rkey_destroy(mh->rkey);
  }

  ucp_mem_unmap(ctx->sdcclUcxCtx, mh->ucpMemh);
  free(mhandle);

  return sdcclSuccess;
}

sdcclResult_t sdcclUcxRegMrDmaBuf(void *comm, void *data, size_t size,
                                    int type, uint64_t offset, int fd,
                                    int mrFlags, void **mhandle) {
  return sdcclUcxRegMr(comm, data, size, type, mrFlags, mhandle);
}

static sdcclUcxRequest_t *sdcclUcxRequestGet(sdcclUcxComm_t *comm) {
  sdcclUcxRequest_t *req = comm->freeReq;

  if (req == NULL) {
    WARN("NET/UCX: unable to allocate SDCCL request");
    return NULL;
  }

  comm->freeReq = req->next;
  req->worker = comm->ucxWorker->worker;
  req->pending = 0;
  req->count = 0;
  return req;
}

static void sdcclUcxRequestRelease(sdcclUcxRequest_t *req) {
  req->next = req->comm->freeReq;
  req->comm->freeReq = req;
}

static void sdcclUcxRequestAdd(sdcclUcxRequest_t *req, int size) {
  req->size[req->count] = size;
  req->pending++;
  req->count++;
}

static sdcclResult_t sdcclUcxSendCheck(sdcclUcxComm_t *comm) {
  ucp_request_param_t params;
  ucp_tag_message_h msg_tag;
  ucp_tag_recv_info_t info_tag;
  ucp_ep_params_t ep_params;
  void *ucpReq;
  ucs_status_t status;

  ucp_worker_progress(comm->ucxWorker->worker);

  if (comm->connectReq != NULL) {
    goto out_check_status;
  }

  msg_tag = ucp_tag_probe_nb(comm->ucxWorker->worker, comm->ctag, tagMask, 1,
                             &info_tag);
  if (msg_tag == NULL) {
    return sdcclSuccess;
  }

  comm->msg = (sdcclUcxConnectMsg_t *)malloc(info_tag.length);
  if (comm->msg == NULL) {
    return sdcclSystemError;
  }

  params.op_attr_mask = 0;
  ucpReq = ucp_tag_msg_recv_nbx(comm->ucxWorker->worker, comm->msg,
                                info_tag.length, msg_tag, &params);
  if (UCS_PTR_IS_ERR(ucpReq)) {
    WARN("Unable to receive connect msg (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucpReq)));
    free(comm->msg);
    comm->msg = NULL;
    return sdcclSystemError;
  } else if (ucpReq == NULL) {
    goto out_set_ready;
  }

  assert(comm->connectReq == NULL);
  comm->connectReq = ucpReq;

out_check_status:
  status = ucp_request_check_status(comm->connectReq);
  if (status == UCS_INPROGRESS) {
    return sdcclSuccess;
  }

  if (status != UCS_OK) {
    free(comm->msg);
    comm->msg = NULL;
    WARN("Send check requested returned error (%s)", ucs_status_string(status));
    return sdcclSystemError;
  }

  ucp_request_free(comm->connectReq);
  comm->connectReq = NULL;

out_set_ready:
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = (ucp_address_t *)(comm->msg + 1);
  UCXCHECK(ucp_ep_create(comm->ucxWorker->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(comm->msg);
  comm->msg = NULL;

  return sdcclSuccess;
}

static void sdcclUcxRecvSetReady(sdcclUcxComm_t *comm) {
  free(comm->msg);
  comm->msg = NULL;
  comm->ready = 1;
}

static void check_handler(void *request, ucs_status_t status, void *userData) {
  assert(status == UCS_OK);
  sdcclUcxRecvSetReady((sdcclUcxComm_t *)userData);
  ucp_request_free(request);
}

sdcclResult_t sdcclUcxRecvCheck(sdcclUcxComm_t *comm) {
  ucp_request_param_t params;
  ucp_address_t *my_addr;
  size_t localAddrLen;
  size_t msgLen;

  if (comm->connectReq != NULL) {
    goto done;
  }

  SDCCLCHECK(sdcclUcxWorkerGetNetaddress(comm->ucxWorker->worker, &my_addr,
                                           &localAddrLen));
  sdcclUcxAddEp(comm->ucxWorker, &comm->sock);

  msgLen = sizeof(sdcclUcxConnectMsg_t) + localAddrLen;
  comm->msg = (sdcclUcxConnectMsg_t *)calloc(1, msgLen);
  comm->msg->addrLen = localAddrLen;
  memcpy(comm->msg + 1, my_addr, localAddrLen);

  params.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send = check_handler;
  params.user_data = comm;

  assert(comm->connectReq == NULL);
  comm->connectReq =
      ucp_tag_send_nbx(comm->ep, comm->msg, msgLen, comm->ctag, &params);
  if (UCS_PTR_IS_ERR(comm->connectReq)) {
    WARN("Unable to send connect message");
    free(comm->msg);
    return sdcclSystemError;
  } else if (comm->connectReq == NULL) {
    sdcclUcxRecvSetReady(comm);
    return sdcclSuccess;
  }

done:
  ucp_worker_progress(comm->ucxWorker->worker);
  return sdcclSuccess;
}

static ucp_tag_t sdcclUcxUcpTag(ucp_tag_t comm_tag, uint64_t tag) {
  assert(tag <= UINT32_MAX);
  assert(comm_tag <= UINT32_MAX);
  return comm_tag + (tag << 32);
}

sdcclResult_t sdcclUcxIsend(void *send_comm, void *data, size_t size, int tag,
                              void *mhandle, void *phandle, void **request) {
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)send_comm;
  sdcclUcxMhandle_t *mh = (sdcclUcxMhandle_t *)mhandle;
  sdcclUcxRequest_t *req;
  void *ucpReq;
  ucp_request_param_t params;

  if (comm->ready == 0) {
    SDCCLCHECK(sdcclUcxSendCheck(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return sdcclSuccess;
    }
  }

  req = sdcclUcxRequestGet(comm);
  if (req == NULL) {
    return sdcclInternalError;
  }

  sdcclUcxRequestAdd(req, size);

  params.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send = send_handler_nbx;
  params.user_data = &req->pending;
  if (mh) {
    params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
    params.memh = mh->ucpMemh;
  }

  ucpReq = ucp_tag_send_nbx(comm->ep, data, size,
                            sdcclUcxUcpTag(comm->tag, tag), &params);
  if (UCS_PTR_IS_ERR(ucpReq)) {
    WARN("ucx_isend: unable to send message (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucpReq)));
    return sdcclSystemError;
  } else if (ucpReq == NULL) {
    req->pending--;
  }

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxIrecv(void *recv_comm, int n, void **data,
                              size_t *sizes, int *tags, void **mhandle,
                              void **phandles, void **request) {
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)recv_comm;
  sdcclUcxMhandle_t **mh = (sdcclUcxMhandle_t **)mhandle;
  void *ucpReq;
  sdcclUcxRequest_t *req;
  ucp_request_param_t params;

  if (comm->ready == 0) {
    SDCCLCHECK(sdcclUcxRecvCheck(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return sdcclSuccess;
    }
  }

  if (n > SDCCL_NET_IB_MAX_RECVS) {
    WARN("ucx_irecv: posting %d but max is %d", n, SDCCL_NET_IB_MAX_RECVS);
    return sdcclInternalError;
  }

  req = sdcclUcxRequestGet(comm);
  if (req == NULL) {
    return sdcclInternalError;
  }

  params.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.recv = recv_handler_nbx;
  params.user_data = &req->pending;

  for (int i = 0; i < n; i++) {
    sdcclUcxRequestAdd(req, sizes[i]);

    if (mh[i]) {
      params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
      params.memh = mh[i]->ucpMemh;
    } else {
      params.op_attr_mask &= ~UCP_OP_ATTR_FIELD_MEMH;
    }

    ucpReq =
        ucp_tag_recv_nbx(comm->ucxWorker->worker, data[i], sizes[i],
                         sdcclUcxUcpTag(comm->tag, tags[i]), tagMask, &params);
    if (UCS_PTR_IS_ERR(ucpReq)) {
      WARN("ucx_irecv: unable to post receive %d/%d (%s)", i, n,
           ucs_status_string(UCS_PTR_STATUS(ucpReq)));
      return sdcclSystemError;
    } else if (ucpReq == NULL) {
      req->pending--;
    }
  }

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxIflush(void *recv_comm, int n, void **data, int *sizes,
                               void **mhandle, void **request) {
  int last = -1;
  int size = 1;
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)recv_comm;
  sdcclUcxMhandle_t **mh = (sdcclUcxMhandle_t **)mhandle;
  sdcclUcxRequest_t *req;
  void *ucpReq;
  ucp_request_param_t params;

  *request = NULL;
  for (int i = 0; i < n; i++)
    if (sizes[i])
      last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1)
    return sdcclSuccess;

  req = sdcclUcxRequestGet(comm);
  if (req == NULL) {
    return sdcclInternalError;
  }

  sdcclUcxRequestAdd(req, size);

  params.op_attr_mask =
      UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send = send_handler_nbx;
  params.user_data = &req->pending;
  ucpReq = ucp_get_nbx(comm->gpuFlush.flushEp, &comm->gpuFlush.hostMem, size,
                       (uint64_t)data[last], mh[last]->rkey, &params);
  if (UCS_PTR_IS_ERR(ucpReq)) {
    WARN("ucx_iflush: unable to read data (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucpReq)));
    return sdcclSystemError;
  } else if (ucpReq == NULL) {
    req->pending--;
  }

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclUcxTest(void *request, int *done, int *size) {
  sdcclUcxRequest_t *req = (sdcclUcxRequest_t *)request;
  unsigned p;

  while (req->pending > 0) {
    p = ucp_worker_progress(req->worker);
    if (!p) {
      *done = 0;
      return sdcclSuccess;
    }
  }

  *done = 1;
  if (size != NULL) {
    /* Posted receives have completed */
    memcpy(size, req->size, sizeof(*size) * req->count);
  }

  sdcclUcxRequestRelease(req);
  return sdcclSuccess;
}

static void wait_close(ucp_worker_h worker, void *ucpReq) {
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(ucpReq)) {
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(ucpReq);
    } while (status == UCS_INPROGRESS);
    ucp_request_free(ucpReq);
  } else if (ucpReq != NULL) {
    WARN("Failed to close UCX endpoint");
  }
}

sdcclResult_t sdcclUcxCloseSend(void *send_comm) {
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)send_comm;
  void *closeReq;

  if (comm) {
    if (comm->ep) {
      closeReq = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucxWorker->worker, closeReq);
      int close = 1;
      SDCCLCHECK(sdcclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    sdcclUcxFreeWorker(comm->ucxWorker);
    free(comm);
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclUcxCloseRecv(void *recv_comm) {
  sdcclUcxComm_t *comm = (sdcclUcxComm_t *)recv_comm;
  void *closeReq;

  if (comm) {
    if (comm->gpuFlush.enabled) {
      closeReq =
          ucp_ep_close_nb(comm->gpuFlush.flushEp, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucxWorker->worker, closeReq);
    }
    if (comm->ep) {
      closeReq = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucxWorker->worker, closeReq);
      int close = 1;
      SDCCLCHECK(sdcclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    sdcclUcxFreeWorker(comm->ucxWorker);
    free(comm);
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclUcxCloseListen(void *listen_comm) {
  sdcclUcxListenComm_t *comm = (sdcclUcxListenComm_t *)listen_comm;

  if (comm) {
    close(comm->sock.fd);
    free(comm);
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclUcxFinalize(void *ctx) {
  sdcclUcxRefCount--;
  return sdcclSuccess;
}

// Additional functions needed for the adaptor interface
sdcclResult_t sdcclUcxGetDevFromName(char *name, int *dev) {
  // Simple implementation - find device by name
  for (int i = 0; i < sdcclNIbDevs; i++) {
    if (strcmp(sdcclIbDevs[i].devName, name) == 0) {
      *dev = i;
      return sdcclSuccess;
    }
  }
  return sdcclSystemError;
}

// UCX network adaptor structure
struct sdcclNetAdaptor sdcclNetUcx = {
    // Basic functions
    "UCX", sdcclUcxInit, sdcclUcxDevices, sdcclUcxGetProperties,

    // Setup functions
    sdcclUcxListen,      // listen
    sdcclUcxConnect,     // connect
    sdcclUcxAccept,      // accept
    sdcclUcxCloseSend,   // closeSend
    sdcclUcxCloseRecv,   // closeRecv
    sdcclUcxCloseListen, // closeListen

    // Memory region functions
    sdcclUcxRegMr,       // regMr
    sdcclUcxRegMrDmaBuf, // regMrDmaBuf
    sdcclUcxDeregMr,     // deregMr

    // Two-sided functions
    sdcclUcxIsend,  // isend
    sdcclUcxIrecv,  // irecv
    sdcclUcxIflush, // iflush
    sdcclUcxTest,   // test

    // One-sided functions
    NULL, // iput - not supported on UCX
    NULL, // iget - not supported on UCX
    NULL, // iputSignal - not supported on UCX

    // Device name lookup
    sdcclUcxGetDevFromName // getDevFromName
};

#endif // USE_UCX