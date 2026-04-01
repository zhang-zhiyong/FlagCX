/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef USE_IBUC

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
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static void sdcclIbucPollCompletions(struct sdcclIbSendComm *comm) {
  if (!comm)
    return;

  struct ibv_wc wcs[64];
  int retransFreed = 0;

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct sdcclIbNetCommDevBase *devBase = &comm->devs[i].base;
    if (!devBase || !devBase->cq)
      continue;
    for (int pollRound = 0; pollRound < 16; pollRound++) {
      int nCqe = 0;
      sdcclWrapIbvPollCq(devBase->cq, 64, wcs, &nCqe);

      if (nCqe == 0)
        break;
      for (int j = 0; j < nCqe; j++) {
        if (wcs[j].status == IBV_WC_SUCCESS &&
            wcs[j].wr_id == SDCCL_RETRANS_WR_ID) {
          comm->outstandingRetrans--;
          retransFreed++;
        }
      }
    }
  }

  // Statistics logging disabled
}

static sdcclResult_t sdcclIbucTestPreCheck(struct sdcclIbRequest *r) {
  if (!r)
    return sdcclInternalError;

  if (r->type == SDCCL_NET_IB_REQ_SEND && r->base->isSend) {
    struct sdcclIbSendComm *sComm = (struct sdcclIbSendComm *)r->base;

    static __thread uint64_t lastLogTime = 0;
    uint64_t nowUs = sdcclIbGetTimeUs();
    if (nowUs - lastLogTime > 100000) {
      lastLogTime = nowUs;
    }

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
              sdcclResult_t pollResult = sdcclIbRetransRecvAckViaUd(sComm, i);
              if (pollResult != sdcclSuccess)
                break;
            }
          } else {
            TRACE(SDCCL_NET,
                  "IBUC Retrans: Skipping ACK poll (QP not ready) devIndex=%d",
                  i);
          }
        }

        uint64_t nowUs2 = sdcclIbGetTimeUs();
        const uint64_t CHECK_INTERVAL_US = 1000;
        if (nowUs2 - sComm->lastTimeoutCheckUs >= CHECK_INTERVAL_US) {
          sdcclResult_t retransResult =
              sdcclIbRetransCheckTimeout(&sComm->retrans, sComm);
          if (retransResult != sdcclSuccess &&
              retransResult != sdcclInProgress) {
            if (sdcclDebugNoWarn == 0)
              INFO(SDCCL_ALL,
                   "%s:%d -> %d (retransmission check failed, continuing)",
                   __FILE__, __LINE__, retransResult);
          }
          sComm->lastTimeoutCheckUs = nowUs2;
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

static sdcclResult_t sdcclIbucProcessWc(struct sdcclIbRequest *r,
                                          struct ibv_wc *wc, int devIndex,
                                          bool *handled) {
  if (!r || !wc || !handled)
    return sdcclInternalError;
  *handled = false;
  if (r->type == SDCCL_NET_IB_REQ_SEND && r->base->isSend &&
      wc->wr_id != SDCCL_RETRANS_WR_ID) {
  }

  if (wc->wr_id == SDCCL_RETRANS_WR_ID) {
    if (r->base->isSend) {
      struct sdcclIbSendComm *sComm = (struct sdcclIbSendComm *)r->base;
      sComm->outstandingRetrans--;
      TRACE(SDCCL_NET, "SEND retrans completed, outstanding_retrans=%d",
            sComm->outstandingRetrans);
    }
    *handled = true;
    return sdcclSuccess;
  }

  if (!r->base->isSend) {
    struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)r->base;

    if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
      int bufIdx = (int)wc->wr_id;

      if (bufIdx < 0 || bufIdx >= rComm->srqMgr.bufCount) {
        WARN("SRQ completion with invalid buffer index: %d (max=%d)", bufIdx,
             rComm->srqMgr.bufCount);
        *handled = true;
        return sdcclSuccess;
      }

      void *chunkAddr = rComm->srqMgr.bufs[bufIdx].buffer;

      if (wc->opcode == IBV_WC_RECV) {
        struct sdcclIbRetransHdr *hdr = (struct sdcclIbRetransHdr *)chunkAddr;
        if (hdr->magic == SDCCL_RETRANS_MAGIC) {
          uint32_t seq = hdr->seq;

          struct sdcclIbAckMsg ackMsg = {0};
          int shouldAck = 0;
          SDCCLCHECK(sdcclIbRetransRecvPacket(&rComm->retrans, seq, &ackMsg,
                                                &shouldAck));

          if (shouldAck) {
            sdcclResult_t ackResult =
                sdcclIbRetransSendAckViaUd(rComm, &ackMsg, 0);
            if (ackResult != sdcclSuccess) {
              TRACE(SDCCL_NET, "Failed to send ACK for seq=%u (result=%d)",
                    seq, ackResult);
            } else {
              TRACE(SDCCL_NET, "Sent ACK for seq=%u, ack_seq=%u", seq,
                    ackMsg.ackSeq);
            }
          }

          TRACE(SDCCL_NET, "Received SEND retransmission from SRQ: seq=%u",
                seq);
        }

        rComm->srqMgr.bufs[bufIdx].inUse = 0;
        rComm->srqMgr.freeBufIndices[rComm->srqMgr.freeBufCount] = bufIdx;
        rComm->srqMgr.freeBufCount++;
        rComm->srqMgr.postSrqCount++;
        *handled = true;
        return sdcclSuccess;
      }

      if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        if (r->type == SDCCL_NET_IB_REQ_RECV && r->nreqs == 1) {
          if (rComm->retrans.enabled && rComm->devs[0].ctrlQp.qp != NULL) {
            uint32_t seq, size;
            sdcclIbDecodeImmData(wc->imm_data, &seq, &size);
            r->recv.sizes[0] = size;

            struct sdcclIbAckMsg ackMsg2 = {0};
            int shouldAck2 = 0;
            SDCCLCHECK(sdcclIbRetransRecvPacket(&rComm->retrans, seq,
                                                  &ackMsg2, &shouldAck2));

            if (shouldAck2) {
              sdcclResult_t ackResult2 =
                  sdcclIbRetransSendAckViaUd(rComm, &ackMsg2, 0);
              if (ackResult2 != sdcclSuccess) {
                TRACE(SDCCL_NET, "Failed to send ACK for seq=%u (result=%d)",
                      seq, ackResult2);
              } else {
                TRACE(SDCCL_NET, "Sent ACK for seq=%u, ack_seq=%u", seq,
                      ackMsg2.ackSeq);
              }
            } else {
              TRACE(SDCCL_NET, "No ACK needed for seq=%u (expect=%u)", seq,
                    rComm->retrans.recvSeq);
            }
          } else {
            r->recv.sizes[0] = wc->imm_data;
          }
        }

        rComm->srqMgr.bufs[bufIdx].inUse = 0;
        rComm->srqMgr.freeBufIndices[rComm->srqMgr.freeBufCount] = bufIdx;
        rComm->srqMgr.freeBufCount++;
        rComm->srqMgr.postSrqCount++;
        r->events[devIndex]--;
        *handled = true;
        return sdcclSuccess;
      }
    }
  }

  return sdcclSuccess;
}

sdcclResult_t sdcclIbucInit() {
  sdcclResult_t ret;
  if (sdcclParamIbDisable()) {
    return sdcclInternalError;
  }
  static int shownIbucHcaEnv = 0;
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
        WARN("NET/IBUC : No IP interface found.");
        ret = sdcclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbucDevs;
      struct ibv_device **devices;

      // Check if user defined which IBUC device:port to use
      char *userIbucEnv = getenv("SDCCL_IB_HCA");
      if (userIbucEnv != NULL && shownIbucHcaEnv++ == 0)
        INFO(SDCCL_NET | SDCCL_ENV, "SDCCL_IB_HCA set to %s", userIbucEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbucEnv && userIbucEnv[0] == '^';
      if (searchNot)
        userIbucEnv++;
      bool searchExact = userIbucEnv && userIbucEnv[0] == '=';
      if (searchExact)
        userIbucEnv++;
      int nUserIfs = parseStringList(userIbucEnv, userIfs, MAX_IB_DEVS);

      if (sdcclSuccess != sdcclWrapIbvGetDeviceList(&devices, &nIbucDevs)) {
        ret = sdcclInternalError;
        goto fail;
      }

      for (int d = 0; d < nIbucDevs && sdcclNIbDevs < MAX_IB_DEVS; d++) {
        struct ibv_context *context;
        if (sdcclSuccess != sdcclWrapIbvOpenDevice(&context, devices[d]) ||
            context == NULL) {
          WARN("NET/IBUC : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (sdcclSuccess != sdcclWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IBUC : Unable to query device %s", devices[d]->name);
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
            WARN("NET/IBUC : Unable to query port_num %d", port_num);
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

          // Enable ADAPTIVE_ROUTING by default on IBUC networks
          // But allow it to be overloaded by an env parameter
          sdcclIbDevs[sdcclNIbDevs].ar =
              (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (sdcclParamIbAdaptiveRouting() != -2)
            sdcclIbDevs[sdcclNIbDevs].ar = sdcclParamIbAdaptiveRouting();

          TRACE(
              SDCCL_NET,
              "NET/IBUC: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d",
              d, devices[d]->name, devices[d]->dev_name,
              sdcclIbDevs[sdcclNIbDevs].portNum,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              sdcclIbDevs[sdcclNIbDevs].speed, context,
              sdcclIbDevs[sdcclNIbDevs].pciPath,
              sdcclIbDevs[sdcclNIbDevs].ar);

          pthread_create(&sdcclIbAsyncThread, NULL, sdcclIbAsyncThreadMain,
                         sdcclIbDevs + sdcclNIbDevs);
          sdcclSetThreadName(sdcclIbAsyncThread, "SDCCL IbucAsync %2d",
                              sdcclNIbDevs);
          pthread_detach(sdcclIbAsyncThread); // will not be pthread_join()'d

          int mergedDev = sdcclNMergedIbDevs;
          if (sdcclParamIbMergeNics()) {
            mergedDev = sdcclIbFindMatchingDev(sdcclNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if
          // there's only one dev inside)
          if (mergedDev == sdcclNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IBUC device
            sdcclIbMergedDevs[mergedDev].ndevs = 1;
            sdcclIbMergedDevs[mergedDev].devs[0] = sdcclNIbDevs;
            sdcclNMergedIbDevs++;
            strncpy(sdcclIbMergedDevs[mergedDev].devName,
                    sdcclIbDevs[sdcclNIbDevs].devName, MAXNAMESIZE);
            // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IBUC device
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
      if (nIbucDevs &&
          (sdcclSuccess != sdcclWrapIbvFreeDeviceList(devices))) {
        ret = sdcclInternalError;
        goto fail;
      };
    }
    if (sdcclNIbDevs == 0) {
      INFO(SDCCL_INIT | SDCCL_NET, "NET/IBUC : No device found.");
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
            int ibucDev = mergedDev->devs[i];
            snprintf(line + strlen(line), 2047 - strlen(line),
                     "[%d] %s:%d/%s%s", ibucDev, sdcclIbDevs[ibucDev].devName,
                     sdcclIbDevs[ibucDev].portNum,
                     sdcclIbDevs[ibucDev].link == IBV_LINK_LAYER_INFINIBAND
                         ? "IB"
                         : "RoCE",
                     // Insert comma to delineate
                     i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line + strlen(line), 2047 - strlen(line), "}");
        } else {
          int ibucDev = mergedDev->devs[0];
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]%s:%d/%s",
                   ibucDev, sdcclIbDevs[ibucDev].devName,
                   sdcclIbDevs[ibucDev].portNum,
                   sdcclIbDevs[ibucDev].link == IBV_LINK_LAYER_INFINIBAND
                       ? "IB"
                       : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN + 1];
      INFO(SDCCL_NET, "NET/IBUC : Using%s %s; OOB %s:%s", line,
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

sdcclResult_t sdcclIbucMalloc(void **ptr, size_t size);
sdcclResult_t sdcclIbucCreateQpWithType(uint8_t ib_port,
                                          struct sdcclIbNetCommDevBase *base,
                                          int access_flags,
                                          enum ibv_qp_type qp_type,
                                          struct sdcclIbQp *qp);
sdcclResult_t sdcclIbucCreateQpWithTypeSrq(
    uint8_t ib_port, struct sdcclIbNetCommDevBase *base, int access_flags,
    enum ibv_qp_type qp_type, struct sdcclIbQp *qp, struct ibv_srq *srq);
sdcclResult_t sdcclIbucRtrQpWithType(struct ibv_qp *qp, uint8_t sGidIndex,
                                       uint32_t dest_qp_num,
                                       struct sdcclIbDevInfo *info,
                                       enum ibv_qp_type qp_type);
sdcclResult_t sdcclIbucRtsQpWithType(struct ibv_qp *qp,
                                       enum ibv_qp_type qp_type);

sdcclResult_t sdcclIbucMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  return (*ptr == NULL) ? sdcclInternalError : sdcclSuccess;
}

static void sdcclIbucAddEvent(struct sdcclIbRequest *req, int devIndex,
                               struct sdcclIbNetCommDevBase *base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

sdcclResult_t sdcclIbucInitCommDevBase(int ibDevN,
                                         struct sdcclIbNetCommDevBase *base) {
  base->ibDevN = ibDevN;
  sdcclIbDev *ibucDev = sdcclIbDevs + ibDevN;
  pthread_mutex_lock(&ibucDev->lock);
  if (0 == ibucDev->pdRefs++) {
    sdcclResult_t res;
    SDCCLCHECKGOTO(sdcclWrapIbvAllocPd(&ibucDev->pd, ibucDev->context), res,
                    failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibucDev->lock);
      return res;
    }
  }
  base->pd = ibucDev->pd;
  pthread_mutex_unlock(&ibucDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for
  // the Recv).
  SDCCLCHECK(sdcclWrapIbvCreateCq(
      &base->cq, ibucDev->context, 2 * MAX_REQUESTS * sdcclParamIbQpsPerConn(),
      NULL, NULL, 0));

  return sdcclSuccess;
}

sdcclResult_t sdcclIbucDestroyBase(struct sdcclIbNetCommDevBase *base) {
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
    sdcclResult_t pdResult =
        sdcclWrapIbvDeallocPd(sdcclIbDevs[base->ibDevN].pd);
    if (pdResult != sdcclSuccess) {
      if (sdcclDebugNoWarn == 0)
        INFO(SDCCL_ALL,
             "Failed to deallocate PD: %d (non-fatal, may have remaining "
             "resources)",
             pdResult);
      res = sdcclSuccess;
    } else {
      res = sdcclSuccess;
    }
  } else {
    res = sdcclSuccess;
  }
  pthread_mutex_unlock(&sdcclIbDevs[base->ibDevN].lock);
  return res;
}

sdcclResult_t sdcclIbucCreateQp(uint8_t ib_port,
                                  struct sdcclIbNetCommDevBase *base,
                                  int access_flags, struct sdcclIbQp *qp) {
  return sdcclIbucCreateQpWithType(ib_port, base, access_flags, IBV_QPT_UC,
                                    qp);
}

sdcclResult_t sdcclIbucCreateQpWithType(uint8_t ib_port,
                                          struct sdcclIbNetCommDevBase *base,
                                          int access_flags,
                                          enum ibv_qp_type qp_type,
                                          struct sdcclIbQp *qp) {
  return sdcclIbucCreateQpWithTypeSrq(ib_port, base, access_flags, qp_type, qp,
                                       NULL);
}

sdcclResult_t sdcclIbucCreateQpWithTypeSrq(
    uint8_t ib_port, struct sdcclIbNetCommDevBase *base, int access_flags,
    enum ibv_qp_type qp_type, struct sdcclIbQp *qp, struct ibv_srq *srq) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = qp_type;

  if (srq != NULL) {
    qpInitAttr.srq = srq;
    qpInitAttr.cap.max_recv_wr = 0;
    TRACE(SDCCL_NET, "Creating UC QP with SRQ: srq=%p", srq);
  } else {
    qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  }

  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2 * MAX_REQUESTS;
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
  qpAttr.qp_access_flags = access_flags;
  SDCCLCHECK(sdcclWrapIbvModifyQp(qp->qp, &qpAttr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucRtrQp(struct ibv_qp *qp, uint8_t sGidIndex,
                               uint32_t dest_qp_num,
                               struct sdcclIbDevInfo *info) {
  return sdcclIbucRtrQpWithType(qp, sGidIndex, dest_qp_num, info, IBV_QPT_UC);
}

sdcclResult_t sdcclIbucRtrQpWithType(struct ibv_qp *qp, uint8_t sGidIndex,
                                       uint32_t dest_qp_num,
                                       struct sdcclIbDevInfo *info,
                                       enum ibv_qp_type qp_type) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;

  // For RC mode, we need additional parameters
  if (qp_type == IBV_QPT_RC) {
    qpAttr.max_dest_rd_atomic = 1;
    qpAttr.min_rnr_timer = 12;
  }
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
  int modifyFlags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  if (qp_type == IBV_QPT_RC) {
    modifyFlags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  }
  SDCCLCHECK(sdcclWrapIbvModifyQp(qp, &qpAttr, modifyFlags));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucRtsQp(struct ibv_qp *qp) {
  return sdcclIbucRtsQpWithType(qp, IBV_QPT_UC);
}

sdcclResult_t sdcclIbucRtsQpWithType(struct ibv_qp *qp,
                                       enum ibv_qp_type qp_type) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.sq_psn = 0;

  // For RC mode, we need additional parameters
  if (qp_type == IBV_QPT_RC) {
    qpAttr.timeout = sdcclParamIbTimeout();
    qpAttr.retry_cnt = sdcclParamIbRetryCnt();
    qpAttr.rnr_retry = 7;
    qpAttr.max_rd_atomic = 1;
  }

  int modifyFlags = IBV_QP_STATE | IBV_QP_SQ_PSN;
  if (qp_type == IBV_QPT_RC) {
    modifyFlags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                   IBV_QP_MAX_QP_RD_ATOMIC;
  }
  SDCCLCHECK(sdcclWrapIbvModifyQp(qp, &qpAttr, modifyFlags));
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucListen(int dev, void *opaqueHandle,
                                void **listenComm) {
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

sdcclResult_t sdcclIbucConnect(int dev, void *opaqueHandle, void **sendComm) {
  struct sdcclIbHandle *handle = (struct sdcclIbHandle *)opaqueHandle;
  struct sdcclIbCommStage *stage = &handle->stage;
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)stage->comm;
  int ready;
  *sendComm = NULL;

  if (stage->state == sdcclIbCommStateConnect)
    goto ibuc_connect_check;
  if (stage->state == sdcclIbCommStateSend)
    goto ibuc_send;
  if (stage->state == sdcclIbCommStateConnecting)
    goto ibuc_connect;
  if (stage->state == sdcclIbCommStateConnected)
    goto ibuc_send_ready;
  if (stage->state != sdcclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return sdcclInternalError;
  }

  SDCCLCHECK(
      sdcclIbucMalloc((void **)&comm, sizeof(struct sdcclIbSendComm)));
  SDCCLCHECK(sdcclSocketInit(&comm->base.sock, &handle->connectAddr,
                               handle->magic, sdcclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = sdcclIbCommStateConnect;
  SDCCLCHECK(sdcclSocketConnect(&comm->base.sock));

ibuc_connect_check:
  /* since sdcclSocketConnect is async, we must check if connection is complete
   */
  SDCCLCHECK(sdcclSocketReady(&comm->base.sock, &ready));
  if (!ready)
    return sdcclSuccess;

  // IBUC Setup
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
    SDCCLCHECK(sdcclIbucInitCommDevBase(ibDevN, &comm->devs[i].base));
    comm->ar = comm->ar &&
               sdcclIbDevs[dev]
                   .ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  struct sdcclIbConnectionMetadata meta;
  meta.ndevs = comm->base.ndevs;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    sdcclIbSendCommDev *commDev = comm->devs + devIndex;
    sdcclIbDev *ibucDev = sdcclIbDevs + commDev->base.ibDevN;
    SDCCLCHECK(sdcclIbucCreateQp(ibucDev->portNum, &commDev->base,
                                   IBV_ACCESS_REMOTE_WRITE,
                                   comm->base.qps + q));
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // Query ece capabilities (enhanced connection establishment)
    SDCCLCHECK(sdcclWrapIbvQueryEce(comm->base.qps[q].qp, &meta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));
    devIndex = (devIndex + 1) % comm->base.ndevs;
  }

  // IBUC always enables retransmission, ignore environment variable
  meta.retransEnabled = 1;

  for (int i = 0; i < comm->base.ndevs; i++) {
    sdcclIbSendCommDev *commDev = comm->devs + i;
    sdcclIbDev *ibucDev = sdcclIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    sdcclIbDevInfo *devInfo = meta.devs + i;
    devInfo->ibPort = ibucDev->portNum;
    devInfo->mtu = ibucDev->portAttr.active_mtu;
    devInfo->lid = ibucDev->portAttr.lid;

    // Prepare my fifo
    SDCCLCHECK(
        sdcclWrapIbvRegMr(&commDev->fifoMr, commDev->base.pd, comm->fifo,
                           sizeof(struct sdcclIbSendFifo) * MAX_REQUESTS *
                               SDCCL_NET_IB_MAX_RECVS,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ));
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // RoCE support
    devInfo->linkLayer = commDev->base.gidInfo.linkLayer =
        ibucDev->portAttr.link_layer;
    if (devInfo->linkLayer == IBV_LINK_LAYER_ETHERNET) {
      SDCCLCHECK(sdcclIbGetGidIndex(ibucDev->context, ibucDev->portNum,
                                      ibucDev->portAttr.gid_tbl_len,
                                      &commDev->base.gidInfo.localGidIndex));
      SDCCLCHECK(sdcclWrapIbvQueryGid(ibucDev->context, ibucDev->portNum,
                                        commDev->base.gidInfo.localGidIndex,
                                        &commDev->base.gidInfo.localGid));
      devInfo->spn = commDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo->iid = commDev->base.gidInfo.localGid.global.interface_id;
    } else {
      commDev->base.gidInfo.localGidIndex = 0;
      memset(&commDev->base.gidInfo.localGid, 0, sizeof(union ibv_gid));
    }

    if (meta.retransEnabled) {
      SDCCLCHECK(sdcclIbCreateCtrlQp(ibucDev->context, commDev->base.pd,
                                       ibucDev->portNum, &commDev->ctrlQp));
      meta.ctrlQpn[i] = commDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibucDev->portAttr.lid;
      meta.ctrlGid[i] = commDev->base.gidInfo.localGid;

      size_t ack_buf_size =
          (sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING) *
          SDCCL_IB_ACK_BUF_COUNT;
      commDev->ackBuffer = malloc(ack_buf_size);
      SDCCLCHECK(sdcclWrapIbvRegMr(&commDev->ackMr, commDev->base.pd,
                                     commDev->ackBuffer, ack_buf_size,
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
               "NET/IBUC: %s %d IbucDev %d Port %d qpn %d mtu %d LID %d "
               "fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "SDCCL MergedDev" : "SDCCL Dev", dev,
               commDev->base.ibDevN, ibucDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, devInfo->lid, devInfo->fifoRkey,
               commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(SDCCL_NET,
               "NET/IBUC: %s %d IbucDev %d Port %d qpn %d mtu %d "
               "query_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
               "comp_mask=0x%x} GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "SDCCL MergedDev" : "SDCCL Dev", dev,
               commDev->base.ibDevN, ibucDev->portNum, meta.qpInfo[q].qpn,
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
  SDCCLCHECK(sdcclIbucMalloc((void **)&stage->buffer, sizeof(meta)));

  memcpy(stage->buffer, &meta, sizeof(meta));

ibuc_send:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_SEND, &comm->base.sock,
                                   stage->buffer, sizeof(meta),
                                   &stage->offset));
  if (stage->offset != sizeof(meta))
    return sdcclSuccess;

  stage->state = sdcclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ibuc_connect:
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
    WARN(
        "NET/IBUC : Local mergedDev=%s has a different number of devices=%d as "
        "remoteDev=%s nRemDevs=%d",
        mergedDev->devName, comm->base.ndevs, remMeta.devName,
        comm->base.nRemDevs);
  }

  int linkLayer;
  linkLayer = remMeta.devs[0].linkLayer;
  for (int i = 1; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].linkLayer != linkLayer) {
      WARN("NET/IBUC : Can't merge net devices with different linkLayer. i=%d "
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

    SDCCLCHECK(sdcclIbucRtrQp(qp, gidIndex, remQpInfo->qpn, remDevInfo));
    SDCCLCHECK(sdcclIbucRtsQp(qp));
  }

  if (linkLayer == IBV_LINK_LAYER_ETHERNET) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct sdcclIbQp *qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct sdcclIbDev *ibucDev = sdcclIbDevs + ibDevN;
      INFO(SDCCL_NET,
           "NET/IBUC: IbucDev %d Port %d qpn %d set_ece={supported=%d, "
           "vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
           ibDevN, ibucDev->portNum, remMeta.qpInfo[q].qpn,
           remMeta.qpInfo[q].eceSupported, remMeta.qpInfo[q].ece.vendor_id,
           remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  SDCCLCHECK(sdcclIbRetransInit(&comm->retrans));
  if (comm->retrans.enabled) {
    // IBUC typically has very few in-flight packets; force immediate ACKs
    comm->retrans.ackInterval = 1;
  }
  comm->lastTimeoutCheckUs = 0;

  // IBUC always enables retransmission, force it on
  comm->retrans.enabled = 1;
  // Force remMeta.retransEnabled = 1 for IBUC to ensure control QP is created
  remMeta.retransEnabled = 1;
  if (!remMeta.retransEnabled) {
    INFO(SDCCL_NET,
         "Receiver disabled retransmission, but IBUC always enables it");
  }

  if (comm->retrans.enabled) {
    INFO(SDCCL_NET,
         "NET/IBUC Sender: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         comm->retrans.minRtoUs, comm->retrans.maxRetry,
         comm->retrans.ackInterval);
  } else {
    INFO(SDCCL_NET, "NET/IBUC Sender: Retransmission DISABLED");
  }

  if (comm->retrans.enabled && remMeta.retransEnabled) {
    bool all_ah_success = true;

    for (int i = 0; i < comm->base.ndevs; i++) {
      sdcclIbSendCommDev *commDev = &comm->devs[i];
      sdcclIbDev *ibucDev = sdcclIbDevs + commDev->base.ibDevN;

      TRACE(SDCCL_NET,
            "Send: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibucDev->portAttr.link_layer);

      sdcclResult_t ah_result = sdcclIbSetupCtrlQpConnection(
          ibucDev->context, commDev->base.pd, &commDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibucDev->portNum, ibucDev->portAttr.link_layer,
          commDev->base.gidInfo.localGidIndex);

      if (ah_result != sdcclSuccess || !commDev->ctrlQp.ah) {
        all_ah_success = false;
        break;
      }

      size_t buf_entry_size =
          sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING;
      for (int r = 0; r < 32; r++) {
        struct ibv_sge sge;
        sge.addr = (uint64_t)((char *)commDev->ackBuffer + r * buf_entry_size);
        sge.length = buf_entry_size;
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

    if (!all_ah_success) {
      comm->retrans.enabled = 0;

      for (int i = 0; i < comm->base.ndevs; i++) {
        if (comm->devs[i].ackMr)
          sdcclWrapIbvDeregMr(comm->devs[i].ackMr);
        if (comm->devs[i].ackBuffer)
          free(comm->devs[i].ackBuffer);
        sdcclIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
      }
    }

    if (all_ah_success && comm->retrans.enabled) {
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

  comm->outstandingSends = 0;
  comm->outstandingRetrans = 0;
  comm->maxOutstanding = sdcclParamIbMaxOutstanding();

  comm->base.ready = 1;
  stage->state = sdcclIbCommStateConnected;
  stage->offset = 0;

ibuc_send_ready:
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

SDCCL_PARAM(IbucGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

sdcclResult_t sdcclIbucAccept(void *listenComm, void **recvComm) {
  struct sdcclIbListenComm *lComm = (struct sdcclIbListenComm *)listenComm;
  struct sdcclIbCommStage *stage = &lComm->stage;
  struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)stage->comm;
  int ready;
  *recvComm = NULL;

  // Pre-declare ALL variables before any goto to avoid crossing initialization
  struct sdcclIbMergedDev *mergedDev;
  struct sdcclIbDev *ibucDev;
  int ibDevN;
  struct sdcclIbRecvCommDev *rCommDev;
  struct sdcclIbDevInfo *remDevInfo;
  struct sdcclIbQp *qp;
  struct ibv_srq *srq = NULL;
  struct sdcclIbConnectionMetadata remMeta;
  struct sdcclIbConnectionMetadata meta;
  memset(&meta, 0,
         sizeof(meta)); // Initialize meta, including meta.retransEnabled = 0

  if (stage->state == sdcclIbCommStateAccept)
    goto ib_accept_check;
  if (stage->state == sdcclIbCommStateRecv)
    goto ib_recv;
  if (stage->state == sdcclIbCommStateSend)
    goto ibuc_send;
  if (stage->state == sdcclIbCommStatePendingReady)
    goto ib_recv_ready;
  if (stage->state != sdcclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return sdcclInternalError;
  }

  SDCCLCHECK(
      sdcclIbucMalloc((void **)&rComm, sizeof(struct sdcclIbRecvComm)));
  stage->comm = rComm;
  stage->state = sdcclIbCommStateAccept;
  SDCCLCHECK(sdcclSocketInit(&rComm->base.sock));
  SDCCLCHECK(sdcclSocketAccept(&rComm->base.sock, &lComm->sock));

ib_accept_check:
  SDCCLCHECK(sdcclSocketReady(&rComm->base.sock, &ready));
  if (!ready)
    return sdcclSuccess;

  // remMeta already declared at function start
  stage->state = sdcclIbCommStateRecv;
  stage->offset = 0;
  SDCCLCHECK(sdcclIbucMalloc((void **)&stage->buffer, sizeof(remMeta)));

ib_recv:
  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, &rComm->base.sock,
                                   stage->buffer, sizeof(remMeta),
                                   &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return sdcclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct sdcclIbConnectionMetadata));

  mergedDev = sdcclIbMergedDevs + lComm->dev;
  rComm->base.ndevs = mergedDev->ndevs;
  rComm->base.nqps = sdcclParamIbQpsPerConn() *
                     rComm->base.ndevs; // We must have at least 1 qp per-device
  rComm->base.isSend = false;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.ndevs) {
    WARN(
        "NET/IBUC : Local mergedDev %s has a different number of devices=%d as "
        "remote %s %d",
        mergedDev->devName, rComm->base.ndevs, remMeta.devName,
        rComm->base.nRemDevs);
  }

  for (int i = 0; i < rComm->base.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = mergedDev->devs[i];
    SDCCLCHECK(sdcclIbucInitCommDevBase(ibDevN, &rCommDev->base));
    ibucDev = sdcclIbDevs + ibDevN;
    SDCCLCHECK(sdcclIbGetGidIndex(ibucDev->context, ibucDev->portNum,
                                    ibucDev->portAttr.gid_tbl_len,
                                    &rCommDev->base.gidInfo.localGidIndex));
    SDCCLCHECK(sdcclWrapIbvQueryGid(ibucDev->context, ibucDev->portNum,
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

  // Create SRQ if retransmission is enabled
  remMeta.retransEnabled = 1;
  meta.retransEnabled = 1;
  srq = NULL;
  if (remMeta.retransEnabled) {
    ibDevN = mergedDev->devs[0];
    ibucDev = sdcclIbDevs + ibDevN;

    sdcclResult_t srq_result = sdcclIbCreateSrq(
        ibucDev->context, rComm->devs[0].base.pd, &rComm->srqMgr);
    if (srq_result == sdcclSuccess) {
      srq = (struct ibv_srq *)rComm->srqMgr.srq;
      TRACE(SDCCL_NET, "Receiver: Created SRQ for retransmission: srq=%p",
            srq);
    } else {
      INFO(SDCCL_NET,
           "Receiver: Failed to create SRQ (result=%d), disabling "
           "retransmission",
           srq_result);
      remMeta.retransEnabled = 0;
      meta.retransEnabled = 0;
      rComm->retrans.enabled = 0;
      srq = NULL;
    }
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
    ibucDev = sdcclIbDevs + ibDevN;

    SDCCLCHECK(sdcclIbucCreateQpWithTypeSrq(ibucDev->portNum, &rCommDev->base,
                                              IBV_ACCESS_REMOTE_WRITE,
                                              IBV_QPT_UC, qp, srq));
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].eceSupported) {
      SDCCLCHECK(sdcclWrapIbvSetEce(qp->qp, &remMeta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));

      if (meta.qpInfo[q].eceSupported)
        SDCCLCHECK(sdcclWrapIbvQueryEce(qp->qp, &meta.qpInfo[q].ece,
                                          &meta.qpInfo[q].eceSupported));
    }

    SDCCLCHECK(sdcclIbucRtrQp(qp->qp, rCommDev->base.gidInfo.localGidIndex,
                                remMeta.qpInfo[q].qpn, remDevInfo));
    SDCCLCHECK(sdcclIbucRtsQp(qp->qp));
  }

  rComm->flushEnabled = 1;

  for (int i = 0; i < mergedDev->ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rCommDev->base.ibDevN;
    ibucDev = sdcclIbDevs + ibDevN;

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
      SDCCLCHECK(sdcclIbucCreateQpWithType(
          ibucDev->portNum, &rCommDev->base,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, IBV_QPT_RC,
          &rCommDev->gpuFlush.qp));
      struct sdcclIbDevInfo devInfo;
      devInfo.lid = ibucDev->portAttr.lid;
      devInfo.linkLayer = ibucDev->portAttr.link_layer;
      devInfo.ibPort = ibucDev->portNum;
      devInfo.spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.iid = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu = ibucDev->portAttr.active_mtu;
      SDCCLCHECK(sdcclIbucRtrQpWithType(
          rCommDev->gpuFlush.qp.qp, rCommDev->base.gidInfo.localGidIndex,
          rCommDev->gpuFlush.qp.qp->qp_num, &devInfo, IBV_QPT_RC));
      SDCCLCHECK(
          sdcclIbucRtsQpWithType(rCommDev->gpuFlush.qp.qp, IBV_QPT_RC));
    }

    if (remMeta.retransEnabled && meta.retransEnabled) {
      SDCCLCHECK(sdcclIbCreateCtrlQp(ibucDev->context, rCommDev->base.pd,
                                       ibucDev->portNum, &rCommDev->ctrlQp));
      meta.ctrlQpn[i] = rCommDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibucDev->portAttr.lid;
      meta.ctrlGid[i] = rCommDev->base.gidInfo.localGid;

      TRACE(SDCCL_NET,
            "Receiver: Control QP created for dev %d, qpn=%u, lid=%u", i,
            meta.ctrlQpn[i], meta.ctrlLid[i]);

      size_t ack_buf_size =
          (sizeof(struct sdcclIbAckMsg) + SDCCL_IB_ACK_BUF_PADDING) *
          SDCCL_IB_ACK_BUF_COUNT;
      rCommDev->ackBuffer = malloc(ack_buf_size);
      SDCCLCHECK(sdcclWrapIbvRegMr(&rCommDev->ackMr, rCommDev->base.pd,
                                     rCommDev->ackBuffer, ack_buf_size,
                                     IBV_ACCESS_LOCAL_WRITE));

      TRACE(SDCCL_NET,
            "Recv: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibucDev->portAttr.link_layer);

      sdcclResult_t ah_result = sdcclIbSetupCtrlQpConnection(
          ibucDev->context, rCommDev->base.pd, &rCommDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibucDev->portNum, ibucDev->portAttr.link_layer,
          rCommDev->base.gidInfo.localGidIndex);

      if (ah_result != sdcclSuccess || !rCommDev->ctrlQp.ah) {
        INFO(SDCCL_NET,
             "Receiver Control QP setup failed for dev %d, disabling "
             "retransmission",
             i);
        rComm->retrans.enabled = 0;
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

    // Fill Handle
    meta.devs[i].lid = ibucDev->portAttr.lid;
    meta.devs[i].linkLayer = rCommDev->base.gidInfo.linkLayer =
        ibucDev->portAttr.link_layer;
    meta.devs[i].ibPort = ibucDev->portNum;
    meta.devs[i].spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].iid = rCommDev->base.gidInfo.localGid.global.interface_id;

    // Adjust the MTU
    remMeta.devs[i].mtu = (enum ibv_mtu)std::min(remMeta.devs[i].mtu,
                                                 ibucDev->portAttr.active_mtu);
    meta.devs[i].mtu = remMeta.devs[i].mtu;

    // Prepare sizes fifo
    SDCCLCHECK(sdcclWrapIbvRegMr(
        &rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo,
        sizeof(int) * MAX_REQUESTS * SDCCL_NET_IB_MAX_RECVS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.ndevs;
  // IBUC always enables retransmission, ignore remote value
  meta.retransEnabled = 1;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = sdcclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer)
    free(stage->buffer);
  SDCCLCHECK(sdcclIbucMalloc((void **)&stage->buffer,
                               sizeof(struct sdcclIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct sdcclIbConnectionMetadata));

ibuc_send:
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

  SDCCLCHECK(sdcclIbRetransInit(&rComm->retrans));
  if (rComm->retrans.enabled) {
    rComm->retrans.ackInterval = 1;
  }

  // IBUC always enables retransmission, force it on
  rComm->retrans.enabled = 1;
  if (meta.retransEnabled == 0) {
    INFO(
        SDCCL_NET,
        "Receiver: Remote disabled retransmission, but IBUC always enables it");
  }

  // Initialize SRQ with recv buffers
  if (rComm->retrans.enabled && rComm->srqMgr.srq != NULL) {
    rComm->srqMgr.postSrqCount = SDCCL_IB_SRQ_SIZE;

    // Post in batches until all are posted
    while (rComm->srqMgr.postSrqCount > 0) {
      SDCCLCHECK(sdcclIbSrqPostRecv(&rComm->srqMgr, SDCCL_IB_ACK_BUF_COUNT));
    }

    INFO(SDCCL_NET,
         "NET/IBUC Receiver: Retransmission ENABLED (Posted %d recv WRs to SRQ "
         "for retransmission)",
         SDCCL_IB_SRQ_SIZE);
  } else {
    INFO(SDCCL_NET, "NET/IBUC Receiver: Retransmission DISABLED");
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

sdcclResult_t sdcclIbucGetRequest(struct sdcclIbNetCommBase *base,
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
  WARN("NET/IBUC : unable to allocate requests");
  *req = NULL;
  return sdcclInternalError;
}

sdcclResult_t sdcclIbucRegMrDmaBufInternal(sdcclIbNetCommDevBase *base,
                                             void *data, size_t size, int type,
                                             uint64_t offset, int fd,
                                             int mrFlags, ibv_mr **mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0)
    pageSize = sysconf(_SC_PAGESIZE);
  struct sdcclIbMrCache *cache = &sdcclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  sdcclResult_t res;
  pthread_mutex_lock(&sdcclIbDevs[base->ibDevN].lock);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        SDCCLCHECKGOTO(
            sdcclRealloc(&cache->slots, cache->population, cache->capacity),
            res, returning);
      }
      // Deregister / register
      struct ibv_mr *mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
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
sdcclIbucGetNetCommDevBase(sdcclIbNetCommBase *base, int devIndex) {
  if (base->isSend) {
    struct sdcclIbSendComm *sComm = (struct sdcclIbSendComm *)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct sdcclIbRecvComm *rComm = (struct sdcclIbRecvComm *)base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
sdcclResult_t sdcclIbucRegMrDmaBuf(void *comm, void *data, size_t size,
                                     int type, uint64_t offset, int fd,
                                     int mrFlags, void **mhandle) {
  assert(size > 0);
  struct sdcclIbNetCommBase *base = (struct sdcclIbNetCommBase *)comm;
  struct sdcclIbMrHandle *mhandleWrapper =
      (struct sdcclIbMrHandle *)malloc(sizeof(struct sdcclIbMrHandle));
  for (int i = 0; i < base->ndevs; i++) {
    struct sdcclIbNetCommDevBase *devComm =
        sdcclIbucGetNetCommDevBase(base, i);
    SDCCLCHECK(sdcclIbucRegMrDmaBufInternal(devComm, data, size, type, offset,
                                              fd, mrFlags,
                                              mhandleWrapper->mrs + i));
  }
  *mhandle = (void *)mhandleWrapper;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucRegMr(void *comm, void *data, size_t size, int type,
                               int mrFlags, void **mhandle) {
  return sdcclIbucRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mrFlags,
                               mhandle);
}

sdcclResult_t sdcclIbucDeregMrInternal(sdcclIbNetCommDevBase *base,
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
  WARN("NET/IBUC: could not find mr %p inside cache of %d entries", mhandle,
       cache->population);
  res = sdcclInternalError;
returning:
  pthread_mutex_unlock(&sdcclIbDevs[base->ibDevN].lock);
  return res;
}

sdcclResult_t sdcclIbucDeregMr(void *comm, void *mhandle) {
  struct sdcclIbMrHandle *mhandleWrapper = (struct sdcclIbMrHandle *)mhandle;
  struct sdcclIbNetCommBase *base = (struct sdcclIbNetCommBase *)comm;
  for (int i = 0; i < base->ndevs; i++) {
    struct sdcclIbNetCommDevBase *devComm =
        sdcclIbucGetNetCommDevBase(base, i);
    SDCCLCHECK(sdcclIbucDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return sdcclSuccess;
}

SDCCL_PARAM(IbucSplitDataOnQps, "IBUC_SPLIT_DATA_ON_QPS", 0);

sdcclResult_t sdcclIbucMultiSend(struct sdcclIbSendComm *comm, int slot) {
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
  int nqps =
      sdcclParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    sdcclIbQp *qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    for (int r = 0; r < nreqs; r++) {
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

    // Ensure lastWr has IBV_SEND_SIGNALED set for each QP
    // (it was set before the loop, but we need to ensure it's set for each QP)
    lastWr->send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    // Call ibv_post_send directly to handle ENOMEM (send queue full) gracefully
    int ret = qp->qp->context->ops.post_send(qp->qp, comm->wrs, &bad_wr);
    if (ret != IBV_SUCCESS) {
      // If send queue is full (ENOMEM), poll completions from all devices and
      // retry
      if (ret == ENOMEM) {
        struct ibv_wc wcs[64];
        // Poll all devices' CQs to free up send queue space
        for (int dev_i = 0; dev_i < comm->base.ndevs; dev_i++) {
          struct sdcclIbNetCommDevBase *devBase = &comm->devs[dev_i].base;
          if (!devBase || !devBase->cq)
            continue;
          for (int poll_round = 0; poll_round < 16; poll_round++) {
            int n_cqe = 0;
            sdcclWrapIbvPollCq(devBase->cq, 64, wcs, &n_cqe);
            if (n_cqe == 0)
              break;
          }
        }
        // Retry sending after polling
        ret = qp->qp->context->ops.post_send(qp->qp, comm->wrs, &bad_wr);
        // If still failing after polling, continue retrying with more
        // aggressive polling
        int retry_count = 0;
        while (ret == ENOMEM && retry_count < 3) {
          // More aggressive polling
          for (int dev_i = 0; dev_i < comm->base.ndevs; dev_i++) {
            struct sdcclIbNetCommDevBase *devBase = &comm->devs[dev_i].base;
            if (!devBase || !devBase->cq)
              continue;
            for (int poll_round = 0; poll_round < 32; poll_round++) {
              int n_cqe = 0;
              sdcclWrapIbvPollCq(devBase->cq, 64, wcs, &n_cqe);
              if (n_cqe == 0)
                break;
            }
          }
          sched_yield(); // Yield CPU to allow other threads/processes to make
                         // progress
          ret = qp->qp->context->ops.post_send(qp->qp, comm->wrs, &bad_wr);
          retry_count++;
        }
      }
      // If still failing, check if it's ENOMEM (don't warn) or other error
      if (ret != IBV_SUCCESS) {
        if (ret != ENOMEM) {
          WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p",
               strerror(ret), comm->wrs, bad_wr);
        }
        SDCCLCHECK(sdcclSystemError);
      }
    }

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }

  if (comm->retrans.enabled && nreqs == 1) {
    sdcclResult_t add_result = sdcclIbRetransAddPacket(
        &comm->retrans, seq, reqs[0]->send.size, reqs[0]->send.data,
        slots[0].addr, // remote_addr
        reqs[0]->send.lkeys, (uint32_t *)slots[0].rkeys);
    SDCCLCHECK(add_result);
  }

  comm->outstandingSends++;

  return sdcclSuccess;
}

sdcclResult_t sdcclIbucIsend(void *sendComm, void *data, size_t size, int tag,
                               void *mhandle, void *phandle, void **request) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IBUC: sdcclIbucIsend() called when comm->base.ready == 0");
    return sdcclInternalError;
  }
  // Removed sdcclIbucPollCompletions call to match ibrc behavior
  // Completions are handled in Test function, not in Isend
  // This prevents potential hang issues

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
  for (int r = 1; r < nreqs; r++) {
    int spin_count = 0;
    while (slots[r].idx != idx) {
      if (++spin_count > 1000) {
        sched_yield(); // Yield CPU to prevent busy-wait hang
        spin_count = 0;
      }
    }
  }
  __sync_synchronize();
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
      WARN("NET/IBUC : req %d/%d tag %x peer %s posted incorrect receive info: "
           "size %ld addr %lx rkeys[0]=%x",
           r, nreqs, tag, sdcclSocketToString(&addr, line), slots[r].size,
           slots[r].addr, slots[r].rkeys[0]);
      return sdcclInternalError;
    }

    struct sdcclIbRequest *req;
    SDCCLCHECK(sdcclIbucGetRequest(&comm->base, &req));
    req->type = SDCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents =
        sdcclParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      sdcclIbQp *qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      sdcclIbucAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same
      // set of QPs inside sdcclIbucMultiSend()
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
    SDCCLCHECK(sdcclIbucMultiSend(comm, slot));

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

sdcclResult_t sdcclIbucPostFifo(struct sdcclIbRecvComm *comm, int n,
                                  void **data, size_t *sizes, int *tags,
                                  void **mhandles,
                                  struct sdcclIbRequest *req) {
  return sdcclIbCommonPostFifo(comm, n, data, sizes, tags, mhandles, req,
                                sdcclIbucAddEvent);
}

sdcclResult_t sdcclIbucIrecv(void *recvComm, int n, void **data,
                               size_t *sizes, int *tags, void **mhandles,
                               void **phandles, void **request) {
  struct sdcclIbRecvComm *comm = (struct sdcclIbRecvComm *)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IBUC: sdcclIbucIrecv() called when comm->base.ready == 0");
    return sdcclInternalError;
  }
  if (n > SDCCL_NET_IB_MAX_RECVS)
    return sdcclInternalError;

  struct sdcclIbRequest *req;
  SDCCLCHECK(sdcclIbucGetRequest(&comm->base, &req));
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
      sdcclParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Post recvs - skip if using SRQ
  // When using SRQ, all QPs share SRQ for recv, don't post per-request recv
  if (!comm->retrans.enabled || comm->srqMgr.srq == NULL) {
    // Normal mode: post recv to each QP
    struct ibv_recv_wr *bad_wr;
    for (int i = 0; i < nqps; i++) {
      struct sdcclIbQp *qp = comm->base.qps + comm->base.qpIndex;
      sdcclIbucAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
      SDCCLCHECK(sdcclWrapIbvPostRecv(qp->qp, &wr, &bad_wr));
      comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
    }
  } else {
    // SRQ mode: don't post recv, but still set up events for completion
    for (int i = 0; i < nqps; i++) {
      struct sdcclIbQp *qp = comm->base.qps + comm->base.qpIndex;
      sdcclIbucAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
      comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
    }
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  SDCCLCHECK(sdcclIbucPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucIflush(void *recvComm, int n, void **data, int *sizes,
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
  SDCCLCHECK(sdcclIbucGetRequest(&comm->base, &req));
  req->type = SDCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  // struct sdcclIbMrHandle *mhandle = (struct sdcclIbMrHandle
  // *)mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  // For flush operations, we use RC QP which supports RDMA_READ
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    // Use RDMA_READ for flush operations
    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = ((struct sdcclIbMrHandle *)mhandles[last])->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr *bad_wr;
    SDCCLCHECK(
        sdcclWrapIbvPostSend(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    sdcclIbucAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucTest(void *request, int *done, int *sizes) {
  static const struct sdcclIbCommonTestOps kIbucTestOps = {
      .component = "NET/IBUC",
      .pre_check = sdcclIbucTestPreCheck,
      .process_wc = sdcclIbucProcessWc,
  };
  struct sdcclIbRequest *r = (struct sdcclIbRequest *)request;
  sdcclResult_t result =
      sdcclIbCommonTestDataQp(r, done, sizes, &kIbucTestOps);
  return result;
}

sdcclResult_t sdcclIbucCloseSend(void *sendComm) {
  struct sdcclIbSendComm *comm = (struct sdcclIbSendComm *)sendComm;
  if (comm) {
    if (comm->retrans.enabled) {
      SDCCLCHECK(sdcclIbRetransDestroy(&comm->retrans));
    }

    SDCCLCHECK(sdcclSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int n_cqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          sdcclWrapIbvPollCq(commDev->base.cq, 64, wcs, &n_cqe);
          if (n_cqe == 0)
            break;
        }
      }
    }

    // Clean up retransmission resources (control QP) BEFORE destroying data QPs
    // This ensures control QP CQ completions are drained before PD is released
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (comm->retrans.enabled) {
        SDCCLCHECK(sdcclIbDestroyCtrlQp(&commDev->ctrlQp));
        if (commDev->ackMr != NULL)
          SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->ackMr));
        if (commDev->ackBuffer != NULL)
          free(commDev->ackBuffer);
        if (i == 0 && comm->retransHdrMr != NULL) {
          SDCCLCHECK(sdcclWrapIbvDeregMr(comm->retransHdrMr));
        }
      }
    }

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        SDCCLCHECK(sdcclWrapIbvDestroyQp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbSendCommDev *commDev = comm->devs + i;
      if (commDev->fifoMr != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL)
        SDCCLCHECK(sdcclWrapIbvDeregMr(comm->remSizesFifo.mrs[i]));
      SDCCLCHECK(sdcclIbucDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IBUC");
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucCloseRecv(void *recvComm) {
  struct sdcclIbRecvComm *comm = (struct sdcclIbRecvComm *)recvComm;
  if (comm) {
    if (comm->retrans.enabled) {
      SDCCLCHECK(sdcclIbRetransDestroy(&comm->retrans));
    }

    SDCCLCHECK(sdcclSocketClose(&comm->base.sock));

    // First, poll all CQs to drain completions before destroying QPs
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbRecvCommDev *commDev = comm->devs + i;
      if (commDev->base.cq) {
        struct ibv_wc wcs[64];
        int n_cqe = 0;
        // Poll multiple times to drain all pending completions
        for (int j = 0; j < 16; j++) {
          sdcclWrapIbvPollCq(commDev->base.cq, 64, wcs, &n_cqe);
          if (n_cqe == 0)
            break;
        }
      }
    }

    // Clean up retransmission resources (control QP) BEFORE destroying data QPs
    // This ensures control QP CQ completions are drained before PD is released
    for (int i = 0; i < comm->base.ndevs; i++) {
      struct sdcclIbRecvCommDev *commDev = comm->devs + i;
      if (comm->retrans.enabled) {
        SDCCLCHECK(sdcclIbDestroyCtrlQp(&commDev->ctrlQp));
        if (commDev->ackMr != NULL)
          SDCCLCHECK(sdcclWrapIbvDeregMr(commDev->ackMr));
        if (commDev->ackBuffer != NULL)
          free(commDev->ackBuffer);
      }
    }

    // Destroy QPs first, before destroying SRQ
    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL)
        SDCCLCHECK(sdcclWrapIbvDestroyQp(comm->base.qps[q].qp));

    if (comm->srqMgr.srq != NULL) {
      SDCCLCHECK(sdcclIbDestroySrq(&comm->srqMgr));
      TRACE(SDCCL_NET, "Receiver: Destroyed SRQ");
    }

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
      SDCCLCHECK(sdcclIbucDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucCloseListen(void *listenComm) {
  struct sdcclIbListenComm *comm = (struct sdcclIbListenComm *)listenComm;
  if (comm) {
    SDCCLCHECK(sdcclSocketClose(&comm->sock));
    free(comm);
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclIbucGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < sdcclNMergedIbDevs; i++) {
    if (strcmp(sdcclIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return sdcclSuccess;
    }
  }
  return sdcclSystemError;
}

sdcclResult_t sdcclIbucGetProperties(int dev, void *props) {
  struct sdcclIbMergedDev *mergedDev = sdcclIbMergedDevs + dev;
  sdcclNetProperties_t *properties = (sdcclNetProperties_t *)props;

  properties->name = mergedDev->devName;
  properties->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device
  struct sdcclIbDev *ibucDev = sdcclIbDevs + mergedDev->devs[0];
  properties->pciPath = ibucDev->pciPath;
  properties->guid = ibucDev->guid;
  properties->ptrSupport = SDCCL_PTR_HOST;

  if (sdcclIbGdrSupport() == sdcclSuccess) {
    properties->ptrSupport |= SDCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  properties->regIsGlobal = 1;
  if (sdcclIbDmaBufSupport(dev) == sdcclSuccess) {
    properties->ptrSupport |= SDCCL_PTR_DMABUF;
  }
  properties->latency = 0; // Not set
  properties->port = ibucDev->portNum + ibucDev->realPort;
  properties->maxComms = ibucDev->maxQp;
  properties->maxRecvs = SDCCL_NET_IB_MAX_RECVS;
  properties->netDeviceType = SDCCL_NET_DEVICE_HOST;
  properties->netDeviceVersion = SDCCL_NET_DEVICE_INVALID_VERSION;
  return sdcclSuccess;
}

// One-sided stubs (not supported by IBUC adaptor)
// Adapter wrapper functions

struct sdcclNetAdaptor sdcclNetIbuc = {
    // Basic functions
    "IBUC", sdcclIbucInit, sdcclIbDevices, sdcclIbucGetProperties,

    // Setup functions
    sdcclIbucListen, sdcclIbucConnect, sdcclIbucAccept, sdcclIbucCloseSend,
    sdcclIbucCloseRecv, sdcclIbucCloseListen,

    // Memory region functions
    sdcclIbucRegMr, sdcclIbucRegMrDmaBuf, sdcclIbucDeregMr,

    // Two-sided functions
    sdcclIbucIsend, sdcclIbucIrecv, sdcclIbucIflush, sdcclIbucTest,

    // One-sided functions
    NULL, // iput - not supported on IBUC
    NULL, // iget - not supported on IBUC
    NULL, // iputSignal - not supported on IBUC

    // Device name lookup
    sdcclIbucGetDevFromName};

#endif // USE_IBUC
