/*************************************************************************
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 *
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_IBVWRAP_H_
#define SDCCL_IBVWRAP_H_

#ifdef SDCCL_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#include "core.h"
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

typedef enum ibv_return_enum {
  IBV_SUCCESS = 0, //!< The operation was successful
} ibv_return_t;

sdcclResult_t sdcclWrapIbvSymbols(void);
/* SDCCL wrappers of IB verbs functions */
sdcclResult_t sdcclWrapIbvForkInit(void);
sdcclResult_t sdcclWrapIbvGetDeviceList(struct ibv_device ***ret,
                                          int *num_devices);
sdcclResult_t sdcclWrapIbvFreeDeviceList(struct ibv_device **list);
const char *sdcclWrapIbvGetDeviceName(struct ibv_device *device);
sdcclResult_t sdcclWrapIbvOpenDevice(struct ibv_context **ret,
                                       struct ibv_device *device);
sdcclResult_t sdcclWrapIbvCloseDevice(struct ibv_context *context);
sdcclResult_t sdcclWrapIbvGetAsyncEvent(struct ibv_context *context,
                                          struct ibv_async_event *event);
sdcclResult_t sdcclWrapIbvAckAsyncEvent(struct ibv_async_event *event);
sdcclResult_t sdcclWrapIbvQueryDevice(struct ibv_context *context,
                                        struct ibv_device_attr *device_attr);
sdcclResult_t sdcclWrapIbvQueryPort(struct ibv_context *context,
                                      uint8_t port_num,
                                      struct ibv_port_attr *port_attr);
sdcclResult_t sdcclWrapIbvQueryGid(struct ibv_context *context,
                                     uint8_t port_num, int index,
                                     union ibv_gid *gid);
sdcclResult_t sdcclWrapIbvQueryQp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                    int attr_mask,
                                    struct ibv_qp_init_attr *init_attr);
sdcclResult_t sdcclWrapIbvAllocPd(struct ibv_pd **ret,
                                    struct ibv_context *context);
sdcclResult_t sdcclWrapIbvDeallocPd(struct ibv_pd *pd);
sdcclResult_t sdcclWrapIbvRegMr(struct ibv_mr **ret, struct ibv_pd *pd,
                                  void *addr, size_t length, int access);
struct ibv_mr *sdcclWrapDirectIbvRegMr(struct ibv_pd *pd, void *addr,
                                        size_t length, int access);
sdcclResult_t sdcclWrapIbvRegMrIova2(struct ibv_mr **ret, struct ibv_pd *pd,
                                       void *addr, size_t length, uint64_t iova,
                                       int access);
/* DMA-BUF support */
sdcclResult_t sdcclWrapIbvRegDmabufMr(struct ibv_mr **ret, struct ibv_pd *pd,
                                        uint64_t offset, size_t length,
                                        uint64_t iova, int fd, int access);
struct ibv_mr *sdcclWrapDirectIbvRegDmabufMr(struct ibv_pd *pd,
                                              uint64_t offset, size_t length,
                                              uint64_t iova, int fd,
                                              int access);
sdcclResult_t sdcclWrapIbvDeregMr(struct ibv_mr *mr);
sdcclResult_t sdcclWrapIbvCreateCompChannel(struct ibv_comp_channel **ret,
                                              struct ibv_context *context);
sdcclResult_t
sdcclWrapIbvDestroyCompChannel(struct ibv_comp_channel *channel);
sdcclResult_t sdcclWrapIbvCreateCq(struct ibv_cq **ret,
                                     struct ibv_context *context, int cqe,
                                     void *cq_context,
                                     struct ibv_comp_channel *channel,
                                     int comp_vector);
sdcclResult_t sdcclWrapIbvDestroyCq(struct ibv_cq *cq);
static inline sdcclResult_t sdcclWrapIbvPollCq(struct ibv_cq *cq,
                                                 int num_entries,
                                                 struct ibv_wc *wc,
                                                 int *num_done) {
  int done = cq->context->ops.poll_cq(
      cq, num_entries, wc); /*returns the number of wcs or 0 on success, a
                               negative number otherwise*/
  if (done < 0) {
    WARN("Call to ibv_poll_cq() returned %d", done);
    return sdcclSystemError;
  }
  *num_done = done;
  return sdcclSuccess;
}
sdcclResult_t sdcclWrapIbvCreateQp(struct ibv_qp **ret, struct ibv_pd *pd,
                                     struct ibv_qp_init_attr *qp_init_attr);
sdcclResult_t sdcclWrapIbvModifyQp(struct ibv_qp *qp,
                                     struct ibv_qp_attr *attr, int attr_mask);
sdcclResult_t sdcclWrapIbvDestroyQp(struct ibv_qp *qp);
sdcclResult_t sdcclWrapIbvQueryEce(struct ibv_qp *qp, struct ibv_ece *ece,
                                     int *supported);
sdcclResult_t sdcclWrapIbvSetEce(struct ibv_qp *qp, struct ibv_ece *ece,
                                   int *supported);
/* SRQ support */
sdcclResult_t sdcclWrapIbvCreateSrq(struct ibv_srq **ret, struct ibv_pd *pd,
                                      struct ibv_srq_init_attr *srq_init_attr);
sdcclResult_t sdcclWrapIbvDestroySrq(struct ibv_srq *srq);

static inline sdcclResult_t
sdcclWrapIbvPostSrqRecv(struct ibv_srq *srq, struct ibv_recv_wr *wr,
                         struct ibv_recv_wr **bad_wr) {
  int ret = srq->context->ops.post_srq_recv(
      srq, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_srq_recv() failed with error %s", strerror(ret));
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

static inline sdcclResult_t
sdcclWrapIbvPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr,
                      struct ibv_send_wr **bad_wr) {
  int ret = qp->context->ops.post_send(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    // Don't warn on ENOMEM (Cannot allocate memory) as it's expected when send
    // queue is full
    if (ret != ENOMEM) {
      WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p",
           strerror(ret), wr, *bad_wr);
    }
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

static inline sdcclResult_t
sdcclWrapIbvPostRecv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                      struct ibv_recv_wr **bad_wr) {
  int ret = qp->context->ops.post_recv(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_recv() failed with error %s", strerror(ret));
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclWrapIbvEventTypeStr(char **ret, enum ibv_event_type event);

#endif // End include guard
