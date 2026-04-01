/*************************************************************************
 * Copyright (c) 2023 BAAI. All rights reserved.
 * All rights reserved.
 *
 * IBUC Retransmission Support - Header
 ************************************************************************/

#ifndef SDCCL_IBUC_RETRANS_H_
#define SDCCL_IBUC_RETRANS_H_

#include "sdccl_common.h"
#include "ib_common.h"
#include <stdint.h>
#include <time.h>

// Retransmission constants
#define SDCCL_RETRANS_MAGIC                                                   \
  0xDEADBEEF // Magic number for retransmission header
#define SDCCL_RETRANS_WR_ID                                                   \
  0xFFFFFFFEULL // WR ID for retransmission completions

extern int64_t sdcclParamIbRetransEnable(void);
extern int64_t sdcclParamIbRetransTimeout(void);
extern int64_t sdcclParamIbRetransMaxRetry(void);
extern int64_t sdcclParamIbRetransAckInterval(void);
extern int64_t sdcclParamIbMaxOutstanding(void);

static inline uint64_t sdcclIbGetTimeUs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static inline int sdcclIbSeqLess(uint32_t a, uint32_t b) {
  uint16_t a16 = a & 0xFFFF;
  uint16_t b16 = b & 0xFFFF;
  return (int16_t)(a16 - b16) < 0;
}

static inline int sdcclIbSeqLeq(uint32_t a, uint32_t b) {
  uint16_t a16 = a & 0xFFFF;
  uint16_t b16 = b & 0xFFFF;
  return (int16_t)(a16 - b16) <= 0;
}

sdcclResult_t sdcclIbRetransInit(struct sdcclIbRetransState *state);

sdcclResult_t sdcclIbRetransDestroy(struct sdcclIbRetransState *state);

sdcclResult_t sdcclIbRetransAddPacket(struct sdcclIbRetransState *state,
                                        uint32_t seq, uint32_t size, void *data,
                                        uint64_t remote_addr, uint32_t *lkeys,
                                        uint32_t *rkeys);

sdcclResult_t sdcclIbRetransProcessAck(struct sdcclIbRetransState *state,
                                         struct sdcclIbAckMsg *ack_msg);

sdcclResult_t sdcclIbRetransCheckTimeout(struct sdcclIbRetransState *state,
                                           struct sdcclIbSendComm *comm);

sdcclResult_t sdcclIbRetransRecvPacket(struct sdcclIbRetransState *state,
                                         uint32_t seq,
                                         struct sdcclIbAckMsg *ack_msg,
                                         int *should_ack);

sdcclResult_t sdcclIbRetransPiggybackAck(struct sdcclIbSendFifo *fifo_elem,
                                           struct sdcclIbAckMsg *ack_msg);

sdcclResult_t sdcclIbRetransExtractAck(struct sdcclIbSendFifo *fifo_elem,
                                         struct sdcclIbAckMsg *ack_msg);

static inline uint32_t sdcclIbEncodeImmData(uint32_t seq, uint32_t size) {
  return ((seq & 0xFFFF) << 16) | (size & 0xFFFF);
}

static inline void sdcclIbDecodeImmData(uint32_t imm_data, uint32_t *seq,
                                         uint32_t *size) {
  *seq = (imm_data >> 16) & 0xFFFF;
  *size = imm_data & 0xFFFF;
}

void sdcclIbRetransPrintStats(struct sdcclIbRetransState *state,
                               const char *prefix);

sdcclResult_t sdcclIbCreateCtrlQp(struct ibv_context *context,
                                    struct ibv_pd *pd, uint8_t port_num,
                                    struct sdcclIbCtrlQp *ctrlQp);

sdcclResult_t sdcclIbDestroyCtrlQp(struct sdcclIbCtrlQp *ctrlQp);

sdcclResult_t
sdcclIbSetupCtrlQpConnection(struct ibv_context *context, struct ibv_pd *pd,
                              struct sdcclIbCtrlQp *ctrlQp,
                              uint32_t remote_qpn, union ibv_gid *remote_gid,
                              uint16_t remote_lid, uint8_t port_num,
                              uint8_t link_layer, uint8_t local_gid_index);

sdcclResult_t sdcclIbRetransSendAckViaUd(struct sdcclIbRecvComm *comm,
                                           struct sdcclIbAckMsg *ack_msg,
                                           int devIndex);

sdcclResult_t sdcclIbRetransRecvAckViaUd(struct sdcclIbSendComm *comm,
                                           int devIndex);

sdcclResult_t sdcclIbRetransResendViaSend(struct sdcclIbSendComm *comm,
                                            uint32_t seq);

sdcclResult_t sdcclIbCreateSrq(struct ibv_context *context, struct ibv_pd *pd,
                                 struct sdcclIbSrqMgr *srqMgr);

sdcclResult_t sdcclIbDestroySrq(struct sdcclIbSrqMgr *srqMgr);

sdcclResult_t sdcclIbSrqPostRecv(struct sdcclIbSrqMgr *srqMgr, int count);

#endif // SDCCL_IBUC_RETRANS_H_
