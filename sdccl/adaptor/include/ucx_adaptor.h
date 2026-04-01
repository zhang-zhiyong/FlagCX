/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef UCX_ADAPTOR_H
#define UCX_ADAPTOR_H

#ifdef USE_UCX

#include "check.h"
#include "ib_common.h"
#include "socket.h"
#include <pthread.h>
#include <ucp/api/ucp.h>

// UCX Constants

// UCX Communication State Enum
enum sdcclUcxCommState {
  sdcclUcxCommStateStart = 0,
  sdcclUcxCommStateConnect = 1,
  sdcclUcxCommStateAccept = 3,
};

// UCX Communication Stage Structure
struct sdcclUcxCommStage {
  enum sdcclUcxCommState state;
  uint8_t iteration;
  void *sock;
  void *comm;
};

// UCX Memory Handle
typedef struct sdcclUcxMhandle {
  ucp_mem_h ucpMemh;
  ucp_rkey_h rkey;
  int memType;
} sdcclUcxMhandle_t;

// UCX Endpoint List
struct sdcclUcxEpList {
  struct sdcclSocket *sock;
  struct sdcclUcxEpList *next;
};

// UCX Worker Structure
typedef struct sdcclUcxWorker {
  ucp_worker_h worker; /* ucp worker associated with ctx */
  ucp_context_h ctx;   /* ucp_context bounded to specific device */
  struct sdcclUcxEpList
      *eps; /* oob conection to all endpoints that were opened on this worker */

  int count;        /* number of connections that uses this worker */
  int dev;          /* Managed device */
  pthread_t thread; /* Owner thread */

  struct sdcclUcxWorker *next;
} sdcclUcxWorker_t;

// UCX Listen Handle
typedef struct sdcclUcxListenHandle {
  union sdcclSocketAddress connectAddr; /* reciever socket address */
  uint64_t magic;                        /* random number to help debugging */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used. */
  struct sdcclUcxCommStage stage;
} sdcclUcxListenHandle_t;

// UCX Listen Communicator
typedef struct sdcclUcxListenComm {
  int dev;                  /* device number in sdcclIbDevs which will
                             * be used to recieve data */
  struct sdcclSocket sock; /* socket for OOB connection */
  ucp_context_h ctx; /* ucp_context associated with specific device dev */
  sdcclUcxWorker_t *ucxWorker; /* sdcclUcxWorker created on ctx, worker can
                           be shared between multiple connections */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used.*/
  struct sdcclUcxCommStage stage;
} sdcclUcxListenComm_t;

// UCX Connect Message
typedef struct sdcclUcxConnectMsg {
  size_t addrLen;
} sdcclUcxConnectMsg_t;

// Forward declaration
struct sdcclUcxComm;

// UCX Request Structure
typedef struct sdcclUcxRequest {
  struct sdcclUcxRequest *next; /* Next request in the free list */
  struct sdcclUcxComm *comm;    /* Owning communicator */
  ucp_worker_h worker;           /* Worker for all requests */
  int pending;                   /* How many requests are still pending */
  int count;                     /* How many requests are contained */
  int size[SDCCL_NET_IB_MAX_RECVS];
} sdcclUcxRequest_t;

// UCX GPU Flush Structure
typedef struct sdcclUcxGpuFlush {
  int enabled;
  int hostMem;
  ucp_ep_h flushEp;
} sdcclUcxGpuFlush_t;

// UCX Context Structure
typedef struct sdcclUcxCtx {
  ucp_context_h sdcclUcxCtx;
  sdcclUcxGpuFlush_t gpuFlush;
} sdcclUcxCtx_t;

// UCX Communicator Structure
typedef struct sdcclUcxComm {
  ucp_context_h ctx;            /* ucp_context bounded to specific device */
  sdcclUcxGpuFlush_t gpuFlush; /* flushing handle */
  sdcclUcxWorker_t *ucxWorker; /* ucp worker associated with ctx */
  ucp_ep_h ep;                  /* ucp endpoint created on worker */
  ucp_tag_t tag;  /* datapath tag to filter out message that are not
                     belong to this connnection */
  ucp_tag_t ctag; /* controlpath tag to filter out message that are not
                     belong to this connnection */
  struct sdcclSocket sock; /* socket for OOB connection */
  int ready; /* indicates that receive communicator is fully initialized */
  sdcclUcxRequest_t reqs[MAX_REQUESTS]; /* max inflight requests */
  sdcclUcxRequest_t *freeReq;           /* first request available */
  sdcclUcxConnectMsg_t *msg; /* message to establish reverse connection */
  void *connectReq;           /* msg request */
} sdcclUcxComm_t;

// UCX Macros
#define UCXCHECK(cmd)                                                          \
  do {                                                                         \
    ucs_status_t e = cmd;                                                      \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%s'\n", __FILE__, __LINE__,               \
           ucs_status_string(e));                                              \
      return sdcclInternalError;                                              \
    }                                                                          \
  } while (0)

#define UCXCHECK_VOID(cmd)                                                     \
  do {                                                                         \
    ucs_status_t e = cmd;                                                      \
    if (UCS_OK != e) {                                                         \
      WARN("Failed: UCX error %s:%d '%s'\n", __FILE__, __LINE__,               \
           ucs_status_string(e));                                              \
    }                                                                          \
  } while (0)

#endif // USE_UCX

#endif // UCX_ADAPTOR_H
