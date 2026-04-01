/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_GROUP_H_
#define SDCCL_GROUP_H_

#include "assert.h"
#include "comm.h"

typedef sdcclResult_t (*sdcclInitFunc_t)(sdcclHeteroComm_t *newcomm,
                                           int ndev, sdcclUniqueId commId,
                                           int myrank, int cudaDev);

sdcclResult_t sdcclAsyncInit(sdcclInitFunc_t func,
                               sdcclHeteroComm_t *newcomm, int ndev,
                               sdcclUniqueId commId, int myrank, int cudaDev);

typedef enum sdcclGroupJobState {
  sdcclGroupJobRunning = 0,
  sdcclGroupJobDone = 1,
  sdcclGroupJobJoined = 2,
} sdcclGroupJobState_t;

struct sdcclAsyncJob {
  struct sdcclAsyncJob *next;
  pthread_t thread;
  sdcclResult_t result;
  sdcclResult_t (*func)(struct sdcclAsyncJob *);
  void (*undo)(struct sdcclAsyncJob *);
  void (*destructor)(void *);
  sdcclGroupJobState_t state;
  volatile uint32_t *abortFlag;      /* point to comm abortFlag */
  volatile uint32_t *childAbortFlag; /* point to child abortFlag */
  sdcclHeteroComm_t comm;
};

sdcclResult_t
sdcclAsyncLaunch(struct sdcclAsyncJob *job,
                  sdcclResult_t (*func)(struct sdcclAsyncJob *),
                  void (*undo)(struct sdcclAsyncJob *),
                  void (*destructor)(void *), sdcclHeteroComm_t comm);

struct sdcclGroupJob {
  struct sdcclAsyncJob base;
  struct sdcclHeteroComm **groupCommHeadPtr;
  struct sdcclHeteroComm **groupCommPreconnectHeadPtr;
  sdcclResult_t *groupErrorPtr;
  volatile bool *abortFlagPtr;
  int *groupBlockingPtr;
  struct sdcclIntruQueue<struct sdcclAsyncJob, &sdcclAsyncJob::next>
      *asyncJobsPtr;
  bool initialized;
};

sdcclResult_t sdcclGroupStartInternal();
sdcclResult_t sdcclGroupEndInternal();
sdcclResult_t sdcclAsyncJobComplete(struct sdcclAsyncJob *job);

////////////////////////////////////////////////////////////////////////////////

extern __thread int sdcclGroupDepth; // depth of sdcclGroupStart nesting
extern __thread sdcclResult_t sdcclGroupError;
extern __thread struct sdcclHeteroComm *sdcclGroupCommHead;
extern __thread struct sdcclHeteroComm *sdcclGroupCommPreconnectHead;
extern __thread int sdcclGroupBlocking;
extern __thread struct sdcclGroupJob *sdcclGroupJobMainPtr;
extern __thread struct sdcclGroupJob sdcclGroupJobMain;
extern __thread struct sdcclIntruQueue<struct sdcclAsyncJob,
                                        &sdcclAsyncJob::next>
    sdcclAsyncJobs;

sdcclResult_t sdcclGroupErrCheck(sdcclResult_t ret);
void sdcclGroupCommJoin(struct sdcclHeteroComm *comm);
void sdcclGroupCommPreconnect(struct sdcclHeteroComm *comm);
sdcclResult_t sdcclGroupCommLeave(struct sdcclHeteroComm *comm);
// Not implemented
sdcclResult_t sdcclGroupJobAbort(struct sdcclGroupJob *groupJob);
// Not implemented
sdcclResult_t sdcclGroupJobComplete(struct sdcclGroupJob *groupJob);
sdcclResult_t sdcclHeteroGroupStart();
sdcclResult_t sdcclHeteroGroupEnd();

inline sdcclResult_t sdcclGroupErrCheck(sdcclResult_t ret) {
  if (sdcclGroupDepth > 0) {
    if (ret != sdcclSuccess && ret != sdcclInProgress)
      sdcclGroupError = ret;
  }
  return ret;
}

// Add comm to this thread's group
inline void sdcclGroupCommJoin(struct sdcclHeteroComm *comm) {
  if (comm->groupNext == reinterpret_cast<struct sdcclHeteroComm *>(0x1)) {
    comm->groupNext = sdcclGroupCommHead;
    sdcclGroupCommHead = comm;
  }
}

// Add comm to this thread's group needing preconnect
inline void sdcclGroupCommPreconnect(struct sdcclHeteroComm *comm) {
  if (comm->preconnectNext ==
      reinterpret_cast<struct sdcclHeteroComm *>(0x1)) {
    comm->preconnectNext = sdcclGroupCommPreconnectHead;
    sdcclGroupCommPreconnectHead = comm;
  }
}

// Comm has left group
inline sdcclResult_t sdcclGroupCommLeave(struct sdcclHeteroComm *comm) {
  comm->groupNext = reinterpret_cast<struct sdcclHeteroComm *>(0x1);
  return sdcclSuccess;
}

inline sdcclResult_t sdcclGroupStartInternal() {
  sdcclGroupDepth++;
  return sdcclSuccess;
}

#endif
