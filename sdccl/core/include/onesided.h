/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Transport-agnostic one-sided handle info and globals.
 * Moved from ib_common.h so that core layer files do not depend on
 * the IB adaptor header.
 ************************************************************************/

#ifndef SDCCL_ONESIDED_H_
#define SDCCL_ONESIDED_H_

#include <stdint.h>

#define SDCCL_MAX_ONE_SIDE_HANDLES 128

struct sdcclOneSideHandleInfo {
  uintptr_t *baseVas;
  uint32_t *rkeys;
  uint32_t *lkeys;
  void *localMrHandle; // local rank's MR handle for deregMr
  void *localRecvComm; // recvComm used for MR registration (PD match)
  // Full-mesh IB connections (including self loopback, aligned with NCCL GIN)
  void **fullSendComms; // [nRanks] per-peer sendComm (NULL if not owner)
  void **fullRecvComms; // [nRanks] per-peer recvComm (NULL if not owner)
  int nRanks;           // number of ranks (for cleanup iteration)
};

// Handle table for per-window MR addressing (aligned with NCCL GIN per-window
// MR)
extern struct sdcclOneSideHandleInfo
    *globalOneSideHandleTable[SDCCL_MAX_ONE_SIDE_HANDLES];
extern int globalOneSideHandleCount;

// Global variable for one-sided signal handles (separate MR, NCCL-style)
extern struct sdcclOneSideHandleInfo *globalOneSideSignalHandles;

// Global variable for one-sided staging handles (for PutValue inline)
extern struct sdcclOneSideHandleInfo *globalOneSideStagingHandles;

#endif // SDCCL_ONESIDED_H_
