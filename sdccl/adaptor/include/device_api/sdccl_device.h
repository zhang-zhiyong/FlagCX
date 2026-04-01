/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * SDCCL Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On Vendor: wraps vendor device API types and functions.
 * On other platforms: provides fallback implementations using IPC.
 *
 * This header is safe to include from both .cu files (nvcc) and
 * .cc files (g++).  Device-only functions (Sections 5-8) are guarded
 * by SDCCL_DEVICE_COMPILE so they are invisible to host compilers
 * on all platforms.
 ************************************************************************/

#ifndef SDCCL_DEVICE_API_H_
#define SDCCL_DEVICE_API_H_

#include <cstddef> // ptrdiff_t, size_t

#include "device_utils.h"
#include "sdccl.h"
#include "sdccl_kernel.h"

// Device traits — provides DeviceAPI with all type/function dispatch.
// Also defines SDCCL_DEVICE_API_VENDOR when vendor backend is active.
// Action types (sdcclDevNet_*, sdcclDescriptorSmem, etc.) are defined
// inside device_traits.h before the backend-specific trait includes.
#include "device_traits.h"

// Forward declaration for typed vendor device comm handle
struct sdcclInnerDevComm;
typedef struct sdcclInnerDevComm *sdcclInnerDevComm_t;

// ============================================================
// Section 1: sdcclDevCommInternal — Host-Side Opaque Handle
//
#define SDCCL_MAX_INTER_PEERS 256

// Backing struct for sdcclDevComm_t (declared in sdccl_kernel.h).
// Populated by sdcclDevCommCreate, freed by sdcclDevCommDestroy.
// Unified capability-based design: baseline always populated,
// IPC and Vendor layers added when available.
// ============================================================
struct sdcclDevCommInternal {
  // ---- Baseline (always set) ----
  int rank, nRanks;
  int intraRank, intraSize;
  void *fifoBuffer; // Device-accessible FIFO (from heteroComm, may be null)
  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint64_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint64_t
      *localBarrierFlags; // this rank's inbox buffer (nLocalRanks × CTA_COUNT)
  uint64_t
      intraBarrierEpoch; // monotonically increasing, set by host before launch
  int nBarriers;         // = SDCCL_DEVICE_CTA_COUNT (needed in kernel)
  // Host-side cleanup bookkeeping (not passed to kernel)
  int barrierIpcIndex;  // index into comm->ipcTable (-1 if no IPC barrier)
  int *localRankToRank; // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;

  // ---- Inter-node signal relay (set if nInterPeers > 0, else nullptr) ----
  uint64_t *interSignalFlags;     // device pointer (from hostGetDevicePointer)
  uint64_t *interSignalFlagsHost; // host pointer (for recv thread + dealloc)
  uint64_t
      interBarrierEpoch; // inter-node epoch (separate from intraBarrierEpoch)
  int nInterPeers;       // number of inter-node peers (set on ALL ranks)
  bool isInterLeader;    // true only on localRank 0 (manages connections)
  int *interPeerRanks;   // global ranks of inter-node peers
  // netAdaptor connections for signal relay (one-sided RDMA atomic)
  void **signalSendComms;  // [nInterPeers] sendComm (for iputSignal)
  void **barrierRecvComms; // [nInterPeers] recvComm (kept alive for QP)
  void *barrierHandleInfo; // sdcclOneSideHandleInfo* with rkeys/baseVas
  // netAdaptor pointer (cached for proxy)
  void *netAdaptorPtr;

  // ---- One-sided Fallback layer (set if interSignalCount/interCounterCount >
  // 0)
  // ----
  uint64_t *signalBuffer; // GPU memory (sdcclMemAlloc), [signalCount] entries
  uint64_t
      *shadowBuffer; // GPU memory (local only, no MR), [signalCount] entries
  uint64_t
      *counterBuffer; // GPU memory (sdcclMemAlloc), [counterCount] entries
  int signalCount;
  int counterCount;
  int contextCount; // = reqs.interContextCount (default 4)
  // Host-only: MR handles + staging for cleanup
  void *signalBufferMr;        // MR handle for signalBuffer
  void *counterBufferMr;       // MR handle for counterBuffer
  void *putValueStagingBuffer; // 8 bytes host-pinned, MR registered
  void *putValueStagingMr;     // MR handle for staging buffer

  // ---- Vendor device comm (set if adaptor->devCommCreate succeeds, else NULL)
  // ----
  sdcclInnerDevComm_t devComm; // Typed vendor handle (per-adaptor struct)
};

// ============================================================
// Section 2: sdcclDevMemInternal — Host-Side Memory Handle
//
// Backing struct for sdcclDevMem_t.
// Created by sdcclDevMemCreate, freed by sdcclDevMemDestroy.
// Unified capability-based design: rawPtr always populated,
// IPC and Window layers added when available.
// Capabilities detected by null-checks:
//   devPeerPtrs != nullptr  → IPC available
//   window != nullptr       → Window available (Vendor or default)
// ============================================================
struct sdcclDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr;   // = buff parameter
  bool hasWindow; // true if any window layer is available (basic or symmetric)
  bool isSymmetric; // true only for SDCCL_WIN_COLL_SYMMETRIC (enables
                    // one-sided)

  // ---- Per-window MR layer (set by sdcclDevMemCreate from handle table) ----
  int mrIndex; // index into globalOneSideHandleTable (-1 if not registered)
  uintptr_t mrBase; // handles[mrIndex]->baseVas[myRank] (cached for device)

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  void **devPeerPtrs; // cached from comm->ipcTable[ipcIndex].devPeerPtrs
  int ipcIndex;       // index into comm->ipcTable (-1 if no IPC)
  int intraRank;      // this rank's local rank index (for IPC local pointer)

  // ---- Window layer (opaque pointer to DeviceAPI::Window) ----
  void *window;    // Points to vendor Window or defaultDeviceImpl::Window
                   // (fallback)
  void *winHandle; // Host-side handle for cleanup
};
#ifndef SDCCL_DEV_MEM_T_DEFINED
#define SDCCL_DEV_MEM_T_DEFINED
typedef struct sdcclDevMemInternal *sdcclDevMem_t;
#endif

// ============================================================
// Section 3: sdcclDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::DevComm which contains all fields.
// On Vendor: DevComm = vendor DevComm
// On default: DevComm = {rank, nRanks, fifoBuffer, barrierPeers, ...}
// ============================================================
struct sdcclDevComm {
  typename DeviceAPI::DevComm _commBase;

  // Wrapper-level fields needed by FIFO encoding on all paths.
  // Populated from sdcclDevCommInternal; safe to be 0 when unused.
  int _signalCount;
  int _counterCount;
  int _contextCount;
  int _nInterPeers;

  SDCCL_HOST_DEVICE_INLINE sdcclDevComm()
      : _commBase(), _signalCount(0), _counterCount(0), _contextCount(0),
        _nInterPeers(0) {}

  SDCCL_HOST_DEVICE_INLINE sdcclDevComm(const sdcclDevCommInternal &di)
      : _signalCount(di.signalCount), _counterCount(di.counterCount),
        _contextCount(di.contextCount), _nInterPeers(di.nInterPeers) {
    if (di.devComm) {
      _commBase = *(typename DeviceAPI::DevComm *)di.devComm;
    } else {
      // Fallback: populate _commBase directly from handle fields.
      // Vendor path: no-op (devComm pointer always set).
      // Dispatch resolved at compile time via DeviceAPI::DevComm.
      DeviceAPI::DevComm::populateFromInternal(_commBase, di);
    }
  }

  // Accessors delegate to _commBase member functions
  SDCCL_DEVICE_INLINE_DECORATOR int getIntraRank() const {
    return _commBase.getIntraRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int getIntraSize() const {
    return _commBase.getIntraSize();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int getRank() const {
    return _commBase.getRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int getSize() const {
    return _commBase.getSize();
  }
  SDCCL_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
    return _commBase.getFifoBuffer();
  }
};

// ============================================================
// Section 4: sdcclDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::Window which contains all fields.
// On Vendor: Window = vendor Window
// On default: Window = {rawPtr, peerPtrs, intraRank, mrBase, mrIndex}
// ============================================================
struct sdcclDevMem {
  typename DeviceAPI::Window _winBase;

  SDCCL_HOST_DEVICE_INLINE sdcclDevMem() : _winBase() {}

  SDCCL_HOST_DEVICE_INLINE sdcclDevMem(const sdcclDevMemInternal &di) {
    if (di.window)
      _winBase = *(typename DeviceAPI::Window *)di.window;
  }

  SDCCL_HOST_DEVICE_INLINE bool hasWindow() const {
    return _winBase.hasAccess();
  }
  SDCCL_HOST_DEVICE_INLINE void *getRawPtr() const {
    return _winBase.getRawPtr();
  }
  SDCCL_HOST_DEVICE_INLINE void **getDevPeerPtrs() const {
    return _winBase.getDevPeerPtrs();
  }
  SDCCL_HOST_DEVICE_INLINE int getMrIndex() const {
    return _winBase.getMrIndex();
  }
};

// ============================================================
// Section 4b: sdcclTeam_t — Team Descriptor
//
// Represents a subset of ranks (intra-node, inter-node, etc.).
// Pure wrapper around DeviceAPI::Team.
// ============================================================
struct sdcclTeam {
  typename DeviceAPI::Team _teamBase;

  SDCCL_HOST_DEVICE_INLINE sdcclTeam() : _teamBase() {}
  SDCCL_HOST_DEVICE_INLINE sdcclTeam(int nr, int r, int s) {
    _teamBase.nRanks = nr;
    _teamBase.rank = r;
    _teamBase.stride = s;
  }
};
typedef struct sdcclTeam sdcclTeam_t;

// ============================================================
// Section 4c: sdcclMulticastHandle — Multicast Memory Handle
//
// Pure wrapper around DeviceAPI::Multimem.
// On Vendor: Multimem = vendor MultimemHandle
// On default: Multimem = {mcBasePtr}
// ============================================================
struct sdcclMulticastHandle {
  typename DeviceAPI::Multimem _multimemBase;

  SDCCL_HOST_DEVICE_INLINE sdcclMulticastHandle() : _multimemBase() {}
};
typedef struct sdcclMulticastHandle sdcclMulticastHandle_t;

// ============================================================
// Section 4d: Barrier Handle Types
//
// sdcclIntraBarrierHandle → vendor intra-barrier handle (Vendor)
// sdcclInterBarrierHandle → vendor inter-barrier handle (Vendor)
// Fallback: placeholder structs (no resource-handle model yet).
// ============================================================
struct sdcclIntraBarrierHandle {
  typename DeviceAPI::IntraBarrierHandle _base;
};
typedef struct sdcclIntraBarrierHandle sdcclIntraBarrierHandle_t;

struct sdcclInterBarrierHandle {
  typename DeviceAPI::InterBarrierHandle _base;
};
typedef struct sdcclInterBarrierHandle sdcclInterBarrierHandle_t;

// Team tag types for barrier session constructors
struct sdcclTeamTagWorld {};
struct sdcclTeamTagIntra {};
struct sdcclTeamTagInter {};

// ============================================================
// Sections 5-8: Device-only functions
//
// These sections use device builtins (threadIdx, __syncthreads, atomics)
// and are only safe under a device compiler (nvcc, hipcc, etc.).
// SDCCL_DEVICE_COMPILE is defined in device_utils.h.
// ============================================================
#ifdef SDCCL_DEVICE_COMPILE

// ============================================================
// Section 5: Team Accessor Functions (Inline Wrappers)
//
// On Vendor: forwards to vendor team functions via _commBase.
// On default: computes from baseline fields in _commBase.
// No #ifdef — DeviceAPI resolves at compile time.
// ============================================================
SDCCL_DEVICE_INLINE_DECORATOR
sdcclTeam_t sdcclTeamIntra(const sdcclDevComm &devComm) {
  sdcclTeam_t team;
  team._teamBase.nRanks = devComm.getIntraSize();
  team._teamBase.rank = devComm.getIntraRank();
  team._teamBase.stride = 1;
  return team;
}
SDCCL_DEVICE_INLINE_DECORATOR
sdcclTeam_t sdcclTeamWorld(const sdcclDevComm &devComm) {
  sdcclTeam_t team;
  team._teamBase.nRanks = devComm.getSize();
  team._teamBase.rank = devComm.getRank();
  team._teamBase.stride = 1;
  return team;
}
SDCCL_DEVICE_INLINE_DECORATOR
sdcclTeam_t sdcclTeamInter(const sdcclDevComm &devComm) {
  sdcclTeam_t team;
  team._teamBase.nRanks = devComm.getSize() / devComm.getIntraSize();
  team._teamBase.rank = devComm.getRank() / devComm.getIntraSize();
  team._teamBase.stride = devComm.getIntraSize();
  return team;
}

// ---- Team Algebra (pure arithmetic on {nRanks, rank, stride}) ----
// These 5 functions are identical on all tiers — no vendor delegation needed.

// Is team b's bPeer also a member of team a?
SDCCL_HOST_DEVICE_INLINE bool
sdcclTeamRankIsMember(sdcclTeam_t a, sdcclTeam_t b, int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int amod = wrank % a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return 0 <= arank && arank < a._teamBase.nRanks && amod == 0;
}

// Convert team b's bPeer to team a's rank.
SDCCL_HOST_DEVICE_INLINE int sdcclTeamRankToTeam(sdcclTeam_t a,
                                                   sdcclTeam_t b, int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return arank;
}

// Extract inner sub-team (first innerSize ranks per stride group).
SDCCL_HOST_DEVICE_INLINE sdcclTeam_t
sdcclTeamInnerFactor(sdcclTeam_t parent, int innerSize) {
  sdcclTeam_t ans;
  ans._teamBase.nRanks = innerSize;
  ans._teamBase.rank = parent._teamBase.rank % innerSize;
  ans._teamBase.stride = parent._teamBase.stride;
  return ans;
}

// Extract outer sub-team (stride groups).
SDCCL_HOST_DEVICE_INLINE sdcclTeam_t
sdcclTeamOuterFactor(sdcclTeam_t parent, int innerSize) {
  sdcclTeam_t ans;
  ans._teamBase.nRanks = parent._teamBase.nRanks / innerSize;
  ans._teamBase.rank = parent._teamBase.rank / innerSize;
  ans._teamBase.stride = parent._teamBase.stride * innerSize;
  return ans;
}

// Return the index'th element of parent minus subset (set difference).
SDCCL_HOST_DEVICE_INLINE int sdcclTeamRankInDifference(sdcclTeam_t parent,
                                                         sdcclTeam_t subset,
                                                         int index) {
  int stride = subset._teamBase.stride / parent._teamBase.stride;
  int below = parent._teamBase.rank - subset._teamBase.rank * stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset._teamBase.nRanks - 1) * stride;
  }
  if (index < below) {
    return index;
  } else if (index - below < (subset._teamBase.nRanks - 1) * (stride - 1)) {
    return below + 1 + ((index - below) / (stride - 1)) * stride +
           (index - below) % (stride - 1);
  } else {
    return below + 1 + (subset._teamBase.nRanks - 1) * stride +
           (index - below - (subset._teamBase.nRanks - 1) * (stride - 1));
  }
}

// ---- DevComm-dependent team conversions ----

// Convert team rank to world rank.
SDCCL_DEVICE_INLINE_DECORATOR int
sdcclTeamRankToWorld(const sdcclDevComm &devComm, sdcclTeam_t team,
                      int rank) {
  return devComm.getRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// Convert team rank to intra-node rank.
SDCCL_DEVICE_INLINE_DECORATOR int
sdcclTeamRankToIntra(const sdcclDevComm &devComm, sdcclTeam_t team,
                      int rank) {
  return devComm.getIntraRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// ============================================================
// Section 6: Cooperative Group Types
//
// Platform-neutral cooperative groups for device-side synchronization.
// Naming: "Tile" = N PEs cooperating (avoids vendor-specific
//         Warp/Wave/Subgroup terms).
//
// All implementations live in DeviceTraits; these are thin wrappers.
// ============================================================

// ---- 6a. sdcclCoopBlock — CTA-level cooperative group ----
struct sdcclCoopBlock {
  typename DeviceAPI::CoopBlock _base;

  SDCCL_HOST_DEVICE_INLINE sdcclCoopBlock() : _base() {}

  SDCCL_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  SDCCL_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6b. sdcclCoopTile<N> — Tile of N threads within a warp ----
template <int N>
struct sdcclCoopTile {
  typename DeviceAPI::template CoopTile<N> _base;

  SDCCL_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int size() const { return N; }
  SDCCL_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
    return _base.laneMask();
  }
  SDCCL_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6c. sdcclCoopThread — single-thread alias ----
typedef sdcclCoopTile<1> sdcclCoopThread;

// ---- 6d. sdcclCoopWarp — full-warp alias ----
typedef sdcclCoopTile<SDCCL_SIMT_WIDTH> sdcclCoopWarp;

// ---- 6e. sdcclCoopTileSpan — consecutive tiles with named barrier ----
struct sdcclCoopTileSpan {
  typename DeviceAPI::CoopTileSpan _base;

  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopTileSpan(int t0, int nTiles, int id)
      : _base(t0, nTiles, id) {}

  SDCCL_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  SDCCL_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6f. sdcclCoopLanes — arbitrary lane bitmask ----
struct sdcclCoopLanes {
  typename DeviceAPI::CoopLanes _base;

  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopLanes(uint32_t lmask = 0xffffffffu)
      : _base(lmask) {}

  SDCCL_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  SDCCL_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
  SDCCL_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
    return _base.getLmask();
  }
};

// ---- 6g. sdcclCoopAny — type-erased cooperative group ----
struct sdcclCoopAny {
  typename DeviceAPI::CoopAny _base;

  sdcclCoopAny() = default;
  sdcclCoopAny(sdcclCoopAny const &) = default;

  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopAny(sdcclCoopBlock b)
      : _base(b._base) {}
  template <int N>
  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopAny(sdcclCoopTile<N> t)
      : _base(t._base) {}
  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopAny(sdcclCoopTileSpan s)
      : _base(s._base) {}
  SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopAny(sdcclCoopLanes l)
      : _base(l._base) {}

  SDCCL_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  SDCCL_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  SDCCL_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6h. Free functions ----

// sdcclCoopGetLaneMask: get the active lane bitmask for a cooperative group
template <int N>
SDCCL_DEVICE_INLINE_DECORATOR uint32_t
sdcclCoopGetLaneMask(sdcclCoopTile<N> coop) {
  return coop.laneMask();
}
SDCCL_DEVICE_INLINE_DECORATOR uint32_t sdcclCoopGetLaneMask(sdcclCoopBlock) {
  return 0xffffffffu;
}
SDCCL_DEVICE_INLINE_DECORATOR uint32_t
sdcclCoopGetLaneMask(sdcclCoopLanes coop) {
  return coop.getLmask();
}
SDCCL_DEVICE_INLINE_DECORATOR uint32_t
sdcclCoopGetLaneMask(sdcclCoopTileSpan) {
  return 0xffffffffu;
}

// sdcclCoopIsThread: compile-time check if group is a single thread
template <int N>
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopIsThread(sdcclCoopTile<N>) {
  return N == 1;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopIsThread(sdcclCoopBlock) {
  return false;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopIsThread(sdcclCoopLanes) {
  return false;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopIsThread(sdcclCoopTileSpan) {
  return false;
}

// sdcclCoopWithinTile: compile-time check if group fits within a single tile
template <int N>
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopWithinTile(sdcclCoopTile<N>) {
  return true;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopWithinTile(sdcclCoopBlock) {
  return false;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopWithinTile(sdcclCoopLanes) {
  return true;
}
SDCCL_DEVICE_INLINE_DECORATOR bool sdcclCoopWithinTile(sdcclCoopTileSpan) {
  return false;
}

// sdcclCoopCoalesced: get a cooperative group of active/safe threads
SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopLanes sdcclCoopCoalesced() {
  return sdcclCoopLanes{DeviceAPI::Intrin::activemask()};
}
template <typename Coop>
SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopWarp sdcclCoopCoalesced(Coop) {
  return sdcclCoopWarp();
}
SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopLanes
sdcclCoopCoalesced(sdcclCoopLanes coop) {
  return coop;
}
template <int N>
SDCCL_DEVICE_INLINE_DECORATOR sdcclCoopTile<N>
sdcclCoopCoalesced(sdcclCoopTile<N> coop) {
  return coop;
}

// ============================================================
// Section 7: sdcclDevBarrier — Barrier Session Wrappers
//
// Thin wrappers delegating to DeviceAPI::DevBarrier<Tag>.
// No #ifdef SDCCL_DEVICE_API_VENDOR — dispatch resolved by DeviceTraits.
// ============================================================

// Primary template
template <typename Tag, typename Coop>
struct sdcclDevBarrier;

// ---- Intra ----
template <typename Coop>
struct sdcclDevBarrier<sdcclBarrierIntra, Coop> {
  typename DeviceAPI::template DevBarrier<sdcclBarrierIntra> _impl;

  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier() : _impl() {}

  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier(Coop coop, const sdcclDevComm &devComm, sdcclTeam_t team,
                   uint32_t index, bool multimem = false,
                   sdcclMulticastHandle mcHandle = {})
      : _impl(devComm._commBase, team._teamBase, index, multimem,
              mcHandle._multimemBase) {}

  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    _impl.arrive(coop._base, order);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    _impl.wait(coop._base, order);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    _impl.sync(coop._base, order);
  }
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// All functions delegate to _winBase member functions — no #ifdef branches.
// On Vendor: forwards to vendor pointer functions via _winBase.
// On default: uses IPC peerPtrs / rawPtr fallback.
// ============================================================
SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetPeerPointer(const sdcclDevMem &mem, size_t offset, sdcclTeam_t team,
                     int peer) {
  return mem._winBase.getPeerPointer(offset, team._teamBase, peer);
}

SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetLocalPointer(const sdcclDevMem &mem, size_t offset) {
  return mem._winBase.getLocalPointer(offset);
}

SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetMulticastPointer(const sdcclDevMem &mem, size_t offset,
                          const sdcclDevComm &devComm) {
  (void)devComm;
  sdcclMulticastHandle_t mmHandle;
  return mem._winBase.getMulticastPointer(offset, mmHandle._multimemBase);
}

// ---- Additional pointer functions ----

// Peer pointer without team parameter.
SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetPeerPointer(const sdcclDevMem &mem, size_t offset, int peer) {
  // Without team, treat as intra-node access
  return mem._winBase.getIntraPointer(offset, peer);
}

// Intra-node rank pointer.
SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetIntraPointer(const sdcclDevMem &mem, size_t offset, int peer) {
  return mem._winBase.getIntraPointer(offset, peer);
}

// Multicast pointer with explicit MulticastHandle.
SDCCL_DEVICE_INLINE_DECORATOR void *
sdcclGetMulticastPointer(const sdcclDevMem &mem, size_t offset,
                          sdcclMulticastHandle_t mmHandle) {
  return mem._winBase.getMulticastPointer(offset, mmHandle._multimemBase);
}

// Reverse lookup: raw pointer → sdcclDevMem.
// Vendor: cooperative search through vendor window table.
// Fallback: not supported (returns empty sdcclDevMem).
template <typename Coop>
SDCCL_DEVICE_INLINE_DECORATOR sdcclDevMem
sdcclFindMem(Coop coop, const sdcclDevComm &devComm, void const *ptr) {
  sdcclDevMem result;
  (void)coop;
  (void)devComm;
  (void)ptr;
  return result;
}

// ============================================================
// Section 8b: sdcclSymPtr<T> — Typed Symmetric Pointer
//
// Value type storing {sdcclDevMem, offset}. Provides typed
// pointer methods and type-aware arithmetic.
// Mirrors vendor's SymPtr<T>.
// ============================================================
template <typename T>
struct sdcclSymPtr {
  sdcclDevMem mem;
  size_t offset;

  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr() : mem(), offset(0) {}
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr(sdcclDevMem m, size_t off)
      : mem(m), offset(off) {}

  // Type conversion (e.g. sdcclSymPtr<float> → sdcclSymPtr<char>)
  template <typename U>
  SDCCL_HOST_DEVICE_INLINE operator sdcclSymPtr<U>() const {
    return {mem, offset};
  }

  // Typed pointer methods (delegate to free functions)
  SDCCL_DEVICE_INLINE_DECORATOR T *localPtr() const {
    return (T *)sdcclGetLocalPointer(mem, offset);
  }
  SDCCL_DEVICE_INLINE_DECORATOR T *peerPtr(sdcclTeam_t team, int peer) const {
    return (T *)sdcclGetPeerPointer(mem, offset, team, peer);
  }
  SDCCL_DEVICE_INLINE_DECORATOR T *peerPtr(int peer) const {
    return (T *)sdcclGetPeerPointer(mem, offset, peer);
  }
  SDCCL_DEVICE_INLINE_DECORATOR T *intraPtr(int peer) const {
    return (T *)sdcclGetIntraPointer(mem, offset, peer);
  }
  SDCCL_DEVICE_INLINE_DECORATOR T *
  multicastPtr(const sdcclDevComm &devComm) const {
    return (T *)sdcclGetMulticastPointer(mem, offset, devComm);
  }
  SDCCL_DEVICE_INLINE_DECORATOR T *
  multicastPtr(sdcclMulticastHandle_t mmHandle) const {
    return (T *)sdcclGetMulticastPointer(mem, offset, mmHandle);
  }

  // Type-aware pointer arithmetic (integer math, no UB)
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(int d) {
    offset += d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(unsigned int d) {
    offset += d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(long d) {
    offset += d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(unsigned long d) {
    offset += d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(long long d) {
    offset += d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator+=(unsigned long long d) {
    offset += d * sizeof(T);
    return *this;
  }

  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(unsigned int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(unsigned long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> &operator-=(unsigned long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
};

// Free operators for sdcclSymPtr<T>
template <typename T, typename Int>
SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> operator+(sdcclSymPtr<T> p, Int d) {
  return p += d;
}
template <typename T, typename Int>
SDCCL_HOST_DEVICE_INLINE sdcclSymPtr<T> operator-(sdcclSymPtr<T> p, Int d) {
  return p -= d;
}
template <typename T>
SDCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(sdcclSymPtr<T> a,
                                              sdcclSymPtr<T> b) {
  return ((ptrdiff_t)a.offset - (ptrdiff_t)b.offset) / (ptrdiff_t)sizeof(T);
}
template <typename T>
SDCCL_HOST_DEVICE_INLINE bool operator==(sdcclSymPtr<T> a,
                                          sdcclSymPtr<T> b) {
  return a.mem._winBase == b.mem._winBase && a.offset == b.offset;
}
template <typename T>
SDCCL_HOST_DEVICE_INLINE bool operator!=(sdcclSymPtr<T> a,
                                          sdcclSymPtr<T> b) {
  return !(a == b);
}

#endif // SDCCL_DEVICE_COMPILE

// ============================================================
// Section 9: Constants
// ============================================================
#ifndef SDCCL_DEVICE_CTA_COUNT
#define SDCCL_DEVICE_CTA_COUNT 36
#endif
#ifndef SDCCL_DEVICE_THREADS_PER_CTA
#define SDCCL_DEVICE_THREADS_PER_CTA 512
#endif

// ============================================================
// Sections 9b-12: sdcclDevNet + Barriers (device-only)
// ============================================================
#ifdef SDCCL_DEVICE_COMPILE

// ============================================================
// Section 10: sdcclDevNet — Device Network (thin wrapper)
//
// Delegates all operations to DeviceAPI::Net which contains
// backend-specific logic (vendor ncclGin or fallback FIFO).
// No #ifdef SDCCL_DEVICE_API_VENDOR in this struct.
// ============================================================
struct sdcclDevNet {
  const sdcclDevComm &_devComm; // for barrier sessions
  typename DeviceAPI::Net _netBase;
  int _contextId;

  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevNet(const sdcclDevComm &dc, int contextIndex = 0)
      : _devComm(dc), _netBase(dc._commBase, contextIndex) {
    int cnt = (dc._contextCount > 0) ? dc._contextCount : 1;
    _contextId = contextIndex % cnt;
  }

  // ---- Two-sided operations ----
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
  send(Coop coop, const sdcclDevMem &mem, size_t offset, size_t count,
       sdcclDataType_t datatype, int peer) const {
    return _netBase.send(coop._base, mem._winBase, offset, count, datatype,
                         peer);
  }
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
  recv(Coop coop, const sdcclDevMem &mem, size_t offset, size_t count,
       sdcclDataType_t datatype, int peer) const {
    return _netBase.recv(coop._base, mem._winBase, offset, count, datatype,
                         peer);
  }
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t term(Coop coop) const {
    return _netBase.term(coop._base);
  }
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t wait(Coop coop) const {
    return _netBase.wait(coop._base);
  }

  // ---- One-sided: put (raw ptr) ----
  template <typename RemoteAction = sdcclDevNet_None,
            typename LocalAction = sdcclDevNet_None,
            typename Coop = sdcclCoopBlock,
            typename DescriptorSmem = sdcclDevNet_None>
  SDCCL_DEVICE_INLINE_DECORATOR void
  put(sdcclTeam_t team, int peer, const sdcclDevMem &dstMem, size_t dstOffset,
      const sdcclDevMem &srcMem, size_t srcOffset, size_t bytes,
      RemoteAction remoteAction = sdcclDevNet_None{},
      LocalAction localAction = sdcclDevNet_None{},
      Coop coop = sdcclCoopBlock{},
      DescriptorSmem descriptor = sdcclDevNet_None{},
      sdcclDeviceScope_t alreadyReleased = sdcclDeviceScopeThread,
      sdcclDeviceScope_t expected_scope = sdcclDeviceScopeDevice) const {
    _netBase.put(team._teamBase, peer, dstMem._winBase, dstOffset,
                 srcMem._winBase, srcOffset, bytes, remoteAction, localAction,
                 coop._base, descriptor, alreadyReleased, expected_scope);
  }

  // ---- One-sided: put (SymPtr) ----
  template <typename T, typename RemoteAction = sdcclDevNet_None,
            typename LocalAction = sdcclDevNet_None,
            typename Coop = sdcclCoopBlock,
            typename DescriptorSmem = sdcclDevNet_None>
  SDCCL_DEVICE_INLINE_DECORATOR void
  put(sdcclTeam_t team, int peer, sdcclSymPtr<T> dst, sdcclSymPtr<T> src,
      size_t nElts, RemoteAction remoteAction = sdcclDevNet_None{},
      LocalAction localAction = sdcclDevNet_None{},
      Coop coop = sdcclCoopBlock{},
      DescriptorSmem descriptor = sdcclDevNet_None{},
      sdcclDeviceScope_t alreadyReleased = sdcclDeviceScopeThread,
      sdcclDeviceScope_t expected_scope = sdcclDeviceScopeDevice) const {
    this->put(team, peer, dst.mem, dst.offset, src.mem, src.offset,
              nElts * sizeof(T), remoteAction, localAction, coop, descriptor,
              alreadyReleased, expected_scope);
  }

  // ---- One-sided: putValue (raw ptr) ----
  template <typename T, typename RemoteAction = sdcclDevNet_None,
            typename Coop = sdcclCoopBlock,
            typename DescriptorSmem = sdcclDevNet_None>
  SDCCL_DEVICE_INLINE_DECORATOR void
  putValue(sdcclTeam_t team, int peer, const sdcclDevMem &dstMem,
           size_t dstOffset, T value,
           RemoteAction remoteAction = sdcclDevNet_None{},
           Coop coop = sdcclCoopBlock{},
           DescriptorSmem descriptor = sdcclDevNet_None{},
           sdcclDeviceScope_t alreadyReleased = sdcclDeviceScopeThread,
           sdcclDeviceScope_t expected_scope = sdcclDeviceScopeDevice) const {
    _netBase.putValue(team._teamBase, peer, dstMem._winBase, dstOffset, value,
                      remoteAction, coop._base, descriptor, alreadyReleased,
                      expected_scope);
  }

  // ---- One-sided: putValue (SymPtr) ----
  template <typename T, typename RemoteAction = sdcclDevNet_None,
            typename Coop = sdcclCoopBlock,
            typename DescriptorSmem = sdcclDevNet_None>
  SDCCL_DEVICE_INLINE_DECORATOR void
  putValue(sdcclTeam_t team, int peer, sdcclSymPtr<T> dst, T value,
           RemoteAction remoteAction = sdcclDevNet_None{},
           Coop coop = sdcclCoopBlock{},
           DescriptorSmem descriptor = sdcclDevNet_None{},
           sdcclDeviceScope_t alreadyReleased = sdcclDeviceScopeThread,
           sdcclDeviceScope_t expected_scope = sdcclDeviceScopeDevice) const {
    this->putValue(team, peer, dst.mem, dst.offset, value, remoteAction, coop,
                   descriptor, alreadyReleased, expected_scope);
  }

  // ---- One-sided: signal ----
  template <typename RemoteAction, typename Coop = sdcclCoopBlock,
            typename DescriptorSmem = sdcclDevNet_None>
  SDCCL_DEVICE_INLINE_DECORATOR void
  signal(sdcclTeam_t team, int peer, RemoteAction remoteAction,
         Coop coop = sdcclCoopBlock{},
         DescriptorSmem descriptor = sdcclDevNet_None{},
         sdcclDeviceScope_t alreadyReleased = sdcclDeviceScopeThread,
         sdcclDeviceScope_t expected_scope = sdcclDeviceScopeDevice) const {
    _netBase.signal(team._teamBase, peer, remoteAction, coop._base, descriptor,
                    alreadyReleased, expected_scope);
  }

  // ---- One-sided: flush ----
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void flush(
      Coop coop,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    _netBase.flush(coop._base, order);
  }

  // ---- One-sided: get (Fallback only, vendor stub exists for compilation)
  // ----
  template <typename Coop = sdcclCoopBlock>
  SDCCL_DEVICE_INLINE_DECORATOR void
  get(sdcclTeam_t team, int peer, const sdcclDevMem &srcMem, size_t srcOffset,
      const sdcclDevMem &dstMem, size_t dstOffset, size_t bytes,
      Coop coop = sdcclCoopBlock{}) const {
    _netBase.get(team._teamBase, peer, srcMem._winBase, srcOffset,
                 dstMem._winBase, dstOffset, bytes, coop._base);
  }

  // ---- Signal operations ----
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void waitSignal(
      Coop coop, sdcclDevNetSignal_t signal, uint64_t least, int bits = 64,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    _netBase.waitSignal(coop._base, signal, least, bits, order);
  }

  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void waitSignalMeetShadow(
      Coop coop, sdcclDevNetSignal_t signal, int bits = 64,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    _netBase.waitSignalMeetShadow(coop._base, signal, bits, order);
  }

  template <typename Coop, typename Uint>
  SDCCL_DEVICE_INLINE_DECORATOR void waitSignalFollowShadow(
      Coop coop, sdcclDevNetSignal_t signal, Uint leastDelta, Uint *before,
      Uint *delta, int bits = 64,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    _netBase.waitSignalFollowShadow(coop._base, signal, leastDelta, before,
                                    delta, bits, order);
  }

  // ---- Shadow manipulation ----
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t *
  getSignalShadowPtr(sdcclDevNetSignal_t signal) const {
    return _netBase.getSignalShadowPtr(signal);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  increaseSignalShadow(sdcclDevNetSignal_t signal, uint64_t delta) const {
    _netBase.increaseSignalShadow(signal, delta);
  }

  SDCCL_DEVICE_INLINE_DECORATOR uint64_t readSignal(
      sdcclDevNetSignal_t signal, int bits = 64,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    return _netBase.readSignal(signal, bits, order);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  resetSignal(sdcclDevNetSignal_t signal) const {
    _netBase.resetSignal(signal);
  }

  // ---- Counter operations ----
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop coop, sdcclDevNetCounter_t counter, uint64_t least, int bits = 56,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    _netBase.waitCounter(coop._base, counter, least, bits, order);
  }

  SDCCL_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      sdcclDevNetCounter_t counter, int bits = 56,
      sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcquire) const {
    return _netBase.readCounter(counter, bits, order);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  resetCounter(sdcclDevNetCounter_t counter) const {
    _netBase.resetCounter(counter);
  }
};

// ============================================================
// Section 11: sdcclDevBarrier<sdcclBarrierInter> — Inter-Node Barrier
// ============================================================
// ---- Inter ----
template <typename Coop>
struct sdcclDevBarrier<sdcclBarrierInter, Coop> {
  typename DeviceAPI::template DevBarrier<sdcclBarrierInter> _impl;

  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier() : _impl() {}

  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier(Coop coop, const sdcclDevNet &net, sdcclTeam_t team,
                   uint32_t index)
      : _impl(net._netBase, net._devComm._commBase, team._teamBase, index,
              net._devComm._nInterPeers) {}

  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
         sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.arrive(coop._base, order, fence);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.wait(coop._base, order, fence);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.sync(coop._base, order, fence);
  }
};

// ---- World ----
template <typename Coop>
struct sdcclDevBarrier<sdcclBarrierWorld, Coop> {
  typename DeviceAPI::template DevBarrier<sdcclBarrierWorld> _impl;

  // World barrier (intra + inter)
  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier(Coop coop, sdcclTeamTagWorld, const sdcclDevNet &net,
                   uint32_t index, bool multimem = false)
      : _impl(sdcclBarrierWorld::World{}, net._netBase, net._devComm._commBase,
              index, multimem, net._devComm._nInterPeers) {}

  // Intra-only barrier
  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier(Coop coop, sdcclTeamTagIntra, const sdcclDevComm &devComm,
                   uint32_t index, bool multimem = false)
      : _impl(sdcclBarrierWorld::Intra{},
              typename DeviceAPI::Net(devComm._commBase, 0), devComm._commBase,
              index, multimem, 0) {}

  // Inter-only barrier (ncclTeamTagRail on NVIDIA)
  SDCCL_DEVICE_INLINE_DECORATOR
  sdcclDevBarrier(Coop coop, sdcclTeamTagInter, const sdcclDevNet &net,
                   uint32_t index, bool multimem = false)
      : _impl(sdcclBarrierWorld::Inter{}, net._netBase, net._devComm._commBase,
              index, multimem, net._devComm._nInterPeers) {}

  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
         sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.arrive(coop._base, order, fence);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.wait(coop._base, order, fence);
  }

  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _impl.sync(coop._base, order, fence);
  }
};

// Backward-compatible aliases
template <typename Coop>
using sdcclIntraBarrierSession = sdcclDevBarrier<sdcclBarrierIntra, Coop>;
template <typename Coop>
using sdcclInterBarrierSession = sdcclDevBarrier<sdcclBarrierInter, Coop>;
template <typename Coop>
using sdcclBarrierSession = sdcclDevBarrier<sdcclBarrierWorld, Coop>;

#endif // SDCCL_DEVICE_COMPILE (Sections 9b-12)

#endif // SDCCL_DEVICE_API_H_
