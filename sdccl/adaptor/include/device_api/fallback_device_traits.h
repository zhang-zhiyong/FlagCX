/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Fallback Device Traits — Common IPC-based implementation.
 *
 * DeviceTraits<Fallback<PlatformTag>> provides:
 *   - Intrin, Atomic: inherited from PlatformTraits<PlatformTag> via using
 *   - Window:   IPC peer pointers + raw pointer
 *   - DevComm:  rank/size + IPC barriers + signal buffers
 *   - Team:     pure arithmetic (nRanks, rank, stride)
 *   - Multimem: placeholder (no multicast)
 *
 * This partial specialization is written ONCE and works for any platform.
 * Adding a new platform (e.g. Cambricon) requires zero changes here.
 ************************************************************************/

#ifndef SDCCL_FALLBACK_DEVICE_TRAITS_H_
#define SDCCL_FALLBACK_DEVICE_TRAITS_H_

#include "sdccl_kernel.h"

template <typename PlatformTag>
struct DeviceTraits<Fallback<PlatformTag>> {
  // Platform capabilities (resolved via PlatformTag)
  using Intrin = typename PlatformTraits<PlatformTag>::Intrin;
  using Atomic = typename PlatformTraits<PlatformTag>::Atomic;

  // ---- Team: Pure arithmetic ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Multimem: Placeholder ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Window: IPC + MR + rawPtr ----
  struct Window {
    void *rawPtr;     // Raw memory pointer (always valid)
    void **peerPtrs;  // IPC peer pointers (nullptr if no IPC)
    int intraRank;    // Local rank index
    uintptr_t mrBase; // MR base VA
    int mrIndex;      // MR table index

    SDCCL_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      if (peerPtrs) {
        int index = team.rank + (peer - team.rank) * team.stride;
        return (char *)peerPtrs[index] + offset;
      }
      return nullptr;
    }

    SDCCL_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      if (peerPtrs)
        return (char *)peerPtrs[intraRank] + offset;
      return (char *)rawPtr + offset;
    }

    SDCCL_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      if (peerPtrs)
        return (char *)peerPtrs[peer] + offset;
      return nullptr;
    }

    SDCCL_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      (void)offset;
      (void)mm;
      return nullptr; // Multicast not available in fallback
    }

    SDCCL_HOST_DEVICE_INLINE bool hasAccess() const {
      return rawPtr != nullptr || peerPtrs != nullptr;
    }
    SDCCL_HOST_DEVICE_INLINE void *getRawPtr() const { return rawPtr; }
    SDCCL_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return peerPtrs; }
    SDCCL_HOST_DEVICE_INLINE int getMrIndex() const { return mrIndex; }

    SDCCL_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return rawPtr == o.rawPtr;
    }
    SDCCL_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- DevComm: All fallback layers ----
  struct DevComm {
    // Baseline
    int rank, nRanks;
    int intraRank, intraSize;
    void *fifoBuffer;

    // IPC barriers
    uint64_t **barrierPeers;
    uint64_t intraBarrierEpoch;
    int nBarriers;

    // Inter-node signal relay
    uint64_t *interSignalFlags;
    int nInterPeers;
    bool isInterLeader;
    uint64_t interBarrierEpoch;

    // One-sided fallback
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextCount;

    SDCCL_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    SDCCL_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    SDCCL_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    SDCCL_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    SDCCL_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
      return fifoBuffer;
    }

    // Populate from host-side handle (deferred template avoids forward-decl)
    template <typename DI>
    static SDCCL_HOST_DEVICE_INLINE void populateFromInternal(DevComm &dc,
                                                               const DI &di) {
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.fifoBuffer = di.fifoBuffer;
      dc.barrierPeers = di.barrierPeers;
      dc.intraBarrierEpoch = di.intraBarrierEpoch;
      dc.nBarriers = di.nBarriers;
      dc.interSignalFlags = di.interSignalFlags;
      dc.nInterPeers = di.nInterPeers;
      dc.isInterLeader = di.isInterLeader;
      dc.interBarrierEpoch = di.interBarrierEpoch;
      dc.signalBuffer = di.signalBuffer;
      dc.shadowBuffer = di.shadowBuffer;
      dc.counterBuffer = di.counterBuffer;
      dc.signalCount = di.signalCount;
      dc.counterCount = di.counterCount;
      dc.contextCount = di.contextCount;
    }
  };

  // ---- Coop types: aliased from PlatformTraits ----
  using CoopBlock = typename PlatformTraits<PlatformTag>::CoopBlock;
  template <int N>
  using CoopTile = typename PlatformTraits<PlatformTag>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<PlatformTag>::CoopThread;
  using CoopWarp = typename PlatformTraits<PlatformTag>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<PlatformTag>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<PlatformTag>::CoopLanes;
  using CoopAny = typename PlatformTraits<PlatformTag>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty on fallback ----
  struct DescriptorSmem {};

  // ---- DevBarrier alias: delegates to standalone DevBarrier<Backend, Tag>
  // ----
  template <typename T>
  using DevBarrier = ::DevBarrier<Fallback<PlatformTag>, T>;

  // ============================================================
  // Static FIFO helpers (used by Net and InterBarrierSession)
  // ============================================================

  // Build trd common header: prim(4) | peerRank(20) | primSpecific(36)
  SDCCL_DEVICE_INLINE_DECORATOR
  static uint64_t buildTrd(uint64_t prim, uint64_t peerRank,
                           uint64_t primSpecific) {
    return ((prim & sdcclTriggerMask(sdcclDeviceTriggerBitsPrim))
            << sdcclDeviceTriggerOffPrim) |
           ((peerRank & sdcclTriggerMask(sdcclDeviceTriggerBitsPeerRank))
            << sdcclDeviceTriggerOffPeerRank) |
           primSpecific;
  }

  // Enqueue a trigger into the device FIFO buffer.
  // Atomically reserves a slot, waits for space, writes 3 words.
  SDCCL_DEVICE_INLINE_DECORATOR
  static sdcclResult_t fifoEnqueue(void *fifoBuffer, uint64_t fstVal,
                                    uint64_t sndVal, uint64_t trdVal) {
    uint64_t *buffer = (uint64_t *)fifoBuffer;
    uint64_t capacity = Atomic::load(&buffer[sdcclFifoIdxCapacity],
                                     sdcclDeviceMemoryOrderRelaxed);

    // 1. Atomically reserve a slot
    uint64_t mySlot =
        Atomic::fetchAdd(&buffer[sdcclFifoIdxProduced], (uint64_t)1,
                         sdcclDeviceMemoryOrderAcqRel);

    // 2. Wait until there's space (mySlot - consumed < capacity)
    int iter = 0;
    while ((int64_t)(mySlot - Atomic::load(&buffer[sdcclFifoIdxConsumed],
                                           sdcclDeviceMemoryOrderAcquire)) >=
           (int64_t)capacity) {
      Intrin::spinBackoff(iter++);
    }

    // 3. Compute slot index and get pointers to slot's 3 uint64_t fields
    uint64_t idx = mySlot % capacity;
    uint64_t *slotFst = buffer + sdcclFifoIdxData +
                        idx * (sizeof(sdcclDeviceTrigger) / sizeof(uint64_t));
    uint64_t *slotSnd = slotFst + 1;
    uint64_t *slotTrd = slotFst + 2;

    // 4. Write fst, snd (payload, relaxed)
    Atomic::store(slotFst, fstVal, sdcclDeviceMemoryOrderRelaxed);
    Atomic::store(slotSnd, sndVal, sdcclDeviceMemoryOrderRelaxed);

    // 5. Write trd with valid bit (release ensures payload visible before
    // control)
    Atomic::store(slotTrd, trdVal | sdcclDeviceTriggerValidMask,
                  sdcclDeviceMemoryOrderRelease);

    return sdcclSuccess;
  }

  // Flush: snapshot produced, then spin until consumed >= snapshot.
  SDCCL_DEVICE_INLINE_DECORATOR
  static sdcclResult_t fifoFlush(void *fifoBuffer) {
    uint64_t *buffer = (uint64_t *)fifoBuffer;
    uint64_t snapshot = Atomic::load(&buffer[sdcclFifoIdxProduced],
                                     sdcclDeviceMemoryOrderAcquire);
    int iter = 0;
    while (Atomic::load(&buffer[sdcclFifoIdxConsumed],
                        sdcclDeviceMemoryOrderAcquire) < snapshot) {
      Intrin::spinBackoff(iter++);
    }
    return sdcclSuccess;
  }

  // Wait: enqueue PrimWait + flush.
  SDCCL_DEVICE_INLINE_DECORATOR
  static sdcclResult_t fifoWait(void *fifoBuffer) {
    fifoEnqueue(fifoBuffer, 0, 0, buildTrd(sdcclDevicePrimWait, 0, 0));
    return fifoFlush(fifoBuffer);
  }

  // ============================================================
  // Net: FIFO-based two-sided + one-sided + GPU-spin signal/counter
  // ============================================================
  struct Net {
    void *fifoBuffer;
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextId;

    SDCCL_DEVICE_INLINE_DECORATOR
    Net(const DevComm &dc, int contextIndex)
        : fifoBuffer(dc.fifoBuffer), signalBuffer(dc.signalBuffer),
          shadowBuffer(dc.shadowBuffer), counterBuffer(dc.counterBuffer),
          signalCount(dc.signalCount), counterCount(dc.counterCount) {
      int cnt = (dc.contextCount > 0) ? dc.contextCount : 1;
      contextId = contextIndex % cnt;
    }

    // ---- Two-sided FIFO encoders ----
    SDCCL_DEVICE_INLINE_DECORATOR void
    enqueueFifoSend(const Window &mem, size_t offset, size_t count,
                    sdcclDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          buildTrd(sdcclDevicePrimSend, peer,
                   ((uint64_t)datatype << sdcclDeviceTriggerOffDatatype) |
                       ((uint64_t)count << sdcclDeviceTriggerOffCount)));
    }

    SDCCL_DEVICE_INLINE_DECORATOR void
    enqueueFifoRecv(const Window &mem, size_t offset, size_t count,
                    sdcclDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          buildTrd(sdcclDevicePrimRecv, peer,
                   ((uint64_t)datatype << sdcclDeviceTriggerOffDatatype) |
                       ((uint64_t)count << sdcclDeviceTriggerOffCount)));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    enqueueFifoTerm(int totalCoops) const {
      return fifoEnqueue(fifoBuffer, (uint64_t)totalCoops, 0,
                         buildTrd(sdcclDevicePrimTerm, 0, 0));
    }

    // ---- Two-sided Coop-scope operations ----
    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    send(Coop coop, Window mem, size_t offset, size_t count,
         sdcclDataType_t datatype, int peer) const {
      coop.sync();
      if (coop.threadRank() == 0)
        enqueueFifoSend(mem, offset, count, datatype, peer);
      coop.sync();
      return sdcclSuccess;
    }

    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    recv(Coop coop, Window mem, size_t offset, size_t count,
         sdcclDataType_t datatype, int peer) const {
      coop.sync();
      if (coop.threadRank() == 0)
        enqueueFifoRecv(mem, offset, count, datatype, peer);
      coop.sync();
      return sdcclSuccess;
    }

    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t term(Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int totalCoops = (SDCCL_GRID_DIM_X * SDCCL_BLOCK_DIM_X) / coop.size();
        enqueueFifoTerm(totalCoops);
      }
      coop.sync();
      return sdcclSuccess;
    }

    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t wait(Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0)
        fifoWait(fifoBuffer);
      coop.sync();
      return sdcclSuccess;
    }

    // ---- One-sided FIFO encoders ----
    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    enqueueFifoPut(size_t srcOffset, size_t dstOffset, size_t size, int peer,
                   int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << sdcclDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << sdcclDeviceTriggerOffDstOffset);
      uint64_t sndValue = (uint64_t)size << sdcclDeviceTriggerOffSize;
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << sdcclDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << sdcclDeviceTriggerOffDstMrIdx);
      return fifoEnqueue(fifoBuffer, fstValue, sndValue,
                         buildTrd(sdcclDevicePrimPut, peer, trdSpecific));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    enqueueFifoGet(size_t srcOffset, size_t dstOffset, size_t size, int peer,
                   int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << sdcclDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << sdcclDeviceTriggerOffDstOffset);
      uint64_t sndValue = (uint64_t)size << sdcclDeviceTriggerOffSize;
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << sdcclDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << sdcclDeviceTriggerOffDstMrIdx);
      return fifoEnqueue(fifoBuffer, fstValue, sndValue,
                         buildTrd(sdcclDevicePrimGet, peer, trdSpecific));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t
    enqueueFifoSignalRaw(int signalIdx, int peer) const {
      uint64_t trdSpecific = ((uint64_t)(contextId * signalCount + signalIdx)
                              << sdcclDeviceTriggerOffSignalIdxSig) |
                             ((uint64_t)1 << sdcclDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(sdcclDevicePrimSignal, peer, trdSpecific));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t enqueueFifoSignal(
        int signalIdx, uint32_t value, int peer, uint64_t bufferType) const {
      int combinedIdx = (bufferType == 0)
                            ? (contextId * signalCount + signalIdx)
                            : (contextId * counterCount + signalIdx);
      uint64_t trdSpecific =
          ((uint64_t)bufferType << sdcclDeviceTriggerOffBufferType) |
          ((uint64_t)combinedIdx << sdcclDeviceTriggerOffSignalIdxSig) |
          ((uint64_t)(value & 0xFFFFu) << sdcclDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(sdcclDevicePrimSignal, peer, trdSpecific));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t enqueueFifoPutValue(
        size_t dstOffset, uint64_t value, int peer, int dstMrIdx) const {
      uint64_t fstValue = (uint64_t)dstOffset &
                          sdcclTriggerMask(sdcclDeviceTriggerBitsDstOffset);
      uint64_t trdSpecific = (uint64_t)dstMrIdx
                             << sdcclDeviceTriggerOffDstMrIdx;
      return fifoEnqueue(fifoBuffer, fstValue, value,
                         buildTrd(sdcclDevicePrimPutValue, peer, trdSpecific));
    }

    SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t enqueueFifoPutSignal(
        size_t srcOffset, size_t dstOffset, size_t size, int signalIdx,
        uint32_t signalValue, int peer, int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << sdcclDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << sdcclDeviceTriggerOffDstOffset);
      uint64_t sndValue = ((uint64_t)size << sdcclDeviceTriggerOffSize) |
                          ((uint64_t)(signalValue & 0xFFFFu)
                           << sdcclDeviceTriggerOffSignalValuePut);
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << sdcclDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << sdcclDeviceTriggerOffDstMrIdx) |
          ((uint64_t)(contextId * signalCount + signalIdx)
           << sdcclDeviceTriggerOffSignalIdx);
      return fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          buildTrd(sdcclDevicePrimPutSignal, peer, trdSpecific));
    }

    // ---- MR offset helper ----
    SDCCL_DEVICE_INLINE_DECORATOR
    static size_t toDataOffset(const Window &win, size_t off) {
      void *ptr = win.getLocalPointer(off);
      return (uintptr_t)ptr - win.mrBase;
    }

    // ---- Action decomposition helpers ----
    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool isSignal(T) const {
      return false;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool
    isSignal(sdcclDevNet_SignalInc) const {
      return true;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool
    isSignal(sdcclDevNet_SignalAdd) const {
      return true;
    }

    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr int getSignalIdx(T) const {
      return 0;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr int
    getSignalIdx(sdcclDevNet_SignalInc a) const {
      return a.signal;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr int
    getSignalIdx(sdcclDevNet_SignalAdd a) const {
      return a.signal;
    }

    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr uint32_t getSignalValue(T) const {
      return 0;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr uint32_t
    getSignalValue(sdcclDevNet_SignalInc) const {
      return 1;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr uint32_t
    getSignalValue(sdcclDevNet_SignalAdd a) const {
      return (uint32_t)a.value;
    }

    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool canFuseSignal(T) const {
      return false;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool
    canFuseSignal(sdcclDevNet_SignalInc) const {
      return true;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool
    canFuseSignal(sdcclDevNet_SignalAdd) const {
      return true;
    }

    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool isCounter(T) const {
      return false;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr bool
    isCounter(sdcclDevNet_CounterInc) const {
      return true;
    }

    template <typename T>
    SDCCL_DEVICE_INLINE_DECORATOR constexpr int getCounterIdx(T) const {
      return 0;
    }
    SDCCL_DEVICE_INLINE_DECORATOR constexpr int
    getCounterIdx(sdcclDevNet_CounterInc a) const {
      return a.counter;
    }

    // ---- One-sided: put (raw Window) ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    SDCCL_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        sdcclDeviceScope_t ar, sdcclDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        if (canFuseSignal(ra)) {
          enqueueFifoPutSignal(srcDataOff, dstDataOff, bytes, getSignalIdx(ra),
                               getSignalValue(ra), peer, src.getMrIndex(),
                               dst.getMrIndex());
        } else {
          enqueueFifoPut(srcDataOff, dstDataOff, bytes, peer, src.getMrIndex(),
                         dst.getMrIndex());
          if (isSignal(ra))
            enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
        }
        if (isCounter(la))
          enqueueFifoSignal(getCounterIdx(la), 1, 0, 1);
      }
      coop.sync();
    }

    // ---- One-sided: get (Coop-scope, Fallback only) ----
    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      (void)team;
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoGet(srcDataOff, dstDataOff, bytes, peer, src.getMrIndex(),
                       dst.getMrIndex());
      }
      coop.sync();
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    SDCCL_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc desc, sdcclDeviceScope_t ar,
             sdcclDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoPutValue(dstDataOff, (uint64_t)value, peer,
                            dst.getMrIndex());
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    SDCCL_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           sdcclDeviceScope_t ar, sdcclDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
      }
      coop.sync();
    }

    // ---- flush: drain FIFO (snapshot-spin, no PrimWait) ----
    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, sdcclDeviceMemoryOrder_t order) const {
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0 && fifoBuffer != nullptr) {
        fifoFlush(fifoBuffer);
      }
      coop.sync();
    }

    // ---- waitSignal: GPU spin on signalBuffer[ctx*N+id] ----
    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, sdcclDevNetSignal_t signalId, uint64_t least,
               int bits, sdcclDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = contextId * signalCount + (int)signalId;
        int iter = 0;
        while (Atomic::load(&signalBuffer[idx],
                            sdcclDeviceMemoryOrderAcquire) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, sdcclDevNetSignal_t signalId, int bits,
                         sdcclDeviceMemoryOrder_t order) const {
      int idx = contextId * signalCount + (int)signalId;
      uint64_t shadow = ((volatile uint64_t *)shadowBuffer)[idx];
      waitSignal(coop, signalId, shadow, bits, order);
    }

    template <typename Coop, typename Uint>
    SDCCL_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, sdcclDevNetSignal_t signalId, Uint delta,
                           Uint *outSignalValue, Uint *outShadowValue, int bits,
                           sdcclDeviceMemoryOrder_t order) const {
      int idx = contextId * signalCount + (int)signalId;
      uint64_t shadow = ((volatile uint64_t *)shadowBuffer)[idx];
      uint64_t target = shadow + (uint64_t)delta;
      waitSignal(coop, signalId, target, bits, order);
      shadowBuffer[idx] = target;
      if (outSignalValue)
        *outSignalValue = (Uint)target;
      if (outShadowValue)
        *outShadowValue = (Uint)target;
    }

    // ---- Shadow manipulation ----
    SDCCL_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(sdcclDevNetSignal_t signalId) const {
      return &shadowBuffer[contextId * signalCount + (int)signalId];
    }

    SDCCL_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(sdcclDevNetSignal_t signalId, uint64_t delta) const {
      shadowBuffer[contextId * signalCount + (int)signalId] += delta;
    }

    SDCCL_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(sdcclDevNetSignal_t signalId, int bits,
               sdcclDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      int idx = contextId * signalCount + (int)signalId;
      return Atomic::load(&signalBuffer[idx], sdcclDeviceMemoryOrderAcquire);
    }

    SDCCL_DEVICE_INLINE_DECORATOR void
    resetSignal(sdcclDevNetSignal_t signalId) const {
      int idx = contextId * signalCount + (int)signalId;
      Atomic::store(&signalBuffer[idx], (uint64_t)0,
                    sdcclDeviceMemoryOrderRelease);
    }

    // ---- Counter: GPU spin on counterBuffer[ctx*N+id] ----
    template <typename Coop>
    SDCCL_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, sdcclDevNetCounter_t counterId, uint64_t least,
                int bits, sdcclDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = contextId * counterCount + (int)counterId;
        int iter = 0;
        while (Atomic::load(&counterBuffer[idx],
                            sdcclDeviceMemoryOrderAcquire) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    SDCCL_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(sdcclDevNetCounter_t counterId, int bits,
                sdcclDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      int idx = contextId * counterCount + (int)counterId;
      return Atomic::load(&counterBuffer[idx], sdcclDeviceMemoryOrderAcquire);
    }

    SDCCL_DEVICE_INLINE_DECORATOR void
    resetCounter(sdcclDevNetCounter_t counterId) const {
      int idx = contextId * counterCount + (int)counterId;
      Atomic::store(&counterBuffer[idx], (uint64_t)0,
                    sdcclDeviceMemoryOrderRelease);
    }
  };
};

// ============================================================
// DevBarrier specializations for Fallback<P>
//
// Standalone partial specializations (C++ forbids explicit specialization
// of member templates inside a partial class specialization).
// ============================================================

// ---- DevBarrier<Fallback<P>, sdcclBarrierIntra> ----
// Thread-striped per-peer inbox barrier using IPC-mapped atomics.
template <typename P>
struct DevBarrier<Fallback<P>, sdcclBarrierIntra> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using DevComm = typename DeviceTraits<Fallback<P>>::DevComm;
  using Team = typename DeviceTraits<Fallback<P>>::Team;
  using Multimem = typename DeviceTraits<Fallback<P>>::Multimem;

  uint64_t **_peerBuffers;
  int _nRanks, _myRank;
  int _nBarriers;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Default ctor (no-op, for barrier composition)
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier()
      : _peerBuffers(nullptr), _nRanks(0), _myRank(0), _nBarriers(0),
        _ctaIndex(0), _epoch(0) {}

  // Active ctor
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier(const DevComm &dc, Team team, uint32_t index, bool = false,
             const Multimem & = {})
      : _peerBuffers(dc.barrierPeers), _nRanks(team.nRanks), _myRank(team.rank),
        _nBarriers(dc.nBarriers), _ctaIndex(index),
        _epoch(dc.intraBarrierEpoch) {}

  // arrive: thread-striped store epoch+1 to each peer's inbox slot for me
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    coop.sync();
    for (int i = SDCCL_THREAD_IDX_X; i < _nRanks - 1;
         i += SDCCL_BLOCK_DIM_X) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      Atomic::store(&_peerBuffers[peer][_myRank * _nBarriers + _ctaIndex],
                    _epoch + 1, sdcclDeviceMemoryOrderRelease);
    }
  }

  // wait: thread-striped spin on own inbox slots from each peer
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    for (int i = SDCCL_THREAD_IDX_X; i < _nRanks - 1;
         i += SDCCL_BLOCK_DIM_X) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      int iter = 0;
      while (Atomic::load(&_peerBuffers[_myRank][peer * _nBarriers + _ctaIndex],
                          sdcclDeviceMemoryOrderAcquire) < _epoch + 1) {
        Intrin::spinBackoff(iter++);
      }
    }
    _epoch += 1;
    coop.sync();
  }

  // sync = arrive + wait
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel) {
    arrive(coop, order);
    wait(coop, order);
  }
};

// ---- DevBarrier<Fallback<P>, sdcclBarrierInter> ----
// Inter-node barrier via FIFO BarrierSignal + host-mapped interSignalFlags.
// Only the inter leader actually sends/waits; non-leaders are no-ops.
template <typename P>
struct DevBarrier<Fallback<P>, sdcclBarrierInter> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using DevComm = typename DeviceTraits<Fallback<P>>::DevComm;
  using Team = typename DeviceTraits<Fallback<P>>::Team;
  using Net = typename DeviceTraits<Fallback<P>>::Net;

  uint64_t *_interSignals;
  void *_fifoBuffer;
  int _nInterPeers;
  bool _isLeader;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Default ctor (no-op)
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier()
      : _interSignals(nullptr), _fifoBuffer(nullptr), _nInterPeers(0),
        _isLeader(false), _ctaIndex(0), _epoch(0) {}

  // Active ctor
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier(const Net &net, const DevComm &dc, Team, uint32_t index,
             int nInterPeers)
      : _interSignals(dc.interSignalFlags), _fifoBuffer(dc.fifoBuffer),
        _nInterPeers(nInterPeers), _isLeader(dc.isInterLeader),
        _ctaIndex(index), _epoch(dc.interBarrierEpoch) {}

  // arrive: FIFO BarrierSignal (leader only)
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
         sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    _epoch += _nInterPeers;
    coop.sync();
    if (coop.threadRank() == 0 && _isLeader) {
      DeviceTraits<Fallback<P>>::fifoEnqueue(
          _fifoBuffer, (uint64_t)_ctaIndex, 0,
          DeviceTraits<Fallback<P>>::buildTrd(sdcclDevicePrimBarrierSignal, 0,
                                              0));
    }
    coop.sync();
  }

  // wait: spin on host-mapped inter signal array (leader only)
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    coop.sync();
    if (coop.threadRank() == 0 && _isLeader) {
      int iter = 0;
      while (Atomic::load(&_interSignals[_ctaIndex],
                          sdcclDeviceMemoryOrderAcquire) < _epoch) {
        Intrin::spinBackoff(iter++);
      }
    }
    coop.sync();
  }

  // sync = arrive + wait
  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    arrive(coop, order);
    wait(coop, order);
  }
};

// ---- DevBarrier<Fallback<P>, sdcclBarrierWorld> ----
// Composes intra + inter barriers.
// Three-phase pattern for multi-node: intra → inter → intra.
// Single-node: just one intra sync.
template <typename P>
struct DevBarrier<Fallback<P>, sdcclBarrierWorld> {
  using DevComm = typename DeviceTraits<Fallback<P>>::DevComm;
  using Team = typename DeviceTraits<Fallback<P>>::Team;
  using Net = typename DeviceTraits<Fallback<P>>::Net;

  DevBarrier<Fallback<P>, sdcclBarrierIntra> _intra;
  DevBarrier<Fallback<P>, sdcclBarrierInter> _inter;
  int _nInterPeers;

  // World barrier: intra (IPC) + inter (FIFO Signal)
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier(sdcclBarrierWorld::World, const Net &net, const DevComm &dc,
             uint32_t index, bool multimem, int nInterPeers)
      : _intra(dc, Team{dc.intraSize, dc.intraRank, 1}, index),
        _inter(net, dc, Team{}, index, nInterPeers), _nInterPeers(nInterPeers) {
  }

  // Intra-only barrier: inter is default constructed (no-op)
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier(sdcclBarrierWorld::Intra, const Net &, const DevComm &dc,
             uint32_t index, bool multimem, int)
      : _intra(dc, Team{dc.intraSize, dc.intraRank, 1}, index), _inter(),
        _nInterPeers(0) {}

  // Inter-only barrier: intra is default constructed (no-op)
  SDCCL_DEVICE_INLINE_DECORATOR
  DevBarrier(sdcclBarrierWorld::Inter, const Net &net, const DevComm &dc,
             uint32_t index, bool, int nInterPeers)
      : _intra(), _inter(net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
         sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _intra.arrive(coop, sdcclDeviceMemoryOrderRelease);
      _intra.wait(coop, sdcclDeviceMemoryOrderRelease);
      _inter.arrive(coop, order);
    } else {
      _intra.arrive(coop, order);
    }
  }

  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _inter.wait(coop, order);
      _intra.arrive(coop, sdcclDeviceMemoryOrderAcquire);
      _intra.wait(coop, sdcclDeviceMemoryOrderAcquire);
    } else {
      _intra.wait(coop, order);
    }
  }

  template <typename Coop>
  SDCCL_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       sdcclDeviceMemoryOrder_t order = sdcclDeviceMemoryOrderAcqRel,
       sdcclGinFenceLevel fence = sdcclGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      // Phase 1: intra sync
      _intra.arrive(coop, sdcclDeviceMemoryOrderRelease);
      _intra.wait(coop, sdcclDeviceMemoryOrderRelease);
      // Phase 2: inter signal+wait (leader only)
      _inter.arrive(coop, order);
      _inter.wait(coop, order);
      // Phase 3: intra sync (broadcast inter completion)
      _intra.arrive(coop, sdcclDeviceMemoryOrderAcquire);
      _intra.wait(coop, sdcclDeviceMemoryOrderAcquire);
    } else {
      // Single-node: one intra sync
      _intra.arrive(coop, order);
      _intra.wait(coop, order);
    }
  }
};

#endif // SDCCL_FALLBACK_DEVICE_TRAITS_H_
