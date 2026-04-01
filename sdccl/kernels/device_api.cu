/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * SDCCL Device API kernels.
 *
 * 1. Intra-node AllReduce — peer pointer + barrier based.
 *    Vendor (>= 2.28): wraps vendor DevComm + vendor Window + vendor barrier.
 *    Fallback:    IPC peer pointers + atomics barrier.
 *    Same kernel code compiles for both paths.
 *
 * 2. Inter-node AlltoAll — two separate kernels:
 *    a) One-sided (put): thread-stride loop, put + waitSignal + flush.
 *    b) Two-sided (send/recv): thread-0 block-stride loop, FIFO + term/wait.
 *    Both paths wrapped by bar.sync() pre/post barriers.
 *
 * Host-side sdcclDevCommCreate/Destroy are in sdccl_device.cc.
 ************************************************************************/

#include "device_api/sdccl_device.h"
#include "nvidia_adaptor.h"
#include "global_comm.h"
#include "sdccl_kernel.h"
#include <cuda_runtime.h>

// Helper: advance intra-barrier epoch by nBarSyncs, accounting for topology.
// Each bar.sync does syncsPerBarrier intra syncs (1 single-node, 2 multi-node).
static inline void advanceIntraEpoch(sdcclDevComm_t devComm, int nBarSyncs) {
  int syncsPerBarrier = (devComm->nInterPeers > 0) ? 2 : 1;
  devComm->intraBarrierEpoch += nBarSyncs * syncsPerBarrier;
}

// ==========================================================================
// 1. Intra-node AllReduce
// ==========================================================================

// Intra-node AllReduce: each block reads from all peers via team-based
// sdcclGetPeerPointer, reduces (sum), and writes result back to all peers.
template <typename T>
__global__ void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclIntraAllReduceKernel(sdcclDevComm devComm, sdcclDevMem mem,
                               size_t offset, size_t count) {
  // AllReduce requires peer pointer access (window or IPC)
  if (!mem.hasWindow()) {
    if (SDCCL_THREAD_IDX_X == 0 && SDCCL_BLOCK_IDX_X == 0) {
      printf("sdcclIntraAllReduceKernel: no peer access (no window, no IPC), "
             "skipping\n");
    }
    return;
  }

  sdcclTeam_t intra = sdcclTeamIntra(devComm);

  // Create barrier session using simplified SDCCL API (4 params).
  sdcclDevBarrier<sdcclBarrierIntra, sdcclCoopBlock> bar{
      sdcclCoopBlock(), devComm, intra, SDCCL_BLOCK_IDX_X};

  // Pre-reduce barrier (acquire — ensure peer writes are visible)
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderAcquire);

  const int rank = devComm.getIntraRank();
  const int nRanks = devComm.getIntraSize();
  const int globalTid =
      SDCCL_THREAD_IDX_X + SDCCL_BLOCK_DIM_X * (rank + SDCCL_BLOCK_IDX_X * nRanks);
  const int globalNthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X * nRanks;

  // Phase 1: Reduce — sum data from all intra-node peers
  // Phase 2: Write — store result to all intra-node peers
  for (size_t o = globalTid; o < count; o += globalNthreads) {
    T v = T(0);
    for (int peer = 0; peer < nRanks; peer++) {
      T* inputPtr = (T*)sdcclGetPeerPointer(mem, offset, intra, peer);
      v += inputPtr[o];
    }
    for (int peer = 0; peer < nRanks; peer++) {
      T* outputPtr = (T*)sdcclGetPeerPointer(mem, offset, intra, peer);
      outputPtr[o] = v;
    }
  }

  // Post-reduce barrier (release ordering — ensure writes are visible)
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelease);
}

// Host-side launcher
template <typename T>
static cudaError_t launchSdcclIntraAllReduce(sdcclDevComm devComm,
                                              sdcclDevMem mem,
                                              size_t offset, size_t count,
                                              cudaStream_t stream) {
  sdcclIntraAllReduceKernel<T>
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         stream>>>(devComm, mem, offset, count);
  return cudaGetLastError();
}

// Explicit instantiations for common types
template cudaError_t launchSdcclIntraAllReduce<float>(sdcclDevComm,
                                                       sdcclDevMem, size_t,
                                                       size_t, cudaStream_t);
template cudaError_t launchSdcclIntraAllReduce<double>(sdcclDevComm,
                                                        sdcclDevMem, size_t,
                                                        size_t, cudaStream_t);

// Host-side function — launches the kernel using caller-provided
// registered buffer and device communicator.
sdcclResult_t sdcclIntraAllReduce(sdcclDevMem_t devMem, size_t count,
                                        sdcclDataType_t datatype,
                                        sdcclDevComm_t devComm,
                                        sdcclStream_t stream) {
  if (devComm == nullptr || devMem == nullptr) {
    return sdcclInternalError;
  }

  cudaStream_t cudaStream = *(cudaStream_t *)stream;

  // Unified constructors — work for both Vendor and Fallback
  sdcclDevComm devCommKernel(*devComm);
  sdcclDevMem devMemKernel(*devMem);

  cudaError_t err;
  switch (datatype) {
  case sdcclFloat32:
    err = launchSdcclIntraAllReduce<float>(devCommKernel, devMemKernel, 0,
                                            count, cudaStream);
    break;
  case sdcclFloat64:
    err = launchSdcclIntraAllReduce<double>(devCommKernel, devMemKernel, 0,
                                             count, cudaStream);
    break;
  default:
    return sdcclInvalidArgument;
  }

  // Advance barrier epoch for next launch (2 syncs, each epoch += 1)
  devComm->intraBarrierEpoch += 2;

  return (err == cudaSuccess) ? sdcclSuccess : sdcclUnhandledDeviceError;
}

// ==========================================================================
// Inter-node One-sided AlltoAll
//
// Thread-stride loop: each thread dispatches put ops to different peers.
// put() posts FIFO descriptor (Fallback) or one-sided descriptor (Vendor).
// After all puts, waitSignal + flush ensure completion.
//
// Buffer layout: [rank0_data][rank1_data]...[rankN_data], each of size `count`
// sendMem: data at offset peerRank * count * elementSize is sent to peerRank
// recvMem: data from peerRank is stored at offset peerRank * count * elementSize
// ==========================================================================

SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterOneSidedAlltoAllKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclDevComm devComm) {

  // contextIndex=0: all CTAs share signal slot 0. readSignal is taken before
  // bar.sync so the baseline is captured before any signals from this round arrive.
  sdcclDevNet net(devComm, 0);
  // Unified barrier: intra (IPC) + inter (FIFO signal relay).
  // Single-node: intra sync only.  Multi-node: three-phase intra/inter/intra.
  sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
      sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);

  // Read signal baseline before pre-barrier so it reflects the pre-round state.
  uint64_t signalValue = net.readSignal(0);

  // Pre-communication barrier
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

  int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
  int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
  for (int peer = tid; peer < nRanks; peer += nthreads) {
    net.put(sdcclTeamWorld(devComm), peer, recvMem, myRank * size,
            sendMem, peer * size, size, sdcclDevNet_SignalInc{0},
            sdcclDevNet_None{}, sdcclCoopThread{});
  }

  net.waitSignal(sdcclCoopBlock{}, 0, signalValue + nRanks);
  net.flush(sdcclCoopBlock{});

  // Post-communication barrier
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
}

// ==========================================================================
// Inter-node Two-sided AlltoAll
//
// Thread-0 block-stride loop dispatches send/recv via FIFO.
// term() + wait() for group semantic completion.
//
// Buffer layout: same as one-sided.
// ==========================================================================

SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTwoSidedAlltoAllKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                      size_t count, sdcclDataType_t datatype,
                                      sdcclDevComm devComm) {

  sdcclDevNet net(devComm, SDCCL_BLOCK_IDX_X);
  // Unified barrier: intra (IPC) + inter (FIFO signal relay).
  // Single-node: intra sync.  Multi-node: three-phase intra/inter/intra.
  sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
      sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);

  int nRanks = devComm.getSize();
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);

  // Pre-communication barrier
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

  // All operations are Coop-scope: every thread participates,
  // only threadRank==0 touches FIFO.
  for (int peer = SDCCL_BLOCK_IDX_X; peer < nRanks;
       peer += SDCCL_GRID_DIM_X) {
    size_t offset = peer * size;
    net.send(sdcclCoopBlock{}, sendMem, offset, count, datatype, peer);
    net.recv(sdcclCoopBlock{}, recvMem, offset, count, datatype, peer);
  }

  net.term(sdcclCoopBlock{});
  net.wait(sdcclCoopBlock{});

  // Post-communication barrier
  bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
}

// Host-side one-sided AlltoAll function.
sdcclResult_t sdcclInterOneSidedAlltoAll(sdcclDevMem_t sendMem,
                                           sdcclDevMem_t recvMem, size_t count,
                                           sdcclDataType_t datatype,
                                           sdcclDevComm_t devComm,
                                           sdcclStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return sdcclInternalError;
  }

  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);

  sdcclInterOneSidedAlltoAllKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance barrier epochs: 2 bar.syncs per kernel.
  // sdcclDevBarrier<sdcclBarrierWorld>::sync does (nInterPeers>0): 2 intra + 1 inter,
  //                              or (nInterPeers==0): 1 intra.
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? sdcclSuccess : sdcclUnhandledDeviceError;
}

// Host-side two-sided AlltoAll function.
sdcclResult_t sdcclInterTwoSidedAlltoAll(sdcclDevMem_t sendMem,
                                            sdcclDevMem_t recvMem, size_t count,
                                            sdcclDataType_t datatype,
                                            sdcclDevComm_t devComm,
                                            sdcclStream_t stream) {
  if (devComm == nullptr || sendMem == nullptr || recvMem == nullptr) {
    return sdcclInternalError;
  }

  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);

  sdcclInterTwoSidedAlltoAllKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);

  cudaError_t err = cudaGetLastError();

  // Advance barrier epochs (2 bar.syncs per kernel)
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;

  return (err == cudaSuccess) ? sdcclSuccess : sdcclUnhandledDeviceError;
}


// ==========================================================================
// Inter-node One-sided Device API Tests (K2-K9)
//
// Eight focused kernels, each covering one Device API facet.
// Signal/counter slot assignments:
//   slot 0: SignalAdd, CounterPipeline, FlushDecouple
//   slot 1: PutValue, SignalOnly
//   slot 2: FollowShadow, MeetShadow
//   counter 0: CounterInc
// DevCommRequirements: interSignalCount=3, interCounterCount=1
// All kernels follow: pre-bar.sync / core logic / post-bar.sync.
// ==========================================================================

// Shared IPC alltoall helper: each thread copies its portion of all intra peers.
// NOTE: assumes chunkSize is a multiple of 4 bytes (i.e., element type is
// >= 4-byte aligned -- float, int32, etc.). Sub-4-byte types (half, int8)
// with odd counts will lose tail bytes.
SDCCL_DEVICE_INLINE_DECORATOR void
ipcAlltoAll(const sdcclDevMem &sendMem, const sdcclDevMem &recvMem,
            sdcclTeam_t intra, int intraSize, int intraBase,
            int myWorldRank, size_t chunkSize) {
  int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
  int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
  size_t nWords = chunkSize / sizeof(uint32_t);
  for (int lr = 0; lr < intraSize; lr++) {
    int worldPeer = intraBase + lr;
    uint32_t *src = (uint32_t *)sdcclGetLocalPointer(
        sendMem, (size_t)worldPeer * chunkSize);
    uint32_t *dst = (uint32_t *)sdcclGetPeerPointer(
        recvMem, (size_t)myWorldRank * chunkSize, intra, lr);
    for (size_t w = (size_t)tid; w < nWords; w += (size_t)nthreads)
      dst[w] = src[w];
  }
}

// put + SignalInc
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestPutSignalIncKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    // Hybrid: DevNet for inter + IPC for intra
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(sdcclTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              sdcclDevNet_SignalInc{0}, sdcclDevNet_None{},
              sdcclCoopThread{});
    }
    net.waitSignal(sdcclCoopBlock{}, 0, s0 + (uint64_t)nInterRanks);
    net.flush(sdcclCoopBlock{});
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    // Single-node: IPC only, no DevNet
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// put data + separate SignalAdd: decouples data transfer from signalling.
// One-sided path: two NIC ops (WRITE then ATOMIC+2 on slot 0).
// Fallback path: PrimPut + PrimSignal(value=2) (two FIFO entries, slot 0).
// Contrast with K1 where both paths fuse into a single chained WR.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestPutSignalAddDecoupledKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                   size_t count, sdcclDataType_t datatype,
                                   sdcclDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(sdcclTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              sdcclDevNet_None{}, sdcclDevNet_None{},
              sdcclCoopThread{});
      net.signal(sdcclTeamWorld(devComm), peer,
                 sdcclDevNet_SignalAdd{0, 2}, sdcclCoopThread{});
    }
    net.waitSignal(sdcclCoopBlock{}, 0, s0 + (uint64_t)nInterRanks * 2);
    net.flush(sdcclCoopBlock{});
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// put + CounterInc two-round pipeline
// Round 1: put with CounterInc; waitCounter; stamp sentinel; Round 2: put again.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestCounterPipelineKernel(sdcclDevMem sendMem,
                                         sdcclDevMem recvMem, size_t count,
                                         sdcclDataType_t datatype,
                                         sdcclDevComm devComm,
                                         uint64_t *resultBuf) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);
  int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
  int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
  float *sendRaw = (float *)sendMem.getRawPtr();

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    uint64_t c0 = net.readCounter(0);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // Round 1: IPC for intra + DevNet for inter
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(sdcclTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              sdcclDevNet_SignalInc{0}, sdcclDevNet_CounterInc{0},
              sdcclCoopThread{});
    }
    net.waitCounter(sdcclCoopBlock{}, 0, c0 + (uint64_t)nInterRanks);

    // Stamp sentinel
    for (int peer = tid; peer < nRanks; peer += nthreads)
      sendRaw[(ptrdiff_t)peer * (ptrdiff_t)count] = 999.0f;
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // Round 2: same
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(sdcclTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              sdcclDevNet_SignalInc{0}, sdcclDevNet_CounterInc{0},
              sdcclCoopThread{});
    }
    net.waitCounter(sdcclCoopBlock{}, 0, c0 + 2 * (uint64_t)nInterRanks);
    net.waitSignal(sdcclCoopBlock{}, 0, s0 + 2 * (uint64_t)nInterRanks);
    net.flush(sdcclCoopBlock{});

    if (SDCCL_BLOCK_IDX_X == 0 && SDCCL_THREAD_IDX_X == 0) {
      resultBuf[0] = net.readCounter(0);
      resultBuf[1] = (uint64_t)nInterRanks;
    }
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // Round 1
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // Stamp sentinel
    for (int peer = tid; peer < nRanks; peer += nthreads)
      sendRaw[(ptrdiff_t)peer * (ptrdiff_t)count] = 999.0f;
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // Round 2
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    if (SDCCL_BLOCK_IDX_X == 0 && SDCCL_THREAD_IDX_X == 0) {
      resultBuf[0] = 0; // no counter in IPC mode
      resultBuf[1] = 0; // nInterRanks = 0
    }
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// putValue
// Each rank writes value=myRank*1000+peer to peer's recvBuff[putValBase+myRank*8].
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestPutValueKernel(sdcclDevMem recvMem, sdcclDevComm devComm,
                                  size_t putValBase) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
  int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      if (peer >= intraBase && peer < intraBase + intraSize) {
        // IPC: direct write to peer's recvBuff
        int lr = peer - intraBase;
        uint64_t *dst = (uint64_t *)sdcclGetPeerPointer(
            recvMem, putValBase + (size_t)myRank * sizeof(uint64_t), intra, lr);
        *dst = val;
      } else {
        net.putValue(sdcclTeamWorld(devComm), peer,
                     recvMem, putValBase + (size_t)myRank * sizeof(uint64_t),
                     val, sdcclDevNet_SignalInc{1}, sdcclCoopThread{});
      }
    }
    if (nInterRanks > 0)
      net.waitSignal(sdcclCoopBlock{}, 1, s1 + (uint64_t)nInterRanks);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      uint64_t val = (uint64_t)myRank * 1000u + (uint64_t)peer;
      int lr = peer - intraBase;
      uint64_t *dst = (uint64_t *)sdcclGetPeerPointer(
          recvMem, putValBase + (size_t)myRank * sizeof(uint64_t), intra, lr);
      *dst = val;
    }
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// signal standalone
// Each rank signals all other peers on slot 1; waits for nRanks-1 incoming.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestSignalKernel(sdcclDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  int nInterRanks = nRanks - intraSize;

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s1 = net.readSignal(1);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads)
      if (peer != myRank && (peer < intraBase || peer >= intraBase + intraSize))
        net.signal(sdcclTeamWorld(devComm), peer,
                   sdcclDevNet_SignalInc{1}, sdcclCoopThread{});
    if (nInterRanks > 0)
      net.waitSignal(sdcclCoopBlock{}, 1, s1 + (uint64_t)nInterRanks);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// put + flush decoupled
// put(None,None) → flush (src drain) → signal → waitSignal → flush (dst).
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestFlushDecoupleKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                       size_t count, sdcclDataType_t datatype,
                                       sdcclDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  int nInterRanks = nRanks - intraSize;
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    uint64_t s0 = net.readSignal(0);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // IPC for intra peers
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // DevNet puts (None,None) for inter peers only
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.put(sdcclTeamWorld(devComm), peer,
              recvMem, (size_t)myRank * size,
              sendMem, (size_t)peer * size, size,
              sdcclDevNet_None{}, sdcclDevNet_None{},
              sdcclCoopThread{});
    }
    net.flush(sdcclCoopBlock{});

    // Signal inter peers only
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      net.signal(sdcclTeamWorld(devComm), peer,
                 sdcclDevNet_SignalInc{0}, sdcclCoopThread{});
    }
    net.waitSignal(sdcclCoopBlock{}, 0, s0 + (uint64_t)nInterRanks);
    net.flush(sdcclCoopBlock{});
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// waitSignalFollowShadow
// All ranks signal all peers on slot 2; FollowShadow advances shadow by nRanks.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestFollowShadowKernel(sdcclDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(sdcclTeamWorld(devComm), peer,
                 sdcclDevNet_SignalInc{2}, sdcclCoopThread{});
    uint64_t before, delta;
    net.waitSignalFollowShadow(sdcclCoopBlock{}, (sdcclDevNetSignal_t)2,
                                (uint64_t)nRanks, &before, &delta);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// increaseSignalShadow + waitSignalMeetShadow
// Block 0 thread 0 advances shadow; all blocks signal peers then waitMeetShadow.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestMeetShadowKernel(sdcclDevComm devComm) {
  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    int nRanks = devComm.getSize();
    int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
    int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    if (SDCCL_BLOCK_IDX_X == 0 && SDCCL_THREAD_IDX_X == 0) {
      net.increaseSignalShadow((sdcclDevNetSignal_t)2, (uint64_t)nRanks);
      __threadfence();
    }
    for (int peer = tid; peer < nRanks; peer += nthreads)
      net.signal(sdcclTeamWorld(devComm), peer,
                 sdcclDevNet_SignalInc{2}, sdcclCoopThread{});
    net.waitSignalMeetShadow(sdcclCoopBlock{}, (sdcclDevNetSignal_t)2);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// resetSignal + resetCounter + 32-bit readSignal
// Resets all used signal/counter slots; records post-reset values in resultBuf.
SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestResetKernel(sdcclDevComm devComm, uint64_t *resultBuf) {
  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    if (SDCCL_BLOCK_IDX_X == 0 && SDCCL_THREAD_IDX_X == 0) {
      net.resetSignal(0);
      net.resetSignal(1);
      net.resetSignal(2);
      net.resetCounter(0);
      *net.getSignalShadowPtr(2) = 0;
      (void)net.readSignal(0, 32);
      resultBuf[0] = net.readSignal(0);
      resultBuf[1] = net.readSignal(1);
      resultBuf[2] = net.readSignal(2);
      resultBuf[3] = net.readCounter(0);
    }
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    if (SDCCL_BLOCK_IDX_X == 0 && SDCCL_THREAD_IDX_X == 0) {
      resultBuf[0] = 0;
      resultBuf[1] = 0;
      resultBuf[2] = 0;
      resultBuf[3] = 0;
    }
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

// --------------------------------------------------------------------------
// Host wrappers
// --------------------------------------------------------------------------

sdcclResult_t sdcclInterTestPutSignalInc(sdcclDevMem_t sendMem,
                                        sdcclDevMem_t recvMem, size_t count,
                                        sdcclDataType_t datatype,
                                        sdcclDevComm_t devComm,
                                        sdcclStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);
  sdcclInterTestPutSignalIncKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestPutSignalAddDecoupled(sdcclDevMem_t sendMem,
                                        sdcclDevMem_t recvMem, size_t count,
                                        sdcclDataType_t datatype,
                                        sdcclDevComm_t devComm,
                                        sdcclStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);
  sdcclInterTestPutSignalAddDecoupledKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestCounterPipeline(sdcclDevMem_t sendMem,
                                              sdcclDevMem_t recvMem,
                                              size_t count,
                                              sdcclDataType_t datatype,
                                              sdcclDevComm_t devComm,
                                              sdcclStream_t stream,
                                              uint64_t *resultBuf) {
  if (!devComm || !sendMem || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);
  sdcclInterTestCounterPipelineKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc, resultBuf);
  cudaError_t err = cudaGetLastError();
  // K3 CounterPipeline: 3 bar.syncs (pre-barrier + post-round1 + post-round2).
  // Each bar.sync does syncsPerBarrier intra syncs (1 single-node, 2 multi-node).
  advanceIntraEpoch(devComm, 3);
  devComm->interBarrierEpoch += 3 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestPutValue(sdcclDevMem_t recvMem,
                                       sdcclDevComm_t devComm,
                                       sdcclStream_t stream,
                                       size_t putValBase) {
  if (!devComm || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem rm(*recvMem);
  sdcclInterTestPutValueKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(rm, dc, putValBase);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestSignal(sdcclDevComm_t devComm,
                                         sdcclStream_t stream) {
  if (!devComm) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclInterTestSignalKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K5 SignalOnly: multi-node = 2 bar.syncs (pre + post), single-node = 1 bar.sync
  // (no inter-peer ops to bracket, barrier-only). Epoch accounting reflects this.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4; // 2 bar.syncs × 2 intra syncs each
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1; // 1 bar.sync × 1 intra sync
  }
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestFlushDecouple(sdcclDevMem_t sendMem,
                                            sdcclDevMem_t recvMem,
                                            size_t count,
                                            sdcclDataType_t datatype,
                                            sdcclDevComm_t devComm,
                                            sdcclStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);
  sdcclInterTestFlushDecoupleKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestFollowShadow(sdcclDevComm_t devComm,
                                           sdcclStream_t stream) {
  if (!devComm) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclInterTestFollowShadowKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K8 FollowShadow: multi-node = 2 bar.syncs, single-node = 1 bar.sync.
  // Asymmetric epoch accounting same as K5.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4;
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1;
  }
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestMeetShadow(sdcclDevComm_t devComm,
                                         sdcclStream_t stream) {
  if (!devComm) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclInterTestMeetShadowKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc);
  cudaError_t err = cudaGetLastError();
  // K9 MeetShadow: multi-node = 2 bar.syncs, single-node = 1 bar.sync.
  // Asymmetric epoch accounting same as K5.
  if (devComm->nInterPeers > 0) {
    devComm->intraBarrierEpoch += 4;
    devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  } else {
    devComm->intraBarrierEpoch += 1;
  }
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

sdcclResult_t sdcclInterTestReset(sdcclDevComm_t devComm,
                                    sdcclStream_t stream,
                                    uint64_t *resultBuf) {
  if (!devComm) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclInterTestResetKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(dc, resultBuf);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}

// ==========================================================================
// K8: get AlltoAll
//
// Each rank RDMA-READs peer's sendBuff[myRank*size..] into local
// recvBuff[peer*size..].  Producer fills sendBuff, consumer pulls via get().
// Synchronized by barrier.  No fused signal — get() has no signal action;
// the post-barrier ensures completion visibility.
// ==========================================================================

SDCCL_GLOBAL_DECORATOR void __launch_bounds__(SDCCL_DEVICE_THREADS_PER_CTA)
    sdcclInterTestGetKernel(sdcclDevMem sendMem, sdcclDevMem recvMem,
                                     size_t count, sdcclDataType_t datatype,
                                     sdcclDevComm devComm) {
  int nRanks = devComm.getSize();
  int myRank = devComm.getRank();
  int intraSize = devComm.getIntraSize();
  int intraBase = myRank - devComm.getIntraRank();
  sdcclTeam_t intra = sdcclTeamIntra(devComm);
  size_t size = count * getSdcclDataTypeSizeDevice(datatype);
  int tid = SDCCL_THREAD_IDX_X + SDCCL_BLOCK_IDX_X * SDCCL_BLOCK_DIM_X;
  int nthreads = SDCCL_BLOCK_DIM_X * SDCCL_GRID_DIM_X;

  if (devComm._nInterPeers > 0) {
    sdcclDevNet net(devComm, 0);
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagWorld{}, net, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);

    // IPC for intra-node peers
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);

    // RDMA READ for inter-node peers: pull from peer's sendMem into my recvMem
    for (int peer = tid; peer < nRanks; peer += nthreads) {
      if (peer >= intraBase && peer < intraBase + intraSize) continue;
      // src: peer's sendBuff at offset myRank*size (peer's data for me)
      // dst: my recvBuff at offset peer*size (my slot for peer's data)
      net.get(sdcclTeamWorld(devComm), peer,
              sendMem, (size_t)myRank * size,
              recvMem, (size_t)peer * size, size,
              sdcclCoopThread{});
    }
    net.flush(sdcclCoopBlock{});
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  } else {
    // Intra-only: use IPC
    sdcclDevBarrier<sdcclBarrierWorld, sdcclCoopBlock> bar(
        sdcclCoopBlock(), sdcclTeamTagIntra{}, devComm, SDCCL_BLOCK_IDX_X);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
    ipcAlltoAll(sendMem, recvMem, intra, intraSize, intraBase, myRank, size);
    bar.sync(sdcclCoopBlock(), sdcclDeviceMemoryOrderRelaxed);
  }
}

sdcclResult_t sdcclInterTestGet(sdcclDevMem_t sendMem,
                                          sdcclDevMem_t recvMem, size_t count,
                                          sdcclDataType_t datatype,
                                          sdcclDevComm_t devComm,
                                          sdcclStream_t stream) {
  if (!devComm || !sendMem || !recvMem) return sdcclInternalError;
  sdcclDevComm dc(*devComm);
  sdcclDevMem sm(*sendMem), rm(*recvMem);
  sdcclInterTestGetKernel
      <<<SDCCL_DEVICE_CTA_COUNT, SDCCL_DEVICE_THREADS_PER_CTA, 0,
         *(cudaStream_t *)stream>>>(sm, rm, count, datatype, dc);
  cudaError_t err = cudaGetLastError();
  advanceIntraEpoch(devComm, 2);
  devComm->interBarrierEpoch += 2 * devComm->nInterPeers;
  return err == cudaSuccess ? sdcclSuccess : sdcclUnhandledDeviceError;
}
