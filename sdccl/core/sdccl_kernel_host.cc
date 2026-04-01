#include "comm.h"
#include "sdccl.h"
#include "sdccl_kernel.h"

SDCCL_PARAM(KernelFifoCapacity, "KERNEL_FIFO_CAPACITY", SDCCL_FIFO_CAPACITY);

// ==========================================================================
// sdcclDeviceTrigger accessors — read from trd (common) / fst,snd (payload)
// ==========================================================================

// Common accessors (trd)
SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getPrim() {
  return (trd >> sdcclDeviceTriggerOffPrim) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsPrim);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getPeerRank() {
  return (trd >> sdcclDeviceTriggerOffPeerRank) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsPeerRank);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSlotIdx() {
  return (trd >> sdcclDeviceTriggerOffSlotIdx) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSlotIdx);
}

// Backward compat alias
SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getType() {
  return getPrim();
}

// Two-sided accessors (Send/Recv)
SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getAddr() { return fst; }

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getDatatype() {
  return (trd >> sdcclDeviceTriggerOffDatatype) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsDatatype);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getCount() {
  return (trd >> sdcclDeviceTriggerOffCount) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsCount);
}

// One-sided accessors (Put/PutSignal/PutValue)
SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSrcMrIdx() {
  return (trd >> sdcclDeviceTriggerOffSrcMrIdx) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSrcMrIdx);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getDstMrIdx() {
  return (trd >> sdcclDeviceTriggerOffDstMrIdx) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsDstMrIdx);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSize() {
  return (snd >> sdcclDeviceTriggerOffSize) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSize);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSrcOffset() {
  return (fst >> sdcclDeviceTriggerOffSrcOffset) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSrcOffset);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getDstOffset() {
  return (fst >> sdcclDeviceTriggerOffDstOffset) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsDstOffset);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getValue() { return snd; }

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSignalIdx() {
  // PutSignal uses trd[21:14], Signal/WaitSignal uses trd[33:26]
  uint64_t prim = getPrim();
  if (prim == sdcclDevicePrimPutSignal) {
    return (trd >> sdcclDeviceTriggerOffSignalIdx) &
           sdcclTriggerMask(sdcclDeviceTriggerBitsSignalIdx);
  }
  // Signal, WaitSignal
  return (trd >> sdcclDeviceTriggerOffSignalIdxSig) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSignalIdxSig);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getSignalValue() {
  // PutSignal stores signalValue in snd[15:0], Signal stores in trd[25:10]
  uint64_t prim = getPrim();
  if (prim == sdcclDevicePrimPutSignal) {
    return (snd >> sdcclDeviceTriggerOffSignalValuePut) &
           sdcclTriggerMask(sdcclDeviceTriggerBitsSignalValuePut);
  }
  return (trd >> sdcclDeviceTriggerOffSignalValue) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSignalValue);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getExpectedValue() {
  return (trd >> sdcclDeviceTriggerOffSignalValue) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsSignalValue);
}

SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getBufferType() {
  return (trd >> sdcclDeviceTriggerOffBufferType) &
         sdcclTriggerMask(sdcclDeviceTriggerBitsBufferType);
}

// Term accessor
SDCCL_HOST_DECORATOR uint64_t sdcclDeviceTrigger::getTotalCoops() {
  return fst;
}

// ==========================================================================
// FIFO init / destroy
// ==========================================================================

sdcclResult_t sdcclFifo::sdcclFifoInit() {
  INFO(SDCCL_KERNEL, "sdcclFifoInit called");
  uint64_t sdcclKernelFifoCapacity = sdcclParamKernelFifoCapacity();
  SDCCLCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          sdcclFifoIdxData * sizeof(uint64_t) +
                                              sdcclKernelFifoCapacity *
                                                  sizeof(sdcclDeviceTrigger),
                                          sdcclMemHost, NULL));
  buffer[sdcclFifoIdxCapacity] = sdcclKernelFifoCapacity;
  buffer[sdcclFifoIdxConsumed] = 0;
  buffer[sdcclFifoIdxProduced] = 0;
  buffer[sdcclFifoIdxTerminate] =
      0; // reserved, unused for sdcclDeviceTrigger fifo
  memset((void *)(buffer + sdcclFifoIdxData), 0,
         sdcclKernelFifoCapacity * sizeof(sdcclDeviceTrigger));
  return sdcclSuccess;
}

sdcclResult_t sdcclFifo::sdcclFifoDestroy() {
  INFO(SDCCL_KERNEL, "sdcclFifoDestroy called");
  SDCCLCHECK(deviceAdaptor->deviceFree((void *)buffer, sdcclMemHost, NULL));
  return sdcclSuccess;
}

// ==========================================================================
// Host-side dequeue — polls trd (word2) for valid bit
// ==========================================================================

SDCCL_HOST_DECORATOR sdcclResult_t dequeue(void *fifoBuffer,
                                             sdcclDeviceTrigger_t trigger) {
  volatile uint64_t *buffer = (volatile uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[sdcclFifoIdxCapacity];
  uint64_t cons = buffer[sdcclFifoIdxConsumed];
  uint64_t prod = buffer[sdcclFifoIdxProduced];

  if (prod > cons) {
    // Get pointer to slot's raw uint64_t fields (3 words per entry)
    uint64_t idx = cons % capacity;
    volatile uint64_t *slotFst =
        buffer + sdcclFifoIdxData +
        idx * (sizeof(sdcclDeviceTrigger) / sizeof(uint64_t));
    volatile uint64_t *slotSnd = slotFst + 1;
    volatile uint64_t *slotTrd = slotFst + 2;

    // Wait for valid bit on trd (word2, written last by producer)
    int spins = 0;
    while (!(*slotTrd & sdcclDeviceTriggerValidMask)) {
      __sync_synchronize();
      if (++spins > 1000) {
        sched_yield();
        spins = 0;
      }
    }

    // Memory fence before reading payload
    __sync_synchronize();

    // Copy data (clear valid bit in the copy)
    trigger->fst = *slotFst;
    trigger->snd = *slotSnd;
    trigger->trd = *slotTrd & ~sdcclDeviceTriggerValidMask;

    // Clear trd valid bit in slot for reuse
    *slotTrd = 0;
  } else {
    memset((void *)trigger, 0, sizeof(sdcclDeviceTrigger));
  }
  return sdcclSuccess;
}
