#include "sdccl.h"
#include "sdccl_kernel.h"

SDCCL_PARAM(ReduceFifoCapacity, "REDUCE_FIFO_CAPACITY", SDCCL_FIFO_CAPACITY);

SDCCL_HOST_DECORATOR void
sdcclReduceTrigger::setValue(uint64_t fst, uint64_t snd, uint64_t out,
                              size_t count, size_t nthreads,
                              sdcclDataType_t datatype, sdcclRedOp_t redOp,
                              sdcclReduceTriggerState state) {
  uint64_t tmp[4];
  tmp[0] = fst;
  tmp[1] = snd;
  tmp[2] = out;
  tmp[3] = (count & sdcclTriggerMask(sdcclReduceTriggerBitsCount))
               << sdcclReduceTriggerOffCount |
           (nthreads & sdcclTriggerMask(sdcclReduceTriggerBitsNThreads))
               << sdcclReduceTriggerOffNThreads |
           (datatype & sdcclTriggerMask(sdcclReduceTriggerBitsDatatype))
               << sdcclReduceTriggerOffDatatype |
           (redOp & sdcclTriggerMask(sdcclReduceTriggerBitsRedop))
               << sdcclReduceTriggerOffRedop |
           (state & sdcclTriggerMask(sdcclReduceTriggerBitsState))
               << sdcclReduceTriggerOffState;
  memcpy(this->value, tmp, 4 * sizeof(uint64_t));
}

SDCCL_HOST_DECORATOR uint64_t sdcclReduceTrigger::pollState() {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  return currVal >> sdcclReduceTriggerOffState &
         sdcclTriggerMask(sdcclReduceTriggerBitsState);
}

SDCCL_HOST_DECORATOR void sdcclReduceTrigger::setState(int state) {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  currVal &= ~(sdcclTriggerMask(sdcclReduceTriggerBitsState)
               << sdcclReduceTriggerOffState);
  currVal |= (state & sdcclTriggerMask(sdcclReduceTriggerBitsState))
             << sdcclReduceTriggerOffState;
  __atomic_store_n(&this->value[3], currVal, __ATOMIC_RELEASE);
  TRACE(SDCCL_KERNEL, "setState called, new state=%llu",
        currVal >> sdcclReduceTriggerOffState &
            sdcclTriggerMask(sdcclReduceTriggerBitsState));
}

SDCCL_HOST_DECORATOR sdcclResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             size_t count, size_t nthreads,
                                             sdcclDataType_t datatype,
                                             sdcclRedOp_t redop, int *ret) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[sdcclFifoIdxCapacity];
  int distance = buffer[sdcclFifoIdxProduced] - buffer[sdcclFifoIdxConsumed];
  // red buffer full, wait for kernel to consume
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return sdcclSuccess;
  }
  idx = buffer[sdcclFifoIdxProduced] % capacity;
  sdcclReduceTrigger *trigger =
      ((sdcclReduceTrigger *)(buffer + sdcclFifoIdxData)) + idx;

  // kernel reduce work in progress
  if (trigger->pollState() != sdcclReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return sdcclSuccess;
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop,
                    sdcclReduceTriggerEnqueued);
  __atomic_fetch_add(buffer + sdcclFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(SDCCL_KERNEL,
        "enqueue red: count=%lu, nthreads=%lu, datatype=%d, redop=%d, idx=%d",
        count, nthreads, datatype, redop, idx);

  return sdcclSuccess;
}

sdcclResult_t sdcclFifo::sdcclRedFifoInit() {
  TRACE(SDCCL_INIT, "sdcclRedFifoInit called");
  uint64_t sdcclReduceFifoCapacity = sdcclParamReduceFifoCapacity();
  SDCCLCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          sdcclFifoIdxData * sizeof(uint64_t) +
                                              sdcclReduceFifoCapacity *
                                                  sizeof(sdcclReduceTrigger),
                                          sdcclMemHost, NULL));
  buffer[sdcclFifoIdxCapacity] = sdcclReduceFifoCapacity;
  buffer[sdcclFifoIdxConsumed] = 0;
  buffer[sdcclFifoIdxProduced] = 0;
  buffer[sdcclFifoIdxTerminate] = 0;
  memset((void *)(buffer + sdcclFifoIdxData), 0,
         sdcclReduceFifoCapacity * sizeof(sdcclReduceTrigger));
  __sync_synchronize();
  return sdcclSuccess;
}

sdcclResult_t sdcclFifo::sdcclRedFifoDestroy() {
  INFO(SDCCL_KERNEL, "sdcclRedFifoDestroy called");
  SDCCLCHECK(deviceAdaptor->deviceFree((void *)buffer, sdcclMemHost, NULL));
  return sdcclSuccess;
}
