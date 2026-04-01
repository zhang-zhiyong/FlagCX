#include "launch_kernel.h"
#include "group.h"
#include <stdio.h>

sdcclLaunchFunc_t deviceAsyncKernel = NULL;

SDCCL_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

sdcclDeviceSemaphoreBufferPool::sdcclDeviceSemaphoreBufferPool()
    : capacity(-1), slotId(-1), signalsPool(nullptr), dSignalsPool(nullptr),
      events(nullptr) {}

sdcclDeviceSemaphoreBufferPool::~sdcclDeviceSemaphoreBufferPool() {
  free(events);
  dSignalsPool = nullptr;
  if (signalsPool != nullptr) {
    deviceAdaptor->deviceFree((void *)signalsPool, sdcclMemHost, nullptr);
  }
}

int sdcclDeviceSemaphoreBufferPool::getSlotId() {
  assert(capacity != -1);
  if (events[slotId] != nullptr) {
    // wait for the previous event to complete
    while (deviceAdaptor->eventQuery(events[slotId]) != sdcclSuccess) {
      sched_yield();
    }
    events[slotId] = nullptr;
  }
  // set this slot signals to zero
  int offset = SDCCL_SIGNALS_PER_SEMAPHORE * slotId;
  memset(signalsPool + offset, 0, SDCCL_SIGNALS_PER_SEMAPHORE * sizeof(int));
  int ret = slotId;
  // Move to next slot
  slotId = (slotId + 1) % capacity;
  return ret;
}

void sdcclDeviceSemaphoreBufferPool::initialize() {
  capacity = sdcclParamSemaphoreBufferPoolCapacity();
  slotId = 0;
  // Allocate host-pinned memory for all semaphores (3 ints each)
  deviceAdaptor->deviceMalloc((void **)&signalsPool,
                              capacity * SDCCL_SIGNALS_PER_SEMAPHORE *
                                  sizeof(int),
                              sdcclMemHost, nullptr);
  // Get device pointer alias
  deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);
  // Init events to nullptr
  sdcclCalloc(&events, capacity);
  for (int i = 0; i < capacity; i++) {
    events[i] = nullptr;
  }
}

// Set event for a semaphore
void sdcclDeviceSemaphoreBufferPool::setEvent(int id, sdcclEvent_t event) {
  assert(id >= 0 && id < capacity);
  // events[id] should be set to nullptr before
  events[id] = event;
}

// Return pointer to the start of a semaphore’s signals (host/device)
int *sdcclDeviceSemaphoreBufferPool::getHostPtr(int id) {
  assert(id >= 0 && id < capacity);
  return signalsPool + SDCCL_SIGNALS_PER_SEMAPHORE * id;
}
void *sdcclDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  assert(id >= 0 && id < capacity);
  return static_cast<void *>((static_cast<char *>(dSignalsPool) +
                              SDCCL_SIGNALS_PER_SEMAPHORE * id * sizeof(int)));
}

void cpuAsyncKernel(void *args) {
  sdcclHostSemaphore *semaphore = (sdcclHostSemaphore *)args;
  semaphore->signalStart();
  semaphore->wait();
}