/*************************************************************************
 * Copyright (c) 2025. All Rights Reserved.
 * Single device adaptor test - no multi-GPU or MPI required
 ************************************************************************/

#include <cstring>
#include <gtest/gtest.h>
#include <iostream>

#include "adaptor.h"
#include "sdccl.h"

class DeviceAdaptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize sdccl handle
    sdcclHandleInit(&handler);
    devHandle = handler->devHandle;

    // Get device count and set device 0
    int numDevices = 0;
    devHandle->getDeviceCount(&numDevices);
    ASSERT_GT(numDevices, 0) << "No devices found!";

    std::cout << "Found " << numDevices << " device(s)" << std::endl;

    devHandle->setDevice(0);

    // Create stream
    devHandle->streamCreate(&stream);
  }

  void TearDown() override {
    if (stream) {
      devHandle->streamDestroy(stream);
    }
    sdcclHandleFree(handler);
  }

  sdcclHandlerGroup_t handler = nullptr;
  sdcclDeviceHandle_t devHandle = nullptr;
  sdcclStream_t stream = nullptr;

  static constexpr size_t TEST_SIZE = 1024 * sizeof(float); // 1K floats
  static constexpr size_t TEST_COUNT = 1024;
};

// Test: Get device count and properties
TEST_F(DeviceAdaptorTest, GetDeviceInfo) {
  int numDevices = 0;
  auto result = devHandle->getDeviceCount(&numDevices);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_GT(numDevices, 0);
  std::cout << "Device count: " << numDevices << std::endl;

  int currentDevice = -1;
  result = devHandle->getDevice(&currentDevice);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_EQ(currentDevice, 0);
  std::cout << "Current device: " << currentDevice << std::endl;

  // Get vendor name
  char vendor[128] = {0};
  result = devHandle->getVendor(vendor);
  EXPECT_EQ(result, sdcclSuccess);
  std::cout << "Vendor: " << vendor << std::endl;
}

// Test: Device memory allocation and free
TEST_F(DeviceAdaptorTest, DeviceMemoryAlloc) {
  void *devPtr = nullptr;

  // Allocate device memory
  auto result =
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, sdcclMemDevice, stream);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_NE(devPtr, nullptr);

  // Free device memory
  result = devHandle->deviceFree(devPtr, sdcclMemDevice, stream);
  EXPECT_EQ(result, sdcclSuccess);
}

// Test: Host memory allocation and free
TEST_F(DeviceAdaptorTest, HostMemoryAlloc) {
  void *hostPtr = nullptr;

  // Allocate host memory
  auto result =
      devHandle->deviceMalloc(&hostPtr, TEST_SIZE, sdcclMemHost, stream);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_NE(hostPtr, nullptr);

  // Free host memory
  result = devHandle->deviceFree(hostPtr, sdcclMemHost, stream);
  EXPECT_EQ(result, sdcclSuccess);
}

// Test: Memory copy Host -> Device -> Host
TEST_F(DeviceAdaptorTest, MemoryCopy) {
  void *hostSrc = nullptr;
  void *hostDst = nullptr;
  void *devPtr = nullptr;

  // Allocate memory
  ASSERT_EQ(devHandle->deviceMalloc(&hostSrc, TEST_SIZE, sdcclMemHost, stream),
            sdcclSuccess);
  ASSERT_EQ(devHandle->deviceMalloc(&hostDst, TEST_SIZE, sdcclMemHost, stream),
            sdcclSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, sdcclMemDevice, stream),
      sdcclSuccess);

  // Initialize source data
  float *srcData = static_cast<float *>(hostSrc);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    srcData[i] = static_cast<float>(i);
  }

  // Clear destination
  memset(hostDst, 0, TEST_SIZE);

  // Copy: Host -> Device
  auto result = devHandle->deviceMemcpy(devPtr, hostSrc, TEST_SIZE,
                                        sdcclMemcpyHostToDevice, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Copy: Device -> Host
  result = devHandle->deviceMemcpy(hostDst, devPtr, TEST_SIZE,
                                   sdcclMemcpyDeviceToHost, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Synchronize
  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Verify data
  float *dstData = static_cast<float *>(hostDst);
  for (size_t i = 0; i < TEST_COUNT; i++) {
    EXPECT_FLOAT_EQ(dstData[i], static_cast<float>(i))
        << "Mismatch at index " << i;
  }

  // Cleanup
  devHandle->deviceFree(hostSrc, sdcclMemHost, stream);
  devHandle->deviceFree(hostDst, sdcclMemHost, stream);
  devHandle->deviceFree(devPtr, sdcclMemDevice, stream);
}

// Test: Memory set
TEST_F(DeviceAdaptorTest, MemorySet) {
  void *hostPtr = nullptr;
  void *devPtr = nullptr;

  // Allocate memory
  ASSERT_EQ(devHandle->deviceMalloc(&hostPtr, TEST_SIZE, sdcclMemHost, stream),
            sdcclSuccess);
  ASSERT_EQ(
      devHandle->deviceMalloc(&devPtr, TEST_SIZE, sdcclMemDevice, stream),
      sdcclSuccess);

  // Set device memory to 0
  auto result =
      devHandle->deviceMemset(devPtr, 0, TEST_SIZE, sdcclMemDevice, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Copy back and verify
  result = devHandle->deviceMemcpy(hostPtr, devPtr, TEST_SIZE,
                                   sdcclMemcpyDeviceToHost, stream);
  EXPECT_EQ(result, sdcclSuccess);

  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Verify all zeros
  unsigned char *data = static_cast<unsigned char *>(hostPtr);
  for (size_t i = 0; i < TEST_SIZE; i++) {
    EXPECT_EQ(data[i], 0) << "Non-zero at index " << i;
  }

  // Cleanup
  devHandle->deviceFree(hostPtr, sdcclMemHost, stream);
  devHandle->deviceFree(devPtr, sdcclMemDevice, stream);
}

// Test: Stream operations
TEST_F(DeviceAdaptorTest, StreamOperations) {
  sdcclStream_t newStream = nullptr;

  // Create stream
  auto result = devHandle->streamCreate(&newStream);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_NE(newStream, nullptr);

  // Query stream - result depends on implementation
  // Some backends may return sdcclSuccess, sdcclInProgress, or other values
  result = devHandle->streamQuery(newStream);
  std::cout << "streamQuery result: " << result << std::endl;

  // Synchronize stream (this should always work)
  result = devHandle->streamSynchronize(newStream);
  EXPECT_EQ(result, sdcclSuccess);

  // Destroy stream
  result = devHandle->streamDestroy(newStream);
  EXPECT_EQ(result, sdcclSuccess);
}

// Test: Event operations
TEST_F(DeviceAdaptorTest, EventOperations) {
  sdcclEvent_t event = nullptr;

  // Create event
  auto result = devHandle->eventCreate(&event, sdcclEventDefault);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_NE(event, nullptr);

  // Record event
  result = devHandle->eventRecord(event, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Synchronize event
  result = devHandle->eventSynchronize(event);
  EXPECT_EQ(result, sdcclSuccess);

  // Query event (should be completed after sync)
  result = devHandle->eventQuery(event);
  EXPECT_EQ(result, sdcclSuccess);

  // Destroy event
  result = devHandle->eventDestroy(event);
  EXPECT_EQ(result, sdcclSuccess);
}

// Test: Stream wait event
TEST_F(DeviceAdaptorTest, StreamWaitEvent) {
  sdcclStream_t stream2 = nullptr;
  sdcclEvent_t event = nullptr;

  // Create second stream and event
  ASSERT_EQ(devHandle->streamCreate(&stream2), sdcclSuccess);
  ASSERT_EQ(devHandle->eventCreate(&event, sdcclEventDefault), sdcclSuccess);

  // Record event on first stream
  auto result = devHandle->eventRecord(event, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Make second stream wait for event
  result = devHandle->streamWaitEvent(stream2, event);
  EXPECT_EQ(result, sdcclSuccess);

  // Synchronize both streams
  result = devHandle->streamSynchronize(stream);
  EXPECT_EQ(result, sdcclSuccess);
  result = devHandle->streamSynchronize(stream2);
  EXPECT_EQ(result, sdcclSuccess);

  // Cleanup
  devHandle->eventDestroy(event);
  devHandle->streamDestroy(stream2);
}

// Test: Device synchronize
TEST_F(DeviceAdaptorTest, DeviceSynchronize) {
  auto result = devHandle->deviceSynchronize();
  EXPECT_EQ(result, sdcclSuccess);
}

// Test: Set device
TEST_F(DeviceAdaptorTest, SetDevice) {
  int numDevices = 0;
  devHandle->getDeviceCount(&numDevices);

  // Set to device 0 (always exists)
  auto result = devHandle->setDevice(0);
  EXPECT_EQ(result, sdcclSuccess);

  int currentDevice = -1;
  devHandle->getDevice(&currentDevice);
  EXPECT_EQ(currentDevice, 0);

  // If multiple devices, test switching
  if (numDevices > 1) {
    result = devHandle->setDevice(1);
    EXPECT_EQ(result, sdcclSuccess);

    devHandle->getDevice(&currentDevice);
    EXPECT_EQ(currentDevice, 1);

    // Switch back to device 0
    devHandle->setDevice(0);
  }
}

// Test: Large memory allocation
TEST_F(DeviceAdaptorTest, LargeMemoryAlloc) {
  void *devPtr = nullptr;
  const size_t largeSize = 100 * 1024 * 1024; // 100 MB

  // Allocate large device memory
  auto result =
      devHandle->deviceMalloc(&devPtr, largeSize, sdcclMemDevice, stream);
  EXPECT_EQ(result, sdcclSuccess);
  EXPECT_NE(devPtr, nullptr);

  if (devPtr) {
    // Set memory to verify it's accessible
    result =
        devHandle->deviceMemset(devPtr, 0, largeSize, sdcclMemDevice, stream);
    EXPECT_EQ(result, sdcclSuccess);

    result = devHandle->streamSynchronize(stream);
    EXPECT_EQ(result, sdcclSuccess);

    // Free memory
    devHandle->deviceFree(devPtr, sdcclMemDevice, stream);
  }
}

// Test: Event timing (record and synchronize)
TEST_F(DeviceAdaptorTest, EventTiming) {
  sdcclEvent_t startEvent = nullptr;
  sdcclEvent_t endEvent = nullptr;

  ASSERT_EQ(devHandle->eventCreate(&startEvent, sdcclEventDefault),
            sdcclSuccess);
  ASSERT_EQ(devHandle->eventCreate(&endEvent, sdcclEventDefault),
            sdcclSuccess);

  // Record start event
  auto result = devHandle->eventRecord(startEvent, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Do some work (memory allocation and set)
  void *devPtr = nullptr;
  devHandle->deviceMalloc(&devPtr, TEST_SIZE, sdcclMemDevice, stream);
  devHandle->deviceMemset(devPtr, 0, TEST_SIZE, sdcclMemDevice, stream);

  // Record end event
  result = devHandle->eventRecord(endEvent, stream);
  EXPECT_EQ(result, sdcclSuccess);

  // Synchronize both events
  result = devHandle->eventSynchronize(startEvent);
  EXPECT_EQ(result, sdcclSuccess);
  result = devHandle->eventSynchronize(endEvent);
  EXPECT_EQ(result, sdcclSuccess);

  // Query events (should be completed)
  result = devHandle->eventQuery(startEvent);
  EXPECT_EQ(result, sdcclSuccess);
  result = devHandle->eventQuery(endEvent);
  EXPECT_EQ(result, sdcclSuccess);

  // Cleanup
  devHandle->deviceFree(devPtr, sdcclMemDevice, stream);
  devHandle->eventDestroy(startEvent);
  devHandle->eventDestroy(endEvent);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
