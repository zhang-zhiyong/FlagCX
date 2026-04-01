#include "sdccl.h"
#include "sdccl_kernel.h"
#include "tools.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

namespace {

void fatal(sdcclResult_t res, const char *msg, int rank) {
  if (res != sdcclSuccess) {
    fprintf(stderr, "[rank %d] %s (err=%d)\n", rank, msg, int(res));
    MPI_Abort(MPI_COMM_WORLD, res);
  }
}
} // namespace

int main(int argc, char *argv[]) {
  parser args(argc, argv);
  size_t min_bytes = args.getMinBytes();
  size_t max_bytes = args.getMaxBytes();
  int step_factor = args.getStepFactor();
  int num_warmup_iters = args.getWarmupIters();
  int num_iters = args.getTestIters();
  int print_buffer = args.isPrintBuffer();
  uint64_t split_mask = args.getSplitMask();
  int local_register = args.getLocalRegister();

  // RMA requires sdcclMemAlloc (GDR memory with SYNC_MEMOPS)
  if (local_register < 1) {
    fprintf(stderr,
            "test_put requires -R 1 or -R 2 for GDR buffer allocation.\n");
    return 1;
  }

  sdcclHandlerGroup_t handler;
  sdcclHandleInit(&handler);
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
  sdcclComm_t &comm = handler->comm;
  sdcclDeviceHandle_t &devHandle = handler->devHandle;

  int color = 0;
  int worldSize = 1, worldRank = 0;
  int totalProcs = 1, proc = 0;
  MPI_Comm splitComm;
  initMpiEnv(argc, argv, worldRank, worldSize, proc, totalProcs, color,
             splitComm, split_mask);

  int nGpu;
  devHandle->getDeviceCount(&nGpu);
  devHandle->setDevice(worldRank % nGpu);

  if (proc == 0)
    sdcclGetUniqueId(&uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(sdcclUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  sdcclCommInitRank(&comm, totalProcs, uniqueId, proc);

  int isHomo = 0;
  sdcclIsHomoComm(comm, &isHomo);
  if (isHomo) {
    if (proc == 0)
      printf("Skipping put benchmark: hetero communicator not initialised "
             "(isHomo=%d).\n",
             isHomo);
    sdcclCommDestroy(comm);
    sdcclHandleFree(handler);
    MPI_Finalize();
    return 0;
  }

  if (totalProcs < 2) {
    if (proc == 0)
      printf("test_put requires at least 2 MPI processes\n");
    MPI_Finalize();
    return 0;
  }

  const int senderRank = 0;
  const int receiverRank = 1;
  if (totalProcs != 2) {
    if (proc == 0)
      printf("test_put requires exactly 2 ranks (sender=0, receiver=1).\n");
    MPI_Finalize();
    return 0;
  }

  bool isSender = (proc == senderRank);
  bool isReceiver = (proc == receiverRank);

  sdcclResult_t res;

  size_t signalBytes = sizeof(uint64_t);
  size_t total_iters_per_size = num_warmup_iters + num_iters;
  size_t max_data_iters = std::max(num_warmup_iters, num_iters);
  size_t data_bytes = max_bytes * max_data_iters;
  size_t signal_total_bytes = signalBytes * total_iters_per_size;

  // Data buffer: GDR memory (SYNC_MEMOPS ensures NIC visibility via GDR BAR)
  void *dataWindow = nullptr;
  res = sdcclMemAlloc(&dataWindow, data_bytes);
  if (res != sdcclSuccess || dataWindow == nullptr) {
    fprintf(stderr, "[rank %d] sdcclMemAlloc failed for data (size=%zu)\n",
            proc, data_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(dataWindow, 0, data_bytes, sdcclMemDevice, NULL);

  // Signal buffer: GDR memory (SYNC_MEMOPS for RDMA ATOMIC visibility)
  void *signalWindow = nullptr;
  res = sdcclMemAlloc(&signalWindow, signal_total_bytes);
  if (res != sdcclSuccess || signalWindow == nullptr) {
    fprintf(stderr, "[rank %d] sdcclMemAlloc failed for signal (size=%zu)\n",
            proc, signal_total_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  devHandle->deviceMemset(signalWindow, 0, signal_total_bytes, sdcclMemDevice,
                          NULL);

  // Register data buffer in global reg pool
  void *dataHandle = nullptr;
  res = sdcclCommRegister(comm, dataWindow, data_bytes, &dataHandle);
  fatal(res, "sdcclCommRegister (data) failed", proc);

  // Register data buffer for one-sided operations
  res = sdcclOneSideRegister(comm, dataWindow, data_bytes);
  if (res == sdcclNotSupported) {
    if (proc == 0)
      printf("Skipping put benchmark: net adaptor does not support iput.\n");
    sdcclCommDeregister(comm, dataHandle);
    sdcclMemFree(dataWindow);
    sdcclMemFree(signalWindow);
    sdcclCommDestroy(comm);
    sdcclHandleFree(handler);
    MPI_Finalize();
    return 0;
  }
  fatal(res, "sdcclOneSideRegister (data) failed", proc);

  // Register signal buffer for one-sided operations
  res = sdcclOneSideSignalRegister(comm, signalWindow, signal_total_bytes);
  fatal(res, "sdcclOneSideSignalRegister failed", proc);

  sdcclStream_t stream;
  devHandle->streamCreate(&stream);
  void *dummyBuff = nullptr;
  devHandle->deviceMalloc(&dummyBuff, 1, sdcclMemDevice, NULL);

  // Both sides must call GroupStart/GroupEnd together to ensure synchronization
  sdcclGroupStart(comm);
  if (isSender) {
    sdcclSend(dummyBuff, 1, sdcclChar, receiverRank, comm, stream);
  } else if (isReceiver) {
    sdcclRecv(dummyBuff, 1, sdcclChar, senderRank, comm, stream);
  }
  sdcclGroupEnd(comm);

  // Wait for the connection to be fully established
  devHandle->streamSynchronize(stream);
  devHandle->deviceFree(dummyBuff, sdcclMemDevice, NULL);
  devHandle->streamDestroy(stream);

  // Additional barrier to ensure connection is ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Host staging buffer for sender data fill and receiver verification
  void *hostStaging = nullptr;
  if (posix_memalign(&hostStaging, 64, max_bytes) != 0 ||
      hostStaging == nullptr) {
    fprintf(stderr, "[rank %d] posix_memalign failed for staging (size=%zu)\n",
            proc, max_bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Create stream for receiver-side wait operations
  sdcclStream_t waitStream = nullptr;
  if (isReceiver) {
    devHandle->streamCreate(&waitStream);
  }

  // Benchmark loop
  timer tim;
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    if (size == 0)
      break;

    // Reset signal buffer before each size iteration
    devHandle->deviceMemset(signalWindow, 0, signal_total_bytes,
                            sdcclMemDevice, NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup iterations (signal slots [0 .. num_warmup_iters-1])
    for (int i = 0; i < num_warmup_iters; ++i) {
      size_t signalOffset = i * signalBytes;
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {
        // Fill host staging, then copy H2D to device data buffer
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + current_send_offset,
                                hostStaging, size, sdcclMemcpyHostToDevice,
                                NULL);

        res = sdcclPutSignal(comm, receiverRank, current_send_offset,
                              current_recv_offset, size, signalOffset, 0, 0, 1);
        fatal(res, "sdcclPutSignal warmup failed", proc);
      } else if (isReceiver) {
        res = sdcclWaitSignal(comm, senderRank, signalOffset, 1, waitStream);
        fatal(res, "sdcclWaitSignal warmup failed", proc);
        devHandle->streamSynchronize(waitStream);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tim.reset();

    // Benchmark iterations (signal slots [num_warmup_iters .. total_iters-1])
    for (int i = 0; i < num_iters; ++i) {
      size_t signalOffset = (num_warmup_iters + i) * signalBytes;
      size_t current_send_offset = i * size;
      size_t current_recv_offset = i * size;

      if (isSender) {
        uint8_t value = static_cast<uint8_t>((senderRank + i) & 0xff);
        std::memset(hostStaging, value, size);
        devHandle->deviceMemcpy((char *)dataWindow + current_send_offset,
                                hostStaging, size, sdcclMemcpyHostToDevice,
                                NULL);

        res = sdcclPutSignal(comm, receiverRank, current_send_offset,
                              current_recv_offset, size, signalOffset, 0, 0, 1);
        fatal(res, "sdcclPutSignal failed", proc);
      } else if (isReceiver) {
        res = sdcclWaitSignal(comm, senderRank, signalOffset, 1, waitStream);
        fatal(res, "sdcclWaitSignal failed", proc);
        devHandle->streamSynchronize(waitStream);

        if (print_buffer) {
          // Copy device data to host for verification
          devHandle->deviceMemcpy(
              hostStaging, (char *)dataWindow + current_recv_offset,
              std::min(size, (size_t)64), sdcclMemcpyDeviceToHost, NULL);
          printf("[rank %d] Received data at offset %zu, size %zu:\n", proc,
                 current_recv_offset, size);
          for (size_t j = 0; j < size && j < 64; ++j) {
            printf("%02x ", ((unsigned char *)hostStaging)[j]);
            if ((j + 1) % 16 == 0)
              printf("\n");
          }
          if (size > 64)
            printf("... (truncated)\n");
          else
            printf("\n");
        }
      }
    }

    if (num_iters > 0) {
      double elapsed_time = tim.elapsed() / num_iters;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      elapsed_time /= worldSize;

      double bandwidth = (double)size / 1.0e9 / elapsed_time;
      if (proc == 0 && color == 0) {
        printf("Size: %zu bytes; Avg time: %lf sec; Bandwidth: %lf GB/s\n",
               size, elapsed_time, bandwidth);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);
  res = sdcclCommDeregister(comm, dataHandle);
  fatal(res, "sdcclCommDeregister failed", proc);

  sdcclOneSideDeregister(comm);
  sdcclOneSideSignalDeregister(comm);
  sdcclMemFree(dataWindow);
  sdcclMemFree(signalWindow);
  free(hostStaging);

  if (waitStream != nullptr) {
    devHandle->streamDestroy(waitStream);
  }

  fatal(sdcclCommDestroy(comm), "sdcclCommDestroy failed", proc);
  fatal(sdcclHandleFree(handler), "sdcclHandleFree failed", proc);

  MPI_Finalize();
  return 0;
}