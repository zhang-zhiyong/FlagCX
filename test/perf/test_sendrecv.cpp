#include "sdccl.h"
#include "tools.h"
#include <cstring>
#include <iostream>

#define DATATYPE sdcclFloat

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

  sdcclStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hello = nullptr;
  void *sendHandle = nullptr;
  void *recvHandle = nullptr;
  size_t count;
  timer tim;
  int recvPeer = (proc - 1 + totalProcs) % totalProcs;
  int sendPeer = (proc + 1) % totalProcs;

  if (local_register) {
    // allocate buffer
    sdcclMemAlloc(&sendbuff, max_bytes);
    sdcclMemAlloc(&recvbuff, max_bytes);
    // register buffer
    sdcclCommRegister(comm, sendbuff, max_bytes, &sendHandle);
    sdcclCommRegister(comm, recvbuff, max_bytes, &recvHandle);
  } else {
    devHandle->deviceMalloc(&sendbuff, max_bytes, sdcclMemDevice, NULL);
    devHandle->deviceMalloc(&recvbuff, max_bytes, sdcclMemDevice, NULL);
  }
  hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Warm-up for large size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclGroupStart(comm);
    sdcclSend(sendbuff, max_bytes / sizeof(float), DATATYPE, sendPeer, comm,
               stream);
    sdcclRecv(recvbuff, max_bytes / sizeof(float), DATATYPE, recvPeer, comm,
               stream);
    sdcclGroupEnd(comm);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclGroupStart(comm);
    sdcclSend(sendbuff, min_bytes / sizeof(float), DATATYPE, sendPeer, comm,
               stream);
    sdcclRecv(recvbuff, min_bytes / sizeof(float), DATATYPE, recvPeer, comm,
               stream);
    sdcclGroupEnd(comm);
  }
  devHandle->streamSynchronize(stream);

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    count = size / sizeof(float);

    strcpy((char *)hello, "_0x1234");
    strcpy((char *)hello + size / 3, "_0x5678");
    strcpy((char *)hello + size / 3 * 2, "_0x9abc");

    devHandle->deviceMemcpy(sendbuff, hello, size, sdcclMemcpyHostToDevice,
                            NULL);

    if (proc == 0 && color == 0 && print_buffer) {
      printf("sendbuff = ");
      printf("%s", (const char *)((char *)hello));
      printf("%s", (const char *)((char *)hello + size / 3));
      printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      sdcclGroupStart(comm);
      sdcclSend(sendbuff, count, DATATYPE, sendPeer, comm, stream);
      sdcclRecv(recvbuff, count, DATATYPE, recvPeer, comm, stream);
      sdcclGroupEnd(comm);
    }
    devHandle->streamSynchronize(stream);

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double base_bw = (double)(size) / 1.0E9 / elapsed_time;
    double alg_bw = base_bw;
    double factor = 1;
    double bus_bw = base_bw * factor;
    if (proc == 0 && color == 0) {
      printf("Comm size: %zu bytes; Elapsed time: %lf sec; Algo bandwidth: %lf "
             "GB/s; Bus bandwidth: %lf GB/s\n",
             size, elapsed_time, alg_bw, bus_bw);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    memset(hello, 0, size);
    devHandle->deviceMemcpy(hello, recvbuff, size, sdcclMemcpyDeviceToHost,
                            NULL);
    if (proc == 0 && color == 0 && print_buffer) {
      printf("recvbuff = ");
      printf("%s", (const char *)((char *)hello));
      printf("%s", (const char *)((char *)hello + size / 3));
      printf("%s\n", (const char *)((char *)hello + size / 3 * 2));
    }
  }

  if (local_register) {
    // deregister buffer
    sdcclCommDeregister(comm, sendHandle);
    sdcclCommDeregister(comm, recvHandle);
    // deallocate buffer
    sdcclMemFree(sendbuff);
    sdcclMemFree(recvbuff);
  } else {
    devHandle->deviceFree(sendbuff, sdcclMemDevice, NULL);
    devHandle->deviceFree(recvbuff, sdcclMemDevice, NULL);
  }
  free(hello);
  devHandle->streamDestroy(stream);
  sdcclCommDestroy(comm);
  sdcclHandleFree(handler);

  MPI_Finalize();
  return 0;
}