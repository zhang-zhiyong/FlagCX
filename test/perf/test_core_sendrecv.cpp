#include "sdccl.h"
#include "sdccl_hetero.h"
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

  sdcclHandlerGroup_t handler;
  sdcclHandleInit(&handler);
  sdcclUniqueId_t &uniqueId = handler->uniqueId;
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
    sdcclHeteroGetUniqueId(uniqueId);
  MPI_Bcast((void *)uniqueId, sizeof(sdcclUniqueId), MPI_BYTE, 0, splitComm);
  MPI_Barrier(MPI_COMM_WORLD);

  sdcclHeteroComm_t comm;
  sdcclHeteroCommInitRank(&comm, totalProcs, *uniqueId, proc);

  sdcclStream_t stream;
  devHandle->streamCreate(&stream);

  void *sendbuff = nullptr;
  void *recvbuff = nullptr;
  void *hello = nullptr;
  void *selfsendbuff1 = nullptr;
  void *selfsendbuff2 = nullptr;
  void *selfrecvbuff1 = nullptr;
  void *selfrecvbuff2 = nullptr;
  timer tim;
  int peerSend = (proc + 1) % totalProcs;
  int peerRecv = (proc - 1 + totalProcs) % totalProcs;
  int selfPeer = proc;

  devHandle->deviceMalloc(&sendbuff, max_bytes, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&recvbuff, max_bytes, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff1, 100, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&selfsendbuff2, 200, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff1, 100, sdcclMemDevice, NULL);
  devHandle->deviceMalloc(&selfrecvbuff2, 200, sdcclMemDevice, NULL);
  hello = malloc(max_bytes);
  memset(hello, 0, max_bytes);

  // Warm-up for large size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclHeteroGroupStart();
    sdcclHeteroSend(sendbuff, max_bytes, sdcclChar, peerSend, comm, stream);
    sdcclHeteroRecv(recvbuff, max_bytes, sdcclChar, peerRecv, comm, stream);
    sdcclHeteroGroupEnd();
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclHeteroGroupStart();
    sdcclHeteroSend(sendbuff, min_bytes, sdcclChar, peerSend, comm, stream);
    sdcclHeteroRecv(recvbuff, min_bytes, sdcclChar, peerRecv, comm, stream);
    sdcclHeteroGroupEnd();
  }
  devHandle->streamSynchronize(stream);
  void *testdata1 = malloc(100);
  void *testdata2 = malloc(200);
  memset(testdata1, 0xAA, 100);
  memset(testdata2, 0xBB, 200);
  devHandle->deviceMemcpy(selfsendbuff1, testdata1, 100,
                          sdcclMemcpyHostToDevice, NULL);
  devHandle->deviceMemcpy(selfsendbuff2, testdata2, 200,
                          sdcclMemcpyHostToDevice, NULL);
  memset(testdata1, 0, 100);
  memset(testdata2, 0, 200);
  devHandle->deviceMemcpy(selfrecvbuff1, testdata1, 100,
                          sdcclMemcpyHostToDevice, NULL);
  devHandle->deviceMemcpy(selfrecvbuff2, testdata2, 200,
                          sdcclMemcpyHostToDevice, NULL);

  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {

    for (size_t i = 0; i + 13 <= size; i += 13) {
      strcpy((char *)hello + i, std::to_string(i / (13)).c_str());
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, sdcclMemcpyHostToDevice,
                            NULL);

    if (proc == 0 && color == 0 && print_buffer) {
      printf("sendbuff = ");
      for (size_t i = 0; i + 13 <= 50; i += 13) {
        printf("%c", ((char *)hello)[i]);
      }
      printf("\n");
      memset(testdata1, 0, 100);
      devHandle->deviceMemcpy(testdata1, selfsendbuff1, 100,
                              sdcclMemcpyDeviceToHost, NULL);
      printf("selfsendbuff1 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata1)[i]);
      }
      printf("\n");
      memset(testdata2, 0, 200);
      devHandle->deviceMemcpy(testdata2, selfsendbuff2, 200,
                              sdcclMemcpyDeviceToHost, NULL);
      printf("selfsendbuff2 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata2)[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      sdcclHeteroGroupStart();
      sdcclHeteroSend(sendbuff, size, sdcclChar, peerSend, comm, stream);
      sdcclHeteroRecv(recvbuff, size, sdcclChar, peerRecv, comm, stream);
      sdcclHeteroSend(selfsendbuff1, 100, sdcclChar, selfPeer, comm, stream);
      sdcclHeteroSend(selfsendbuff2, 200, sdcclChar, selfPeer, comm, stream);
      sdcclHeteroRecv(selfrecvbuff2, 200, sdcclChar, selfPeer, comm, stream);
      sdcclHeteroRecv(selfrecvbuff1, 100, sdcclChar, selfPeer, comm, stream);
      sdcclHeteroGroupEnd();
    }
    devHandle->streamSynchronize(stream);

    devHandle->deviceMemcpy(testdata1, selfrecvbuff1, 100,
                            sdcclMemcpyDeviceToHost, NULL);
    devHandle->deviceMemcpy(testdata2, selfrecvbuff2, 200,
                            sdcclMemcpyDeviceToHost, NULL);
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

    if (proc == 0 && color == 0 && print_buffer) {
      memset(hello, 0, size);
      devHandle->deviceMemcpy(hello, recvbuff, size, sdcclMemcpyDeviceToHost,
                              NULL);
      printf("recvbuff = ");
      for (size_t i = 0; i + 13 <= 50; i += 13) {
        printf("%c", ((char *)hello)[i]);
      }
      printf("\n");
      memset(testdata1, 0, 100);
      devHandle->deviceMemcpy(testdata1, selfrecvbuff1, 100,
                              sdcclMemcpyDeviceToHost, NULL);
      printf("selfrecvbuff1 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata1)[i]);
      }
      printf("\n");
      memset(testdata2, 0, 200);
      devHandle->deviceMemcpy(testdata2, selfrecvbuff2, 200,
                              sdcclMemcpyDeviceToHost, NULL);
      printf("selfrecvbuff2 = ");
      for (int i = 0; i < 10; i++) {
        printf("0x%02X ", ((unsigned char *)testdata2)[i]);
      }
      printf("\n");
    }
  }

  // if (local_register) {
  //   // deregister buffer
  //   sdcclCommDeregister(comm, sendHandle);
  //   sdcclCommDeregister(comm, recvHandle);
  //   // deallocate buffer
  //   sdcclMemFree(sendbuff);
  //   sdcclMemFree(recvbuff);
  // } else {
  devHandle->deviceFree(sendbuff, sdcclMemDevice, NULL);
  devHandle->deviceFree(recvbuff, sdcclMemDevice, NULL);
  devHandle->deviceFree(selfsendbuff1, sdcclMemDevice, NULL);
  devHandle->deviceFree(selfsendbuff2, sdcclMemDevice, NULL);
  devHandle->deviceFree(selfrecvbuff1, sdcclMemDevice, NULL);
  devHandle->deviceFree(selfrecvbuff2, sdcclMemDevice, NULL);
  // }
  free(hello);
  free(testdata1);
  free(testdata2);
  sdcclHeteroCommDestroy(comm);
  devHandle->streamDestroy(stream);
  sdcclHandleFree(handler);

  MPI_Finalize();
  return 0;
}