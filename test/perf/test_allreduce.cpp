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

  const char *envLocRed = getenv("SDCCL_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("SDCCL_UNIRUNNER_USE_RINGAG");

  // Warm-up for large size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclAllReduce(sendbuff, recvbuff, max_bytes / sizeof(float), DATATYPE,
                    sdcclSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);

  // Warm-up for small size
  for (int i = 0; i < num_warmup_iters; i++) {
    sdcclAllReduce(sendbuff, recvbuff, min_bytes / sizeof(float), DATATYPE,
                    sdcclSum, comm, stream);
  }
  devHandle->streamSynchronize(stream);
  for (size_t size = min_bytes; size <= max_bytes; size *= step_factor) {
    count = size / sizeof(float);

    for (size_t i = 0; i < count; i++) {
      ((float *)hello)[i] = i % 10 * (1 << proc);
    }

    devHandle->deviceMemcpy(sendbuff, hello, size, sdcclMemcpyHostToDevice,
                            NULL);
    devHandle->deviceMemcpy(recvbuff, hello, size, sdcclMemcpyHostToDevice,
                            NULL);

    if (color == 0 && print_buffer) {
      printf("rank %d sendbuff = ", proc);
      for (size_t i = 0; i < 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tim.reset();
    for (int i = 0; i < num_iters; i++) {
      sdcclAllReduce(sendbuff, recvbuff, count, DATATYPE, sdcclSum, comm,
                      stream);
    }
    devHandle->streamSynchronize(stream);

    double elapsed_time = tim.elapsed() / num_iters;
    MPI_Allreduce(MPI_IN_PLACE, (void *)&elapsed_time, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    elapsed_time /= worldSize;

    double base_bw = (double)(size) / 1.0E9 / elapsed_time;
    double alg_bw = base_bw;
    double factor = ((double)(2 * (totalProcs - 1))) / ((double)(totalProcs));
    if (envLocRed != NULL && atoi(envLocRed) == 1) {
      factor = 1;
    } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
      factor = ((double)(totalProcs - 1)) / ((double)(totalProcs));
    }
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
    if (color == 0 && print_buffer) {
      printf("rank %d recvbuff = ", proc);
      for (size_t i = 0; i < 10; i++) {
        printf("%f ", ((float *)hello)[i]);
      }
      printf("\n");

      if (envLocRed != NULL && atoi(envLocRed) == 1) {
        /* red correctness check */
        int red_correct = 1;
        for (size_t i = 0; i < count; i++) {
          if (i * totalProcs / count == (size_t)proc)
            continue;
          if (((float *)hello)[i] != (float)(i % 10 * (1 << proc))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n ",
                   proc, i, (float)(i % 10 * (1 << proc)), ((float *)hello)[i]);
            red_correct = 0;
            break;
          }
        }
        for (size_t i = proc * count / totalProcs;
             i < (proc + 1) * count / totalProcs; i++) {
          if (((float *)hello)[i] != (float)(i % 10 * (1 << (proc + 1)))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * (1 << (proc + 1))),
                   ((float *)hello)[i]);
            red_correct = 0;
            break;
          }
        }
        printf("rank %d reduce correctness = %d\n", proc, red_correct);
      } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
        /* p2p correctness check */
        int p2p_correct = 1;
        for (size_t i = 0; i < count; i++) {
          if (((float *)hello)[i] !=
              (float)(i % 10 * (1 << (i * totalProcs / count)))) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * (1 << (i * totalProcs / count))),
                   ((float *)hello)[i]);
            p2p_correct = 0;
            break;
          }
        }
        printf("rank %d p2p correctness = %d\n", proc, p2p_correct);
      } else {
        /* all-reduce correctness check */
        int ar_correct = 1;
        for (size_t i = 0; i < count; i++) {
          if ((i % 10 == 0 && ((float *)hello)[i] != 0) ||
              ((float *)hello)[i] / (float)(i % 10 * ((1 << totalProcs) - 1)) >
                  1 + 1e-5 ||
              ((float *)hello)[i] / (float)(i % 10 * ((1 << totalProcs) - 1)) <
                  1 - 1e-5) {
            printf("rank %d wrong output at offset %lu, expected %f, got %f\n",
                   proc, i, (float)(i % 10 * ((1 << totalProcs) - 1)),
                   ((float *)hello)[i]);
            ar_correct = 0;
            break;
          }
        }
        printf("rank %d all-reduce correctness = %d\n", proc, ar_correct);
      }
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
