/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_BOOTSTRAP_H_
#define SDCCL_BOOTSTRAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "sdccl.h"
#include "socket.h"

struct sdcclBootstrapHandle {
  uint64_t magic;
  union sdcclSocketAddress addr;
};
static_assert(sizeof(struct sdcclBootstrapHandle) <= sizeof(sdcclUniqueId),
              "Bootstrap handle is too large to fit inside SDCCL unique ID");

struct bootstrapState {
  struct sdcclSocket listenSock;
  struct sdcclSocket ringRecvSocket;
  struct sdcclSocket ringSendSocket;
  union sdcclSocketAddress *peerCommAddresses;
  union sdcclSocketAddress *peerProxyAddresses;
  struct unexConn *unexpectedConnections;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t *abortFlag;
  char *bootstrapNetIfName;
  void *properties;
};

sdcclResult_t bootstrapNetInit();
sdcclResult_t bootstrapCreateRoot(struct sdcclBootstrapHandle *handle,
                                   bool idFromEnv);
sdcclResult_t bootstrapGetUniqueId(struct sdcclBootstrapHandle *handle);
sdcclResult_t bootstrapInit(struct sdcclBootstrapHandle *handle,
                             void *commState);
sdcclResult_t bootstrapAllGather(void *commState, void *allData, int size);

sdcclResult_t bootstrapSend(void *commState, int peer, int tag, void *data,
                             int size);
sdcclResult_t bootstrapRecv(void *commState, int peer, int tag, void *data,
                             int size);
sdcclResult_t bootstrapBarrier(void *commState, int rank, int nranks, int tag);
sdcclResult_t bootstrapBroadcast(void *commState, int rank, int nranks,
                                  int root, void *bcastData, int size);
sdcclResult_t bootstrapIntraNodeBarrier(void *commState, int *ranks, int rank,
                                         int nranks, int tag);
sdcclResult_t bootstrapIntraNodeBroadcast(void *commState, int *ranks,
                                           int rank, int nranks, int root,
                                           void *bcastData, int size);
sdcclResult_t bootstrapClose(void *commState);
sdcclResult_t bootstrapAbort(void *commState);

/* A bunch of collective communication operators */
/*
 * Broadcast
 *
 * Root device send sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
sdcclResult_t BroadcastBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  sdcclDataType_t datatype, int root);

/* A bunch of collective communication operators */
/*
 * Gather
 *
 * Each rank sends sendcount values from its sendbuff to the root rank.
 * Root rank receives data from rank i at offset i*sendcount in recvbuff.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
sdcclResult_t GatherBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               sdcclDataType_t datatype, int root);

/* A bunch of collective communication operators */
/*
 * Scatter
 *
 * Root rank sends sendcount values to each rank, with data for rank i
 * starting at offset i*sendcount in sendbuff.
 * Each rank receives sendcount values into its recvbuff.
 * Assumes sendcount is equal to recvcount for each rank.
 *
 * In-place operations will happen if recvbuff = sendbuff + rank * sendcount.
 */
sdcclResult_t ScatterBootstrap(void *commState, const void *sendbuff,
                                void *recvbuff, size_t count,
                                sdcclDataType_t datatype, int root);
/* A bunch of collective communication operators */
/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
sdcclResult_t AllGatherBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  sdcclDataType_t datatype);
/*
 * All-Reduce
 *
 * Reduces data arrays of length count(NOT bytes size) in sendbuff using op
 * operation, and leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
sdcclResult_t AllReduceBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t count,
                                  sdcclDataType_t datatype, sdcclRedOp_t op);
/*
 * Reduce
 *
 * Reduces data arrays of length count(NOT bytes size) in sendbuff using op
 * operation, and leaves identical copies of result on root recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
sdcclResult_t ReduceBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               sdcclDataType_t datatype, sdcclRedOp_t op,
                               int root);
/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
sdcclResult_t ReduceScatterBootstrap(void *commState, const void *sendbuff,
                                      void *recvbuff, size_t recvcount,
                                      sdcclDataType_t datatype,
                                      sdcclRedOp_t op);

/*
 * All-to-all
 *
 * Every rank sends j-th block of its own sendbuff to the j-th rank of the
 * communicator. Meanwhile, every rank receives j-th block of its own recvbuff
 * from j-th rank.
 *
 * Every block has the size of count elements.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
sdcclResult_t AlltoAllBootstrap(void *commState, const void *sendbuff,
                                 void *recvbuff, size_t count,
                                 sdcclDataType_t datatype);

/*
 * All-to-all with variable block sizes
 *
 * Every rank sends j-th block of its own sendbuff to the j-th rank of the
 * communicator. Meanwhile, every rank receives j-th block of its own recvbuff
 * from j-th rank.
 *
 * Each block can have different sizes:
 * - sendcounts[j] specifies the number of elements to send to rank j
 * - sdispls[j] specifies the offset in sendbuff for the j-th block
 * - recvcounts[j] specifies the number of elements to receive from rank j
 * - rdispls[j] specifies the offset in recvbuff for the j-th block
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
sdcclResult_t AlltoAllvBootstrap(void *commState, const void *sendbuff,
                                  size_t *sendcounts, size_t *sdispls,
                                  void *recvbuff, size_t *recvcounts,
                                  size_t *rdispls, sdcclDataType_t datatype);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
