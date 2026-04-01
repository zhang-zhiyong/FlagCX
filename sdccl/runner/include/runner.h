/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_RUNNER_H_
#define SDCCL_RUNNER_H_

#include "adaptor.h"
#include "sdccl.h"
#include "global_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NRUNNERS 4
#define sdcclHomoRunner 0
#define sdcclHostRunner 1
#define sdcclHybridRunner 2
#define sdcclUniRunner 3

extern struct sdcclRunner homoRunner;
extern struct sdcclRunner hostRunner;
extern struct sdcclRunner hybridRunner;
extern struct sdcclRunner uniRunner;
extern struct sdcclRunner *sdcclRunners[];

struct sdcclRunner {
  // Communication functions
  sdcclResult_t (*reduce)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, sdcclRedOp_t op,
                           int root, sdcclComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*gather)(const void *sendbuff, void *recvbuff, size_t count,
                           sdcclDataType_t datatype, int root,
                           sdcclComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*scatter)(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, int root,
                            sdcclComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*broadcast)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype, int root,
                              sdcclComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*allReduce)(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype,
                              sdcclRedOp_t op, sdcclComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*reduceScatter)(const void *sendbuff, void *recvbuff,
                                  size_t recvcount, sdcclDataType_t datatype,
                                  sdcclRedOp_t op, sdcclComm_t comm,
                                  sdcclStream_t stream);
  sdcclResult_t (*allGather)(const void *sendbuff, void *recvbuff,
                              size_t sendcount, sdcclDataType_t datatype,
                              sdcclComm_t comm, sdcclStream_t stream);
  sdcclResult_t (*alltoAll)(const void *sendbuff, void *recvbuff, size_t count,
                             sdcclDataType_t datatype, sdcclComm_t comm,
                             sdcclStream_t stream);
  sdcclResult_t (*alltoAllv)(const void *sendbuff, size_t *sendcounts,
                              size_t *sdispls, void *recvbuff,
                              size_t *recvcounts, size_t *rdispls,
                              sdcclDataType_t datatype, sdcclComm_t comm,
                              sdcclStream_t stream);
  sdcclResult_t (*send)(const void *sendbuff, size_t count,
                         sdcclDataType_t datatype, int peer, sdcclComm_t comm,
                         sdcclStream_t stream);
  sdcclResult_t (*recv)(void *recvbuff, size_t count,
                         sdcclDataType_t datatype, int peer, sdcclComm_t comm,
                         sdcclStream_t stream);

  // Group semantics
  sdcclResult_t (*groupStart)();
  sdcclResult_t (*groupEnd)();
};

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
