#ifndef SDCCL_HETERO_H_
#define SDCCL_HETERO_H_

#include "sdccl.h"
#include "type.h"
#include <climits>

typedef struct sdcclHeteroComm *sdcclHeteroComm_t;

sdcclResult_t sdcclHeteroGetVersion(int *version);

/* C++ style */
sdcclResult_t sdcclHeteroSend(const void *sendbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclHeteroComm_t comm, sdcclStream_t stream,
                                int opId = INT_MAX, int step = -1);

/* C++ style */
sdcclResult_t sdcclHeteroRecv(void *recvbuff, size_t count,
                                sdcclDataType_t datatype, int peer,
                                sdcclHeteroComm_t comm, sdcclStream_t stream,
                                int opId = INT_MAX, int step = -1);

sdcclResult_t sdcclHeteroGroupStart();

sdcclResult_t sdcclHeteroGroupEnd();

sdcclResult_t sdcclHeteroGetUniqueId(sdcclUniqueId *out);

sdcclResult_t sdcclHeteroCommInitRank(sdcclHeteroComm_t *newcomm, int nranks,
                                        sdcclUniqueId commId, int myrank);

sdcclResult_t sdcclHeteroCommCount(const sdcclHeteroComm_t comm, int *count);

sdcclResult_t sdcclHeteroCommUserRank(const sdcclHeteroComm_t comm,
                                        int *rank);

sdcclResult_t sdcclHeteroCommDestroy(sdcclHeteroComm_t comm);

sdcclResult_t sdcclHeteroPut(sdcclHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx);

// RDMA READ: pull data from remote peer's srcMrIdx buffer into local dstMrIdx
// buffer
sdcclResult_t sdcclHeteroGet(sdcclHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx);

// Data + signal combined (chained WRITE + ATOMIC in IB backend)
// When size == 0, only signal ATOMIC is posted (signal-only mode)
sdcclResult_t sdcclHeteroPutSignal(sdcclHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue);

sdcclResult_t sdcclHeteroFlush(sdcclHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo);

sdcclResult_t sdcclHeteroWaitSignal(sdcclHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      sdcclStream_t stream);

// Put a 64-bit value to remote peer's buffer at dstOffset.
// Writes value to local staging buffer then does iput from staging MR.
sdcclResult_t sdcclHeteroPutValue(sdcclHeteroComm_t comm, int peer,
                                    uint64_t value, size_t dstOffset,
                                    int dstMrIdx);

#endif