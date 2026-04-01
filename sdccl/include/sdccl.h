#ifndef SDCCL_H_
#define SDCCL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

/* Error type */
typedef enum {
  sdcclSuccess = 0,
  sdcclUnhandledDeviceError = 1,
  sdcclSystemError = 2,
  sdcclInternalError = 3,
  sdcclInvalidArgument = 4,
  sdcclInvalidUsage = 5,
  sdcclRemoteError = 6,
  sdcclInProgress = 7,
  sdcclUnhandledCCLError = 8,
  sdcclNotSupported = 9,
  sdcclNumResults = 10
} sdcclResult_t;

/* Data types */
typedef enum {
  sdcclInt8 = 0,
  sdcclChar = 0,
  sdcclUint8 = 1,
  sdcclInt32 = 2,
  sdcclInt = 2,
  sdcclUint32 = 3,
  sdcclInt64 = 4,
  sdcclUint64 = 5,
  sdcclFloat16 = 6,
  sdcclHalf = 6,
  sdcclFloat32 = 7,
  sdcclFloat = 7,
  sdcclFloat64 = 8,
  sdcclDouble = 8,
  sdcclBfloat16 = 9,
  sdcclNumTypes = 10
} sdcclDataType_t;

/* Reduction operation selector */
typedef enum { sdcclNumRedOps_dummy = 5 } sdcclRedOp_dummy_t;
typedef enum {
  sdcclSum = 0,
  sdcclProd = 1,
  sdcclMax = 2,
  sdcclMin = 3,
  sdcclAvg = 4,
  sdcclRedNoOp = 5,
  sdcclNumRedOps = 5,
  sdcclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(sdcclRedOp_dummy_t))
} sdcclRedOp_t;

size_t getSdcclDataTypeSize(sdcclDataType_t dtype);

/* Communication operation type */
typedef enum {
  sdcclCommOpSend = 0,
  sdcclCommOpRecv = 1,
  sdcclCommOpBroadcast = 2,
  sdcclCommOpGather = 3,
  sdcclCommOpScatter = 4,
  sdcclCommOpReduce = 5,
  sdcclCommOpAllReduce = 6,
  sdcclCommOpAllGather = 7,
  sdcclCommOpReduceScatter = 8,
  sdcclCommOpAlltoAll = 9,
  sdcclCommOpAlltoAllv = 10,
  sdcclCommNoOp = 11,
  sdcclNumCommOps = 12
} sdcclCommOp_t;

typedef enum {
  sdcclMemcpyHostToDevice = 0,
  sdcclMemcpyDeviceToHost = 1,
  sdcclMemcpyDeviceToDevice = 2
} sdcclMemcpyType_t;

typedef enum {
  sdcclMemHost = 0, // pinned memory
  sdcclMemDevice = 1,
  sdcclMemManaged = 2
} sdcclMemType_t;

typedef enum {
  sdcclEventDefault = 0,
  sdcclEventDisableTiming = 1
} sdcclEventType_t;

// TODO: add more vendor types
typedef enum {
  SDCCL_VENDOR_NVIDIA = 0,
  SDCCL_VENDOR_ILUVATAR_COREX = 1,
  SDCCL_VENDOR_MLU = 2,
  SDCCL_VENDOR_METAX = 3,
} sdcclVendorType;

#define SDCCL_UNIQUE_ID_BYTES 256
typedef struct {
  char internal[SDCCL_UNIQUE_ID_BYTES];
} sdcclUniqueId;
typedef sdcclUniqueId *sdcclUniqueId_t;

/* Opaque handle to sdcclComm */
typedef struct sdcclComm *sdcclComm_t;
/* Opaque handle to sdcclStream */
typedef struct sdcclStream *sdcclStream_t;
/* Opaque handle to sdcclEvent */
typedef struct sdcclEvent *sdcclEvent_t;
/* Opaque handle to sdcclIpcMemHandle */
typedef struct sdcclIpcMemHandle *sdcclIpcMemHandle_t;
/* Opaque handle to sdcclWindow */
typedef struct sdcclWindow *sdcclWindow_t;

/* Func(kernel) arguments */
typedef struct {
  sdcclStream_t stream;
  sdcclEvent_t event;
  void **argList;
} sdcclFuncArgs;

struct sdcclDeviceHandle {
  // Basic functions
  sdcclResult_t (*deviceSynchronize)();
  sdcclResult_t (*deviceMemcpy)(void *dst, void *src, size_t size,
                                 sdcclMemcpyType_t type,
                                 sdcclStream_t stream);
  sdcclResult_t (*deviceMemset)(void *ptr, int value, size_t size,
                                 sdcclMemType_t type, sdcclStream_t stream);
  sdcclResult_t (*deviceMalloc)(void **ptr, size_t size, sdcclMemType_t type,
                                 sdcclStream_t stream);
  sdcclResult_t (*deviceFree)(void *ptr, sdcclMemType_t type,
                               sdcclStream_t stream);
  sdcclResult_t (*setDevice)(int dev);
  sdcclResult_t (*getDevice)(int *dev);
  sdcclResult_t (*getDeviceCount)(int *count);
  sdcclResult_t (*getVendor)(char *vendor);
  sdcclResult_t (*hostGetDevicePointer)(void **pDevice, void *pHost);
  // Stream functions
  sdcclResult_t (*streamCreate)(sdcclStream_t *stream);
  sdcclResult_t (*streamDestroy)(sdcclStream_t stream);
  sdcclResult_t (*streamCopy)(sdcclStream_t *newStream, void *oldStream);
  sdcclResult_t (*streamFree)(sdcclStream_t stream);
  sdcclResult_t (*streamSynchronize)(sdcclStream_t stream);
  sdcclResult_t (*streamQuery)(sdcclStream_t stream);
  sdcclResult_t (*streamWaitEvent)(sdcclStream_t stream, sdcclEvent_t event);
  // Event functions
  sdcclResult_t (*eventCreate)(sdcclEvent_t *event,
                                sdcclEventType_t eventType);
  sdcclResult_t (*eventDestroy)(sdcclEvent_t event);
  sdcclResult_t (*eventRecord)(sdcclEvent_t event, sdcclStream_t stream);
  sdcclResult_t (*eventSynchronize)(sdcclEvent_t event);
  sdcclResult_t (*eventQuery)(sdcclEvent_t event);
  // IpcMemHandle functions
  sdcclResult_t (*ipcMemHandleCreate)(sdcclIpcMemHandle_t *handle,
                                       size_t *size);
  sdcclResult_t (*ipcMemHandleGet)(sdcclIpcMemHandle_t handle, void *devPtr);
  sdcclResult_t (*ipcMemHandleOpen)(sdcclIpcMemHandle_t handle,
                                     void **devPtr);
  sdcclResult_t (*ipcMemHandleClose)(void *devPtr);
  sdcclResult_t (*ipcMemHandleFree)(sdcclIpcMemHandle_t handle);
};
typedef struct sdcclDeviceHandle *sdcclDeviceHandle_t;

struct sdcclHandlerGroup {
  sdcclUniqueId_t uniqueId;
  sdcclComm_t comm;
  sdcclDeviceHandle_t devHandle;
};
typedef struct sdcclHandlerGroup *sdcclHandlerGroup_t;

/* Init and free SDCCL handls including sdcclComm_t, sdcclStream_t */
sdcclResult_t sdcclHandleInit(sdcclHandlerGroup_t *handler);

sdcclResult_t sdcclHandleFree(sdcclHandlerGroup_t handler);

/* User buffer registration functions. The actual allocated size might
 * be larger than requested due to granularity requirement. */
sdcclResult_t sdcclMemAlloc(void **ptr, size_t size);
sdcclResult_t sdcclMemFree(void *ptr);

/* Register/Deregister user buffer for zero-copy operation */
sdcclResult_t sdcclCommRegister(const sdcclComm_t comm, void *buff,
                                  size_t size, void **handle);
sdcclResult_t sdcclCommDeregister(const sdcclComm_t comm, void *handle);

/* Window registration flags */
#define SDCCL_WIN_DEFAULT 0x00
#define SDCCL_WIN_COLL_SYMMETRIC 0x01

/* Register/Deregister user buffer for symmetric operation */
sdcclResult_t sdcclCommWindowRegister(sdcclComm_t comm, void *buff,
                                        size_t size, sdcclWindow_t *win,
                                        int winFlags);
sdcclResult_t sdcclCommWindowDeregister(sdcclComm_t comm,
                                          sdcclWindow_t win);

/* Check if the SDCCL communicator type is homogeneous or heterogeneous */
sdcclResult_t sdcclIsHomoComm(sdcclComm_t comm, int *isHomo);

/* Return the version of the SDCCL library in the supplied integer.
 * It contains the underlying adaptor library version and SDCCL core version
 */
sdcclResult_t sdcclGetVersion(int *version);

/* Generates an Id to be used in sdcclCommInitRank. sdcclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling sdcclCommInitRank. */
sdcclResult_t sdcclGetUniqueId(sdcclUniqueId_t *uniqueId);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a device, which has to be set before calling
 * sdcclCommInitRank. sdcclCommInitRank implicitly syncronizes with other
 * ranks, so it must be called by different threads/processes or use
 * sdcclGroupStart/sdcclGroupEnd. */
sdcclResult_t sdcclCommInitRank(sdcclComm_t *comm, int nranks,
                                  sdcclUniqueId_t commId, int rank);

/* Finalize a communicator. sdcclCommFinalize flushes all issued
 * communications, and marks communicator state as sdcclInProgress. The state
 * will change to sdcclSuccess when the communicator is globally quiescent and
 * related resources are freed; then, calling sdcclCommDestroy can locally free
 * the rest of the resources (e.g. communicator itself) without blocking. */
sdcclResult_t sdcclCommFinalize(sdcclComm_t comm);

/* Frees local resources associated with communicator object. */
sdcclResult_t sdcclCommDestroy(sdcclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
sdcclResult_t sdcclCommAbort(sdcclComm_t comm);

/* Resume a communicator. */
sdcclResult_t sdcclCommResume(sdcclComm_t comm);

/* Suspend a communicator. */
sdcclResult_t sdcclCommSuspend(sdcclComm_t comm);

/* Returns a string for each error code. */
const char *sdcclGetErrorString(sdcclResult_t result);

/* Returns a human-readable message of the last error that occurred. */
const char *sdcclGetLastError(sdcclComm_t comm);

/* Checks whether the comm has encountered any asynchronous errors */
sdcclResult_t sdcclCommGetAsyncError(sdcclComm_t comm,
                                       sdcclResult_t *asyncError);

/* Gets the number of ranks in the communicator clique. */
sdcclResult_t sdcclCommCount(const sdcclComm_t comm, int *count);

/* Returns the device number associated with the communicator. */
sdcclResult_t sdcclCommGetDeviceNumber(const sdcclComm_t comm, int *device);

/* Returns the user-ordered "rank" associated with the communicator. */
sdcclResult_t sdcclCommUserRank(const sdcclComm_t comm, int *rank);

/* Returns `(void *)fifoBuffer` associated with the `heteroComm` of the input
 * communicator */
sdcclResult_t sdcclCommFifoBuffer(const sdcclComm_t comm, void **buffer);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the SDCCL stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Barrier
 *
 * Blocks until all processes in the communicator have reached this routine.
 *
 */
sdcclResult_t sdcclBarrier(sdcclComm_t comm, sdcclStream_t stream);

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
sdcclResult_t sdcclReduce(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, sdcclRedOp_t op,
                            int root, sdcclComm_t comm, sdcclStream_t stream);

/*
 * Gather
 *
 * Gathers data arrays of length count in sendbuff into recvbuff.
 * recvbuff may bu NULL on all calls except root device.
 * root is the rank (not the device) where data will reside after the
 * operation is complete.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * count.
 */
sdcclResult_t sdcclGather(const void *sendbuff, void *recvbuff, size_t count,
                            sdcclDataType_t datatype, int root,
                            sdcclComm_t comm, sdcclStream_t stream);

/*
 * Scatter
 *
 * Scatters data arrays of sendcount in sendbuff into recvbuff.
 * sendbuff may bu NULL on all calls except root device.
 * root is the rank (not the device) where data will reside before the
 * operation is started.
 *
 * In-place operations will happen if sendbuff + rank * count == recvbuff.
 */
sdcclResult_t sdcclScatter(const void *sendbuff, void *recvbuff, size_t count,
                             sdcclDataType_t datatype, int root,
                             sdcclComm_t comm, sdcclStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
sdcclResult_t sdcclBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               int root, sdcclComm_t comm,
                               sdcclStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
sdcclResult_t sdcclAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, sdcclDataType_t datatype,
                               sdcclRedOp_t op, sdcclComm_t comm,
                               sdcclStream_t stream);

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
sdcclResult_t sdcclReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, sdcclDataType_t datatype,
                                   sdcclRedOp_t op, sdcclComm_t comm,
                                   sdcclStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other APUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
sdcclResult_t sdcclAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, sdcclDataType_t datatype,
                               sdcclComm_t comm, sdcclStream_t stream);

/*
 * All-to-all
 *
 * Each device sends count values to other APUs into recvbuffer,
 * receiving count values from other APUs into sendbuffer.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
sdcclResult_t sdcclAlltoAll(const void *sendbuff, void *recvbuff,
                              size_t count, sdcclDataType_t datatype,
                              sdcclComm_t comm, sdcclStream_t stream);

/*
 * All-to-allv
 *
 * Each device may send different count values to other APUs into recvbuffer,
 * receiving different count values from other APUs into sendbuffer.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
sdcclResult_t sdcclAlltoAllv(const void *sendbuff, size_t *sendcounts,
                               size_t *sdispls, void *recvbuff,
                               size_t *recvcounts, size_t *rdispls,
                               sdcclDataType_t datatype, sdcclComm_t comm,
                               sdcclStream_t stream);

/*
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call sdcclRecv with the same datatype and the same count
 * from this rank.
 *
 * This operation is blocking for the GPU. If multiple sdcclSend and sdcclRecv
 * operations need to progress concurrently to complete, they must be fused
 * within a sdcclGroupStart/ sdcclGroupEnd section.
 */
sdcclResult_t sdcclSend(const void *sendbuff, size_t count,
                          sdcclDataType_t datatype, int peer,
                          sdcclComm_t comm, sdcclStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call sdcclSend with the same datatype and the same count
 * to this rank.
 *
 * This operation is blocking for the GPU. If multiple sdcclSend and sdcclRecv
 * operations need to progress concurrently to complete, they must be fused
 * within a sdcclGroupStart/ sdcclGroupEnd section.
 */
sdcclResult_t sdcclRecv(void *recvbuff, size_t count,
                          sdcclDataType_t datatype, int peer,
                          sdcclComm_t comm, sdcclStream_t stream);

/*
 * One-sided RDMA operations
 *
 * These operations require prior registration via sdcclOneSideRegister /
 * sdcclOneSideSignalRegister. They are only supported on heterogeneous
 * communicators backed by an RDMA-capable net adaptor.
 */

/* RDMA READ: pull size bytes from remote peer's buffer at srcOffset into the
 * local buffer at dstOffset. srcMrIdx / dstMrIdx index the per-window MR
 * handle table populated by sdcclOneSideRegister. */
sdcclResult_t sdcclGet(sdcclComm_t comm, int peer, size_t srcOffset,
                         size_t dstOffset, size_t size, int srcMrIdx,
                         int dstMrIdx);

/* RDMA WRITE + ATOMIC: write size bytes from local srcOffset to remote
 * dstOffset, then atomically increment the remote signal at signalOffset by
 * signalValue. When size == 0, only the signal ATOMIC is posted. */
sdcclResult_t sdcclPutSignal(sdcclComm_t comm, int peer, size_t srcOffset,
                               size_t dstOffset, size_t size,
                               size_t signalOffset, int srcMrIdx, int dstMrIdx,
                               uint64_t signalValue);

/* Signal only: atomically increment remote peer's signal at signalOffset by
 * signalValue (equivalent to sdcclPutSignal with size == 0). */
sdcclResult_t sdcclSignal(sdcclComm_t comm, int peer, size_t signalOffset,
                            uint64_t signalValue);

/* Wait until the local signal buffer at signalOffset reaches the expected
 * value. Uses device-side streamWaitValue64; stream must not be NULL. */
sdcclResult_t sdcclWaitSignal(sdcclComm_t comm, int peer,
                                size_t signalOffset, uint64_t expected,
                                sdcclStream_t stream);

/*
 * Group semantics
 *
 * When managing multiple APUs from a single thread, and since SDCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping SDCCL calls as being part of the same collective operation is done
 * using sdcclGroupStart and sdcclGroupEnd. sdcclGroupStart will enqueue all
 * collective calls until the sdcclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, sdcclGroupEnd only
 * guarantees that the operations are enqueued on the SDCCL streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and sdcclCommInitRank can be used in
 * conjunction of sdcclGroupStart/sdcclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to SDCCL until sdcclGroupEnd will be fused
 * into a single SDCCL operation. Nothing will be started on the SDCCL stream
 * until sdcclGroupEnd.
 */
sdcclResult_t sdcclGroupStart(sdcclComm_t comm);

/*
 * Group End
 *
 * End a group call. Start a fused SDCCL operation consisting of all calls
 * since sdcclGroupStart. Operations on the SDCCL stream depending on the
 * SDCCL operations need to be called after sdcclGroupEnd.
 */
sdcclResult_t sdcclGroupEnd(sdcclComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
