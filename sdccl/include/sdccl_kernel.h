#ifndef SDCCL_KERNEL_H_
#define SDCCL_KERNEL_H_

#include "adaptor.h"
#include "sdccl.h"

#define SDCCL_FIFO_CAPACITY 128
#define sdcclTriggerMask(w) ((w == 64) ? ~0ull : ((1ull << w) - 1))

typedef enum {
  sdcclDevicePrimSend = 0,
  sdcclDevicePrimRecv = 1,
  sdcclDevicePrimTerm = 2,
  sdcclDevicePrimWait = 3,
  sdcclDevicePrimPut = 4,
  sdcclDevicePrimSignal = 5,
  sdcclDevicePrimBarrierSignal = 6,
  sdcclDevicePrimWaitSignal = 7,
  sdcclDevicePrimPutValue = 8,
  sdcclDevicePrimPutSignal = 9,
  sdcclDevicePrimGet = 10
} sdcclDevicePrim;

// Unified buffer index enumeration for fifo
// Layout: [capacity][consumed][produced][terminate][data...]
// Note: sdcclFifoIdxTerminate is only used by sdcclReduceTrigger fifo
typedef enum {
  sdcclFifoIdxCapacity = 0,
  sdcclFifoIdxConsumed = 1,
  sdcclFifoIdxProduced = 2,
  sdcclFifoIdxTerminate = 3,
  sdcclFifoIdxData = 4
} sdcclFifoIndex;

typedef enum {
  sdcclReduceTriggerAvailable = 0,
  sdcclReduceTriggerEnqueued = 1,
  sdcclReduceTriggerInprogress = 2,
  sdcclReduceTriggerComplete = 3
} sdcclReduceTriggerState;

// ==========================================================================
// sdcclDeviceTrigger bit layout (24 bytes = 3 × uint64_t: fst, snd, trd)
//
// trd (word2, control header — written last with valid bit):
//   [63]    valid
//   [62:59] prim (4 bits)
//   [58:39] peerRank (20 bits)
//   [38:36] slotIdx (3 bits, reserved for future multi-FIFO)
//   [35:0]  prim-specific (36 bits)
//
// fst (word0, payload — written first):
//   prim-specific (64 bits)
//
// snd (word1, payload — written second):
//   prim-specific (64 bits)
// ==========================================================================

// Valid bit (trd[63])
constexpr unsigned int sdcclDeviceTriggerOffValid = 63;
constexpr uint64_t sdcclDeviceTriggerValidMask = (1ULL << 63);

// Common header in trd
constexpr unsigned int sdcclDeviceTriggerOffPrim = 59;
constexpr unsigned int sdcclDeviceTriggerBitsPrim = 4;
constexpr unsigned int sdcclDeviceTriggerOffPeerRank = 39;
constexpr unsigned int sdcclDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int sdcclDeviceTriggerOffSlotIdx = 36;
constexpr unsigned int sdcclDeviceTriggerBitsSlotIdx = 3;

// Two-sided Send/Recv: trd prim-specific
//   trd[35:32] = datatype(4), trd[31:0] = count(32)
//   fst = addr(64), snd = 0
constexpr unsigned int sdcclDeviceTriggerOffDatatype = 32;
constexpr unsigned int sdcclDeviceTriggerBitsDatatype = 4;
constexpr unsigned int sdcclDeviceTriggerOffCount = 0;
constexpr unsigned int sdcclDeviceTriggerBitsCount = 32;

// One-sided Put/PutSignal: trd prim-specific
//   trd[35:29] = srcMrIdx(7), trd[28:22] = dstMrIdx(7)
//   PutSignal: trd[21:14] = signalIdx(8), trd[13:0] = unused
//   fst = srcOffset(32)|dstOffset(32), snd =
//   size(32)|signalValue(16)|reserved(16)
constexpr unsigned int sdcclDeviceTriggerOffSrcMrIdx = 29;
constexpr unsigned int sdcclDeviceTriggerBitsSrcMrIdx = 7;
constexpr unsigned int sdcclDeviceTriggerOffDstMrIdx = 22;
constexpr unsigned int sdcclDeviceTriggerBitsDstMrIdx = 7;
constexpr unsigned int sdcclDeviceTriggerOffSignalIdx = 14;
constexpr unsigned int sdcclDeviceTriggerBitsSignalIdx = 8;
// PutSignal signalValue in snd[15:0] (same max as PrimSignal: 16b)
constexpr unsigned int sdcclDeviceTriggerOffSignalValuePut = 0;
constexpr unsigned int sdcclDeviceTriggerBitsSignalValuePut = 16;
// fst offsets for srcOffset/dstOffset (shared with PutValue dstOffset accessor)
constexpr unsigned int sdcclDeviceTriggerOffSrcOffset = 32;
constexpr unsigned int sdcclDeviceTriggerBitsSrcOffset = 32;
constexpr unsigned int sdcclDeviceTriggerOffDstOffset = 0;
constexpr unsigned int sdcclDeviceTriggerBitsDstOffset = 32;
// snd offset for size
constexpr unsigned int sdcclDeviceTriggerOffSize = 32;
constexpr unsigned int sdcclDeviceTriggerBitsSize = 32;

// One-sided PutValue: trd prim-specific
//   trd[28:22] = dstMrIdx(7) (same position as Put/PutSignal dstMrIdx)
//   fst = 0|dstOffset(32) (fst[31:0], same position as Put/PutSignal)
//   snd = value(64)

// Signal/WaitSignal: all in trd prim-specific
//   trd[35:34] = bufferType(2), trd[33:26] = signalIdx(8),
//   trd[25:10] = signalValue/expectedValue(16), trd[9:0] = unused
//   fst = 0, snd = 0
constexpr unsigned int sdcclDeviceTriggerOffBufferType = 34;
constexpr unsigned int sdcclDeviceTriggerBitsBufferType = 2;
constexpr unsigned int sdcclDeviceTriggerOffSignalIdxSig = 26;
constexpr unsigned int sdcclDeviceTriggerBitsSignalIdxSig = 8;
constexpr unsigned int sdcclDeviceTriggerOffSignalValue = 10;
constexpr unsigned int sdcclDeviceTriggerBitsSignalValue =
    16; // max signal value: 2^16 (65535)

constexpr unsigned int sdcclReduceTriggerBitsAddr = 64;
constexpr unsigned int sdcclReduceTriggerOffCount = 0;
constexpr unsigned int sdcclReduceTriggerBitsCount = 32;
constexpr unsigned int sdcclReduceTriggerOffNThreads =
    sdcclReduceTriggerOffCount + sdcclReduceTriggerBitsCount;
constexpr unsigned int sdcclReduceTriggerBitsNThreads = 16;
constexpr unsigned int sdcclReduceTriggerOffDatatype =
    sdcclReduceTriggerOffNThreads + sdcclReduceTriggerBitsNThreads;
constexpr unsigned int sdcclReduceTriggerBitsDatatype = 4;
constexpr unsigned int sdcclReduceTriggerOffRedop =
    sdcclReduceTriggerOffDatatype + sdcclReduceTriggerBitsDatatype;
constexpr unsigned int sdcclReduceTriggerBitsRedop = 4;
constexpr unsigned int sdcclReduceTriggerOffState =
    sdcclReduceTriggerOffRedop + sdcclReduceTriggerBitsRedop;
/* op state: 0 for available, 1 for enqueued, 2 for in-progress, 3 for done */
constexpr unsigned int sdcclReduceTriggerBitsState = 2;
constexpr unsigned int sdcclReduceTriggerBitsFifoReserved = 1;

struct sdcclDeviceTrigger {
  uint64_t fst; // word0 — payload, written first
  uint64_t snd; // word1 — payload, written second
  uint64_t trd; // word2 — control header (valid bit), written last

  // Common accessors (trd common header)
  SDCCL_HOST_DECORATOR uint64_t getPrim();
  SDCCL_HOST_DECORATOR uint64_t getPeerRank();
  SDCCL_HOST_DECORATOR uint64_t getSlotIdx();

  // Two-sided accessors (Send/Recv)
  SDCCL_HOST_DECORATOR uint64_t getAddr();     // fst
  SDCCL_HOST_DECORATOR uint64_t getDatatype(); // trd[35:32]
  SDCCL_HOST_DECORATOR uint64_t getCount();    // trd[31:0]

  // One-sided accessors (Put/PutSignal/PutValue)
  SDCCL_HOST_DECORATOR uint64_t getSrcMrIdx();  // trd[35:29]
  SDCCL_HOST_DECORATOR uint64_t getDstMrIdx();  // trd[28:22]
  SDCCL_HOST_DECORATOR uint64_t getSize();      // snd[63:32]
  SDCCL_HOST_DECORATOR uint64_t getSrcOffset(); // fst[63:32]
  SDCCL_HOST_DECORATOR uint64_t getDstOffset(); // fst[31:0]
  SDCCL_HOST_DECORATOR uint64_t getValue();     // snd (PutValue)
  SDCCL_HOST_DECORATOR uint64_t
  getSignalIdx(); // trd (PutSignal/Signal/WaitSignal)
  SDCCL_HOST_DECORATOR uint64_t getSignalValue();   // trd (Signal)
  SDCCL_HOST_DECORATOR uint64_t getExpectedValue(); // trd (WaitSignal)
  SDCCL_HOST_DECORATOR uint64_t getBufferType();    // trd (Signal/WaitSignal)

  // Term accessor
  SDCCL_HOST_DECORATOR uint64_t getTotalCoops(); // fst (PrimTerm)

  // Backward compat alias
  SDCCL_HOST_DECORATOR uint64_t getType(); // alias for getPrim()
};
typedef sdcclDeviceTrigger *sdcclDeviceTrigger_t;

struct alignas(16) sdcclReduceTrigger {
  uint64_t value[4];

#ifdef COMPILE_KERNEL
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getInput1();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getInput2();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getOutput();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getCount();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getNThreads();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getDatatype();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getRedop();
  SDCCL_DEVICE_INLINE_DECORATOR uint64_t getState();
  SDCCL_DEVICE_INLINE_DECORATOR void setComplete();
#endif
  SDCCL_HOST_DECORATOR void setValue(uint64_t fst, uint64_t snd, uint64_t out,
                                      size_t count, size_t nthreads,
                                      sdcclDataType_t datatype,
                                      sdcclRedOp_t redOp,
                                      sdcclReduceTriggerState state);
  SDCCL_HOST_DECORATOR uint64_t pollState();
  SDCCL_HOST_DECORATOR void setState(int state);
};
typedef sdcclReduceTrigger *sdcclReduceTrigger_t;

struct sdcclFifo {
  // Unified fifo layout: [capacity][consumed][produced][terminate][data...]
  // sdcclDeviceTrigger fifo: terminate slot is reserved but unused
  // sdcclReduceTrigger fifo: terminate slot is used
  // See sdcclFifoIndex enumeration for index values
  uint64_t *buffer;

public:
  sdcclFifo() {}
  ~sdcclFifo() {}
  sdcclResult_t sdcclFifoInit();
  sdcclResult_t sdcclRedFifoInit();
  sdcclResult_t sdcclFifoDestroy();
  sdcclResult_t sdcclRedFifoDestroy();
};
typedef struct sdcclFifo *sdcclFifo_t;

SDCCL_HOST_DECORATOR sdcclResult_t dequeue(void *fifoBuffer,
                                             sdcclDeviceTrigger_t trigger);
SDCCL_HOST_DECORATOR sdcclResult_t enqueue(void *fifoBuffer, uint64_t addr1,
                                             uint64_t addr2, uint64_t addr3,
                                             size_t count, size_t nthreads,
                                             sdcclDataType_t datatype,
                                             sdcclRedOp_t redop, int *idx);
#ifdef COMPILE_KERNEL
SDCCL_DEVICE_INLINE_DECORATOR sdcclResult_t dequeue(volatile uint64_t *buffer,
                                                      int *idx);

SDCCL_DEVICE_DECORATOR size_t
getSdcclDataTypeSizeDevice(sdcclDataType_t dtype);

SDCCL_GLOBAL_DECORATOR void sdcclCollectiveKernel(void *fifoBuffer);
#endif // COMPILE_KERNEL

void sdcclLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, sdcclStream_t stream);

// ==========================================================================
// Device Communicator — Host-side lifecycle management
// ==========================================================================

// Requirements for creating a device communicator.
// Named fields map to NCCL ncclDevCommRequirements (Vendor).
// Naming: NCCL "lsa" → SDCCL "intra", "gin" → "inter", "multimem" →
// "multicast".
struct sdcclDevCommRequirements {
  bool intraMulticast; // → ncclReqs.lsaMultimem

  int barrierCount;      // → ncclReqs.barrierCount (world barrier)
  int intraBarrierCount; // → ncclReqs.lsaBarrierCount
  int interBarrierCount; // → ncclReqs.railGinBarrierCount

  int intraLLA2ABlockCount; // → ncclReqs.lsaLLA2ABlockCount
  int intraLLA2ASlotCount;  // → ncclReqs.lsaLLA2ASlotCount

  bool interForceEnable; // → ncclReqs.ginForceEnable
  int interContextCount; // → ncclReqs.ginContextCount (hint, default 4)
  int interSignalCount;  // → ncclReqs.ginSignalCount (start at id=0)
  int interCounterCount; // → ncclReqs.ginCounterCount (start at id=0)
};

#define SDCCL_DEV_COMM_REQUIREMENTS_INITIALIZER                               \
  {                                                                            \
    false,       /* intraMulticast */                                          \
        0, 0, 0, /* barrierCount, intraBarrierCount, interBarrierCount */      \
        0, 0,    /* intraLLA2ABlockCount, intraLLA2ASlotCount */               \
        false, 4, 0, 0 /* interForceEnable, interContextCount,                 \
                          interSignalCount, interCounterCount */               \
  }

// Network type enumeration (maps to ncclGinType_t on NVIDIA backend).
typedef enum {
  sdcclNetTypeNone = 0,  // → NCCL_GIN_TYPE_NONE
  sdcclNetTypeProxy = 2, // → NCCL_GIN_TYPE_PROXY
  sdcclNetTypeGdaki = 3, // → NCCL_GIN_TYPE_GDAKI
} sdcclNetType_t;

// Communicator properties — host-side queryable attributes.
struct sdcclCommProperties {
  int rank;
  int nRanks;
  int deviceId;            // → ncclCommProperties.cudaDev (platform-neutral)
  bool deviceApiSupport;   // → ncclCommProperties.deviceApiSupport
  bool multicastSupport;   // → ncclCommProperties.multimemSupport
  sdcclNetType_t netType; // → ncclCommProperties.ginType
};
typedef struct sdcclCommProperties sdcclCommProperties_t;

// Query communicator properties.
// Currently returns placeholder defaults; will delegate to backend
// (e.g. ncclCommQueryProperties) when wired through the adaptor layer.
sdcclResult_t sdcclCommQueryProperties(sdcclComm_t comm,
                                         sdcclCommProperties_t *props);

// Forward declarations for types defined in sdccl_device.h.
struct sdcclTeam;
typedef struct sdcclTeam sdcclTeam_t;
struct sdcclDevCommRequirements;
struct sdcclIntraBarrierHandle;
typedef struct sdcclIntraBarrierHandle sdcclIntraBarrierHandle_t;
struct sdcclInterBarrierHandle;
typedef struct sdcclInterBarrierHandle sdcclInterBarrierHandle_t;

// Create barrier requirement handles (stub — returns sdcclNotSupported).
// SDCCL currently uses intraBarrierCount in DevCommCreate directly;
// the resource-handle model will be implemented when needed.
sdcclResult_t
sdcclIntraBarrierCreateRequirement(sdcclTeam_t team, int nBarriers,
                                    sdcclIntraBarrierHandle_t *outHandle,
                                    sdcclDevCommRequirements *outReq);

sdcclResult_t sdcclInterBarrierCreateRequirement(
    sdcclComm_t comm, sdcclTeam_t team, int nBarriers,
    sdcclInterBarrierHandle_t *outHandle, sdcclDevCommRequirements *outReq);

// Opaque handle to a device communicator (host-side lifetime management).
// Internally wraps ncclDevComm on NVIDIA backend (Vendor),
// or IPC barrier state on fallback (Fallback).
typedef struct sdcclDevCommInternal *sdcclDevComm_t;

// Opaque handle to device memory (host-side lifetime management).
// Internally wraps ncclWindow_t on NVIDIA backend (Vendor),
// or IPC peer pointer table on fallback (Fallback).
#ifndef SDCCL_DEV_MEM_T_DEFINED
#define SDCCL_DEV_MEM_T_DEFINED
typedef struct sdcclDevMemInternal *sdcclDevMem_t;
#endif

// Inter-node one-sided AlltoAll (put + waitSignal + flush).
sdcclResult_t sdcclInterOneSidedAlltoAll(sdcclDevMem_t sendMem,
                                           sdcclDevMem_t recvMem, size_t count,
                                           sdcclDataType_t datatype,
                                           sdcclDevComm_t devComm,
                                           sdcclStream_t stream);

// Inter-node two-sided AlltoAll (send/recv + term/wait via FIFO).
sdcclResult_t sdcclInterTwoSidedAlltoAll(sdcclDevMem_t sendMem,
                                           sdcclDevMem_t recvMem, size_t count,
                                           sdcclDataType_t datatype,
                                           sdcclDevComm_t devComm,
                                           sdcclStream_t stream);

// Inter-node Device API test kernels.
// Each kernel tests one API facet; host verifies after streamSynchronize.
sdcclResult_t sdcclInterTestPutSignalInc(sdcclDevMem_t sendMem,
                                           sdcclDevMem_t recvMem, size_t count,
                                           sdcclDataType_t datatype,
                                           sdcclDevComm_t devComm,
                                           sdcclStream_t stream);

sdcclResult_t sdcclInterTestPutSignalAddDecoupled(
    sdcclDevMem_t sendMem, sdcclDevMem_t recvMem, size_t count,
    sdcclDataType_t datatype, sdcclDevComm_t devComm, sdcclStream_t stream);

sdcclResult_t
sdcclInterTestCounterPipeline(sdcclDevMem_t sendMem, sdcclDevMem_t recvMem,
                               size_t count, sdcclDataType_t datatype,
                               sdcclDevComm_t devComm, sdcclStream_t stream,
                               uint64_t *resultBuf);

sdcclResult_t sdcclInterTestPutValue(sdcclDevMem_t recvMem,
                                       sdcclDevComm_t devComm,
                                       sdcclStream_t stream,
                                       size_t putValBase);

sdcclResult_t sdcclInterTestSignal(sdcclDevComm_t devComm,
                                     sdcclStream_t stream);

sdcclResult_t
sdcclInterTestFlushDecouple(sdcclDevMem_t sendMem, sdcclDevMem_t recvMem,
                             size_t count, sdcclDataType_t datatype,
                             sdcclDevComm_t devComm, sdcclStream_t stream);

sdcclResult_t sdcclInterTestFollowShadow(sdcclDevComm_t devComm,
                                           sdcclStream_t stream);

sdcclResult_t sdcclInterTestMeetShadow(sdcclDevComm_t devComm,
                                         sdcclStream_t stream);

sdcclResult_t sdcclInterTestReset(sdcclDevComm_t devComm,
                                    sdcclStream_t stream, uint64_t *resultBuf);

sdcclResult_t sdcclInterTestGet(sdcclDevMem_t sendMem,
                                  sdcclDevMem_t recvMem, size_t count,
                                  sdcclDataType_t datatype,
                                  sdcclDevComm_t devComm,
                                  sdcclStream_t stream);

// Kernel launch configuration constants.
// Also defined in device_api/sdccl_device.h (with same include guard).
#ifndef SDCCL_DEVICE_CTA_COUNT
#define SDCCL_DEVICE_CTA_COUNT 36
#endif
#ifndef SDCCL_DEVICE_THREADS_PER_CTA
#define SDCCL_DEVICE_THREADS_PER_CTA 512
#endif

// Create a device communicator for custom kernel usage.
// On NVIDIA backend (Vendor), internally calls pncclDevCommCreate.
// On fallback (Fallback), sets up IPC-based barrier across intra-node peers.
// The returned handle must be destroyed with sdcclDevCommDestroy(comm,
// devComm).
sdcclResult_t sdcclDevCommCreate(sdcclComm_t comm,
                                   const sdcclDevCommRequirements *reqs,
                                   sdcclDevComm_t *devComm);

// Destroy a device communicator created by sdcclDevCommCreate.
sdcclResult_t sdcclDevCommDestroy(sdcclComm_t comm, sdcclDevComm_t devComm);

// Create a device memory handle for a registered buffer.
// Registration is the caller's responsibility (Decision 7.16):
//   - IPC mode (win=NULL): caller calls sdcclCommRegister first.
//   - Window mode (win!=NULL): caller calls sdcclCommWindowRegister first.
// This function exchanges IPC handles to build peer pointer tables (both modes)
// and stores the window handle (window mode only).
sdcclResult_t sdcclDevMemCreate(sdcclComm_t comm, void *buff, size_t size,
                                  sdcclWindow_t win, sdcclDevMem_t *devMem);

// Destroy a device memory handle created by sdcclDevMemCreate.
sdcclResult_t sdcclDevMemDestroy(sdcclComm_t comm, sdcclDevMem_t devMem);

// Clean up IPC peer pointer table on comm.
// Must be called after homoComm destroy.
// so that cudaFree does not deadlock on device synchronization.
sdcclResult_t sdcclCommCleanupIpcTable(sdcclComm_t comm);

// Deferred device/host-pinned memory free.
// Collects pointers during DevComm/DevMem cleanup.
void sdcclCommDeferFree(sdcclComm_t comm, void *ptr, int memType);
sdcclResult_t sdcclCommDrainDeferredFrees(sdcclComm_t comm);

// One-sided data buffer registration.
// Must be called after sdcclCommInitRank and before one-sided operations.
sdcclResult_t sdcclOneSideRegister(const sdcclComm_t comm, void *buff,
                                     size_t size);
// Release data buffer resources (MR, network connections, handle arrays).
sdcclResult_t sdcclOneSideDeregister(const sdcclComm_t comm);

// One-sided signal buffer registration (GPU memory with FORCE_SO).
// Must be called after sdcclCommInitRank and before one-sided operations.
sdcclResult_t sdcclOneSideSignalRegister(const sdcclComm_t comm, void *buff,
                                           size_t size);
// Release signal buffer resources (MR, network connections, handle arrays).
sdcclResult_t sdcclOneSideSignalDeregister(const sdcclComm_t comm);

// One-sided staging buffer registration (host-pinned memory for PutValue).
// Must be called after sdcclOneSideSignalRegister (requires full-mesh
// connections).
sdcclResult_t sdcclOneSideStagingRegister(const sdcclComm_t comm, void *buff,
                                            size_t size);
// Release staging buffer MR resources.
sdcclResult_t sdcclOneSideStagingDeregister(const sdcclComm_t comm);

// One-sided barrier MR registration (host-pinned memory for inter-node
// barrier). Collective: ALL ranks must call. Leaders pass recvComm+buff,
// non-leaders pass NULL.
sdcclResult_t
sdcclOneSideBarrierRegister(const sdcclComm_t comm, void *recvComm,
                             void *buff, size_t size,
                             struct sdcclOneSideHandleInfo **outInfo);
// Release barrier MR and free handle info.
sdcclResult_t
sdcclOneSideBarrierDeregister(const sdcclComm_t comm,
                               struct sdcclOneSideHandleInfo *info);

// Intra-node AllReduce using SDCCL Device API.
// The caller provides a registered buffer (via sdcclDevMemCreate)
// already containing the input data.  The kernel runs an in-place
// AllReduce across all intra-node GPUs.
// devComm must be created via sdcclDevCommCreate beforehand.
sdcclResult_t sdcclIntraAllReduce(sdcclDevMem_t devMem, size_t count,
                                    sdcclDataType_t datatype,
                                    sdcclDevComm_t devComm,
                                    sdcclStream_t stream);

#endif
