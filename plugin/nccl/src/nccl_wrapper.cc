/*************************************************************************
 * Copyright (c) 2025, BAAI. All rights reserved.
 *
 * NCCL API wrapper that delegates to SDCCL internally.
 * This allows frameworks using NCCL to transparently use SDCCL
 * without code modifications.
 ************************************************************************/

#include "sdccl.h"
#include "nccl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>

/* ──────────────────────────────────────────────────────────────────────
 * Minimum NCCL version check
 * ────────────────────────────────────────────────────────────────────── */

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 21, 0)
#error                                                                         \
    "NCCL version 2.21.0 or later is required to build the SDCCL NCCL wrapper"
#endif

/* ──────────────────────────────────────────────────────────────────────
 * Compile-time path to the real NCCL shared library.
 * Set via -DREAL_NCCL_PATH=\"...\" in the Makefile.
 * ────────────────────────────────────────────────────────────────────── */

#ifndef REAL_NCCL_PATH
#error                                                                         \
    "REAL_NCCL_PATH must be defined at compile time (e.g. -DREAL_NCCL_PATH=\\\"/usr/local/nccl/lib/libnccl.so.2\\\")"
#endif

/* ──────────────────────────────────────────────────────────────────────
 * TLS recursive guard
 * A thread-local recursive guard prevents infinite recursion when
 * SDCCL's NCCL adaptor calls back into nccl* symbols.  On re-entry
 * the call is forwarded to the real NCCL loaded via dlopen.
 * ────────────────────────────────────────────────────────────────────── */

static thread_local bool inWrapper = false;

struct recursionGuard {
  bool &flag;
  recursionGuard(bool &f) : flag(f) { flag = true; }
  ~recursionGuard() { flag = false; }
  recursionGuard(const recursionGuard &) = delete;
  recursionGuard &operator=(const recursionGuard &) = delete;
};

/* ──────────────────────────────────────────────────────────────────────
 * Real NCCL library loaded via dlopen
 * ────────────────────────────────────────────────────────────────────── */

struct realNccl {
  void *handle;

  /* Function pointers — one per NCCL API that SDCCL may call back into */
  ncclResult_t (*ncclGetVersion)(int *);
  ncclResult_t (*ncclGetUniqueId)(ncclUniqueId *);
  const char *(*ncclGetErrorString)(ncclResult_t);
  const char *(*ncclGetLastError)(ncclComm_t);
  ncclResult_t (*ncclCommInitRank)(ncclComm_t *, int, ncclUniqueId, int);
  ncclResult_t (*ncclCommFinalize)(ncclComm_t);
  ncclResult_t (*ncclCommDestroy)(ncclComm_t);
  ncclResult_t (*ncclCommAbort)(ncclComm_t);
  ncclResult_t (*ncclCommCount)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommCuDevice)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommUserRank)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommGetAsyncError)(ncclComm_t, ncclResult_t *);
  ncclResult_t (*ncclMemAlloc)(void **, size_t);
  ncclResult_t (*ncclMemFree)(void *);
  ncclResult_t (*ncclCommRegister)(const ncclComm_t, void *, size_t, void **);
  ncclResult_t (*ncclCommDeregister)(const ncclComm_t, void *);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
  ncclResult_t (*ncclCommWindowRegister)(ncclComm_t, void *, size_t,
                                         ncclWindow_t *, int);
  ncclResult_t (*ncclCommWindowDeregister)(ncclComm_t, ncclWindow_t);
#endif
  ncclResult_t (*ncclReduce)(const void *, void *, size_t, ncclDataType_t,
                             ncclRedOp_t, int, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclBroadcast)(const void *, void *, size_t, ncclDataType_t,
                                int, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclAllReduce)(const void *, void *, size_t, ncclDataType_t,
                                ncclRedOp_t, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclReduceScatter)(const void *, void *, size_t,
                                    ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                    cudaStream_t);
  ncclResult_t (*ncclAllGather)(const void *, void *, size_t, ncclDataType_t,
                                ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclSend)(const void *, size_t, ncclDataType_t, int,
                           ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclRecv)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                           cudaStream_t);
  ncclResult_t (*ncclGroupStart)();
  ncclResult_t (*ncclGroupEnd)();
};

static realNccl *realNcclInstance = nullptr;
static std::once_flag realNcclOnce;

static void initRealNccl() {
  realNccl *r = new realNccl();
  r->handle = dlopen(REAL_NCCL_PATH, RTLD_LAZY | RTLD_LOCAL);
  if (!r->handle) {
    fprintf(stderr,
            "SDCCL NCCL wrapper: failed to dlopen real NCCL at %s: %s\n",
            REAL_NCCL_PATH, dlerror());
    delete r;
    return;
  }

#define LOAD_SYM(name)                                                         \
  r->name = reinterpret_cast<decltype(r->name)>(dlsym(r->handle, #name));      \
  if (!r->name) {                                                              \
    fprintf(stderr, "SDCCL NCCL wrapper: dlsym failed for " #name ": %s\n",   \
            dlerror());                                                        \
  }

  LOAD_SYM(ncclGetVersion)
  LOAD_SYM(ncclGetUniqueId)
  LOAD_SYM(ncclGetErrorString)
  LOAD_SYM(ncclGetLastError)
  LOAD_SYM(ncclCommInitRank)
  LOAD_SYM(ncclCommFinalize)
  LOAD_SYM(ncclCommDestroy)
  LOAD_SYM(ncclCommAbort)
  LOAD_SYM(ncclCommCount)
  LOAD_SYM(ncclCommCuDevice)
  LOAD_SYM(ncclCommUserRank)
  LOAD_SYM(ncclCommGetAsyncError)
  LOAD_SYM(ncclMemAlloc)
  LOAD_SYM(ncclMemFree)
  LOAD_SYM(ncclCommRegister)
  LOAD_SYM(ncclCommDeregister)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
  LOAD_SYM(ncclCommWindowRegister)
  LOAD_SYM(ncclCommWindowDeregister)
#endif
  LOAD_SYM(ncclReduce)
  LOAD_SYM(ncclBroadcast)
  LOAD_SYM(ncclAllReduce)
  LOAD_SYM(ncclReduceScatter)
  LOAD_SYM(ncclAllGather)
  LOAD_SYM(ncclSend)
  LOAD_SYM(ncclRecv)
  LOAD_SYM(ncclGroupStart)
  LOAD_SYM(ncclGroupEnd)

#undef LOAD_SYM

  realNcclInstance = r;
}

static realNccl &getRealNccl() {
  std::call_once(realNcclOnce, initRealNccl);
  return *realNcclInstance;
}

/* ──────────────────────────────────────────────────────────────────────
 * Internal ncclComm struct
 * ────────────────────────────────────────────────────────────────────── */

struct ncclComm {
  sdcclHandlerGroup_t handler; // handler group (uniqueId, comm, devHandle)
  int rank;
  int nranks;
  sdcclResult_t asyncError;
};

/* ──────────────────────────────────────────────────────────────────────
 * Helper: wrap a cudaStream_t into a temporary sdcclStream_t
 *
 * SDCCL wraps CUDA streams in `struct sdcclStream { cudaStream_t base; }`.
 * The struct definition lives in the adaptor headers (not public), so we
 * define a layout-compatible version here for stream wrapping.
 * We allocate a temporary wrapper, call the SDCCL API, then free it.
 * ────────────────────────────────────────────────────────────────────── */

struct sdcclStream {
  cudaStream_t base;
};

struct SdcclStreamWrapper {
  sdcclStream_t stream;

  SdcclStreamWrapper(cudaStream_t cudaStream) {
    stream = (sdcclStream_t)malloc(sizeof(struct sdcclStream));
    if (stream) {
      stream->base = cudaStream;
    }
  }

  ~SdcclStreamWrapper() { free(stream); }

  SdcclStreamWrapper(const SdcclStreamWrapper &) = delete;
  SdcclStreamWrapper &operator=(const SdcclStreamWrapper &) = delete;
};

/* ──────────────────────────────────────────────────────────────────────
 * Helper: ncclDataType_t -> sdcclDataType_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toSdcclDataType(ncclDataType_t ncclType,
                                     sdcclDataType_t *sdcclType) {
  switch (ncclType) {
    case ncclInt8:
      *sdcclType = sdcclInt8;
      return ncclSuccess;
    case ncclUint8:
      *sdcclType = sdcclUint8;
      return ncclSuccess;
    case ncclInt32:
      *sdcclType = sdcclInt32;
      return ncclSuccess;
    case ncclUint32:
      *sdcclType = sdcclUint32;
      return ncclSuccess;
    case ncclInt64:
      *sdcclType = sdcclInt64;
      return ncclSuccess;
    case ncclUint64:
      *sdcclType = sdcclUint64;
      return ncclSuccess;
    case ncclFloat16:
      *sdcclType = sdcclFloat16;
      return ncclSuccess;
    case ncclFloat32:
      *sdcclType = sdcclFloat32;
      return ncclSuccess;
    case ncclFloat64:
      *sdcclType = sdcclFloat64;
      return ncclSuccess;
    case ncclBfloat16:
      *sdcclType = sdcclBfloat16;
      return ncclSuccess;
    default:
      /* ncclFloat8e4m3 (10), ncclFloat8e5m2 (11) have no SDCCL equivalent */
      return ncclInvalidUsage;
  }
}

/* ──────────────────────────────────────────────────────────────────────
 * Helper: ncclRedOp_t -> sdcclRedOp_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toSdcclRedOp(ncclRedOp_t ncclOp, sdcclRedOp_t *sdcclOp) {
  if ((int)ncclOp >= 0 && (int)ncclOp < (int)ncclNumOps) {
    *sdcclOp = (sdcclRedOp_t)(int)ncclOp;
    return ncclSuccess;
  }
  /* Custom / dynamic reduction ops are not supported */
  return ncclInvalidUsage;
}

/* ──────────────────────────────────────────────────────────────────────
 * Helper: sdcclResult_t -> ncclResult_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toNcclResult(sdcclResult_t res) {
  switch (res) {
    case sdcclSuccess:
      return ncclSuccess;
    case sdcclUnhandledDeviceError:
      return ncclUnhandledCudaError;
    case sdcclSystemError:
      return ncclSystemError;
    case sdcclInternalError:
      return ncclInternalError;
    case sdcclInvalidArgument:
      return ncclInvalidArgument;
    case sdcclInvalidUsage:
      return ncclInvalidUsage;
    case sdcclRemoteError:
      return ncclRemoteError;
    case sdcclInProgress:
      return ncclInProgress;
    default:
      return ncclInternalError;
  }
}

/* ──────────────────────────────────────────────────────────────────────
 * Version / Error String APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetVersion(int *version) {
  if (inWrapper) {
    return getRealNccl().ncclGetVersion(version);
  }
  recursionGuard guard(inWrapper);
  if (version == nullptr)
    return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t result) {
  if (inWrapper) {
    return getRealNccl().ncclGetErrorString(result);
  }
  recursionGuard guard(inWrapper);
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=WARN for details)";
    case ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=WARN for details)";
    case ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

const char *ncclGetLastError(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclGetLastError(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return nullptr;
  return sdcclGetLastError(comm->handler->comm);
}

/* ──────────────────────────────────────────────────────────────────────
 * UniqueId
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
  if (inWrapper) {
    return getRealNccl().ncclGetUniqueId(uniqueId);
  }
  recursionGuard guard(inWrapper);
  if (uniqueId == nullptr)
    return ncclInvalidArgument;

  sdcclUniqueId_t sdcclId = nullptr;
  sdcclResult_t res = sdcclGetUniqueId(&sdcclId);
  if (res != sdcclSuccess) {
    return toNcclResult(res);
  }

  /* sdcclBootstrapHandle fits in NCCL_UNIQUE_ID_BYTES (128).
   * Copy the first 128 bytes of the 256-byte sdcclUniqueId. */
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId->internal, sdcclId->internal, NCCL_UNIQUE_ID_BYTES);

  free(sdcclId);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Init / Finalize / Destroy / Abort
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                              int rank) {
  if (inWrapper) {
    return getRealNccl().ncclCommInitRank(comm, nranks, commId, rank);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;

  ncclComm_t c = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
  if (c == nullptr)
    return ncclSystemError;

  /* Init SDCCL handler group */
  sdcclResult_t res = sdcclHandleInit(&c->handler);
  if (res != sdcclSuccess) {
    free(c);
    return toNcclResult(res);
  }

  /* Reconstruct a sdcclUniqueId from the NCCL 128-byte id.
   * Zero-init the full 256-byte struct, then copy in the 128-byte handle. */
  memset(c->handler->uniqueId, 0, sizeof(sdcclUniqueId));
  memcpy(c->handler->uniqueId->internal, commId.internal, NCCL_UNIQUE_ID_BYTES);

  /* Init the SDCCL communicator */
  res =
      sdcclCommInitRank(&c->handler->comm, nranks, c->handler->uniqueId, rank);
  if (res != sdcclSuccess) {
    sdcclHandleFree(c->handler);
    free(c);
    return toNcclResult(res);
  }

  c->rank = rank;
  c->nranks = nranks;
  c->asyncError = sdcclSuccess;

  *comm = c;
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                    ncclUniqueId commId, int rank,
                                    ncclConfig_t *config) {
  /* Config fields are NCCL-specific; SDCCL has no equivalent.
   * Delegate to the non-config version (which handles the guard). */
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommFinalize(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommFinalize(comm->handler->comm));
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommDestroy(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclResult_t res = sdcclCommDestroy(comm->handler->comm);
  /* sdcclCommDestroy frees internal resources but not the comm struct.
   * sdcclHandleFree will free the comm struct pointer along with handler. */
  sdcclHandleFree(comm->handler);
  free(comm);
  return toNcclResult(res);
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommAbort(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclResult_t res = sdcclCommAbort(comm->handler->comm);
  sdcclHandleFree(comm->handler);
  free(comm);
  return toNcclResult(res);
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Query APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (inWrapper) {
    return getRealNccl().ncclCommCount(comm, count);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommCount(comm->handler->comm, count));
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  if (inWrapper) {
    return getRealNccl().ncclCommCuDevice(comm, device);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommGetDeviceNumber(comm->handler->comm, device));
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (inWrapper) {
    return getRealNccl().ncclCommUserRank(comm, rank);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommUserRank(comm->handler->comm, rank));
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (inWrapper) {
    return getRealNccl().ncclCommGetAsyncError(comm, asyncError);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclResult_t sdcclAsync;
  sdcclResult_t res =
      sdcclCommGetAsyncError(comm->handler->comm, &sdcclAsync);
  if (res != sdcclSuccess)
    return toNcclResult(res);
  *asyncError = toNcclResult(sdcclAsync);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Buffer Registration
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size,
                              void **handle) {
  if (inWrapper) {
    return getRealNccl().ncclCommRegister(comm, buff, size, handle);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(
      sdcclCommRegister(comm->handler->comm, buff, size, handle));
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  if (inWrapper) {
    return getRealNccl().ncclCommDeregister(comm, handle);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommDeregister(comm->handler->comm, handle));
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size,
                                    ncclWindow_t *win, int winFlags) {
  if (inWrapper) {
    return getRealNccl().ncclCommWindowRegister(comm, buff, size, win,
                                                winFlags);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(sdcclCommWindowRegister(
      comm->handler->comm, buff, size, (sdcclWindow_t *)win, winFlags));
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  if (inWrapper) {
    return getRealNccl().ncclCommWindowDeregister(comm, win);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(
      sdcclCommWindowDeregister(comm->handler->comm, (sdcclWindow_t)win));
}
#endif

/* ──────────────────────────────────────────────────────────────────────
 * Memory Allocation
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (inWrapper) {
    return getRealNccl().ncclMemAlloc(ptr, size);
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(sdcclMemAlloc(ptr, size));
}

ncclResult_t ncclMemFree(void *ptr) {
  if (inWrapper) {
    return getRealNccl().ncclMemFree(ptr);
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(sdcclMemFree(ptr));
}

/* ──────────────────────────────────────────────────────────────────────
 * Group Semantics
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGroupStart() {
  if (inWrapper) {
    return getRealNccl().ncclGroupStart();
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(sdcclGroupStart(nullptr));
}

ncclResult_t ncclGroupEnd() {
  if (inWrapper) {
    return getRealNccl().ncclGroupEnd();
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(sdcclGroupEnd(nullptr));
}

/* ──────────────────────────────────────────────────────────────────────
 * Collective Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclAllReduce(sendbuff, recvbuff, count, datatype, op,
                                       comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  sdcclRedOp_t fOp;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toSdcclRedOp(op, &fOp)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(sdcclAllReduce(sendbuff, recvbuff, count, fType, fOp,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclBroadcast(sendbuff, recvbuff, count, datatype,
                                       root, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(sdcclBroadcast(sendbuff, recvbuff, count, fType, root,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclReduce(sendbuff, recvbuff, count, datatype, op,
                                    root, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  sdcclRedOp_t fOp;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toSdcclRedOp(op, &fOp)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(sdcclReduce(sendbuff, recvbuff, count, fType, fOp, root,
                                   comm->handler->comm, sw.stream));
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
                                       comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(sdcclAllGather(sendbuff, recvbuff, sendcount, fType,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                           datatype, op, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  sdcclRedOp_t fOp;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toSdcclRedOp(op, &fOp)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(sdcclReduceScatter(sendbuff, recvbuff, recvcount, fType,
                                          fOp, comm->handler->comm, sw.stream));
}

/* ──────────────────────────────────────────────────────────────────────
 * Point-to-Point Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclSend(const void *sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclSend(sendbuff, count, datatype, peer, comm,
                                  stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(
      sdcclSend(sendbuff, count, fType, peer, comm->handler->comm, sw.stream));
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclRecv(recvbuff, count, datatype, peer, comm,
                                  stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  sdcclDataType_t fType;
  ncclResult_t r;
  if ((r = toSdcclDataType(datatype, &fType)) != ncclSuccess)
    return r;
  SdcclStreamWrapper sw(stream);
  return toNcclResult(
      sdcclRecv(recvbuff, count, fType, peer, comm->handler->comm, sw.stream));
}

/* ──────────────────────────────────────────────────────────────────────
 * Unsupported APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclInvalidUsage;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  return ncclInvalidUsage;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm, ncclConfig_t *config) {
  return ncclInvalidUsage;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList,
                            int excludeRanksCount, ncclComm_t *newcomm,
                            ncclConfig_t *config, int shrinkFlags) {
  return ncclInvalidUsage;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 4)
ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks,
                                      int myrank, int nId,
                                      ncclUniqueId *commIds,
                                      ncclConfig_t *config) {
  return ncclInvalidUsage;
}
#endif

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  return ncclInvalidUsage;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclInvalidUsage;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 22, 3)
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
  return ncclInvalidUsage;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 3)
void ncclResetDebugInit() { /* Deprecated in NCCL, no-op */
}
#endif
