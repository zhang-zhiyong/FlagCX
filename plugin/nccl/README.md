# NCCL Wrapper for SDCCL

This plugin builds a drop-in `libnccl.so` that intercepts NCCL API calls and routes them through SDCCL. Any application or framework that uses NCCL (e.g. PyTorch, DeepSpeed) can transparently use SDCCL without code changes.

## How It Works

The wrapper exports public NCCL symbols matching the version of the system-installed NCCL headers. When an application calls an NCCL function (e.g. `ncclAllReduce`), the wrapper translates the call into the equivalent SDCCL API.

Because SDCCL's internal NCCL adaptor itself calls back into `nccl*` functions, the wrapper uses a **thread-local recursive guard** to prevent infinite recursion. On the first call, the wrapper delegates to SDCCL. If SDCCL re-enters an `nccl*` symbol on the same thread, the wrapper forwards the call to the **real** NCCL library (loaded via `dlopen` at runtime).

```
Application
    |
    v
ncclAllReduce()          <-- wrapper (first entry, delegates to SDCCL)
    |
    v
SDCCL
    |
    v
ncclAllReduce()          <-- wrapper (re-entry detected, forwards to real NCCL)
    |
    v
Real NCCL
```

## Prerequisites

- **SDCCL** built and installed (the wrapper links against `libsdccl.so`)
- **CUDA toolkit** (for `cuda_runtime.h` and `libcudart`)
- **Real NCCL >= 2.21.0** installed somewhere on the system (the wrapper loads it via `dlopen` at runtime). Supported NCCL versions: **2.21 through 2.27**. The wrapper uses compile-time version guards (`NCCL_VERSION_CODE`) to adapt to the installed NCCL version — APIs introduced after 2.21.0 are only built when the system headers support them.

## Building

```bash
cd plugin/nccl
make NCCL_HOME=/path/to/nccl CUDA_HOME=/path/to/cuda
```

### Build Variables

| Variable | Default | Description |
|---|---|---|
| `SDCCL_HOME` | `../../` (auto-detected) | Path to the SDCCL source/build tree |
| `CUDA_HOME` | `/usr/local/cuda` | Path to the CUDA toolkit |
| `NCCL_HOME` | `/usr/local/nccl` | Path to the real NCCL installation |
| `REAL_NCCL_LIB` | `$(NCCL_HOME)/lib/libnccl.so.2` | Full path to the real `libnccl.so.2` |

On systems where NCCL is not installed under a standard `NCCL_HOME/lib/` layout, set `REAL_NCCL_LIB` directly:

```bash
# Debian/Ubuntu with system NCCL
make REAL_NCCL_LIB=/usr/lib/x86_64-linux-gnu/libnccl.so.2

# Custom location
make REAL_NCCL_LIB=/opt/libs/libnccl.so.2
```

Build outputs go to `build/`:

```
build/
  lib/
    libnccl.so            # the wrapper library
```

## Usage

Use `LD_PRELOAD` to inject the wrapper library so that it intercepts all NCCL calls made by the application:

```bash
LD_PRELOAD=/path/to/plugin/nccl/build/lib/libnccl.so.2 python train.py
```

You can also point `LD_LIBRARY_PATH` to the wrapper's lib directory so that the dynamic linker picks it up before the real NCCL:

```bash
export LD_LIBRARY_PATH=/path/to/plugin/nccl/build/lib:$LD_LIBRARY_PATH
python train.py
```

Make sure the SDCCL library is also loadable at runtime (it is set via `rpath` during linking, but you may need to add it to `LD_LIBRARY_PATH` if you moved things around):

```bash
export LD_LIBRARY_PATH=/path/to/sdccl/build/lib:/path/to/plugin/nccl/build/lib:$LD_LIBRARY_PATH
python train.py
```

## Supported APIs

The following NCCL APIs are intercepted and routed through SDCCL:

- **Version / Error**: `ncclGetVersion`, `ncclGetErrorString`, `ncclGetLastError`
- **Unique ID**: `ncclGetUniqueId`
- **Communicator**: `ncclCommInitRank`, `ncclCommInitRankConfig`, `ncclCommFinalize`, `ncclCommDestroy`, `ncclCommAbort`
- **Communicator Query**: `ncclCommCount`, `ncclCommCuDevice`, `ncclCommUserRank`, `ncclCommGetAsyncError`
- **Memory**: `ncclMemAlloc`, `ncclMemFree`
- **Buffer Registration**: `ncclCommRegister`, `ncclCommDeregister`, `ncclCommWindowRegister`, `ncclCommWindowDeregister`
- **Collectives**: `ncclAllReduce`, `ncclBroadcast`, `ncclReduce`, `ncclAllGather`, `ncclReduceScatter`
- **Point-to-Point**: `ncclSend`, `ncclRecv`
- **Group**: `ncclGroupStart`, `ncclGroupEnd`

The following APIs are exported but return `ncclInvalidUsage` (no SDCCL equivalent):

`ncclBcast`, `ncclCommInitAll`, `ncclCommSplit`, `ncclRedOpCreatePreMulSum`, `ncclRedOpDestroy`

The following APIs are version-gated and only built when the system NCCL headers are new enough:

| API | Minimum NCCL version |
|---|---|
| `ncclGroupSimulateEnd` | 2.22.3 |
| `ncclCommInitRankScalable` | 2.23.4 |
| `ncclResetDebugInit` | 2.24.3 |
| `ncclCommShrink` | 2.27.3 |
| `ncclCommWindowRegister`, `ncclCommWindowDeregister` | 2.27.3 |
