/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_ADAPTOR_DEVICE_UTILS_H_
#define SDCCL_ADAPTOR_DEVICE_UTILS_H_

// Device compiler detection — defined when any GPU device compiler is active.
// Extend with __ASCEND_CC__ etc. as new platforms are added.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SDCCL_DEVICE_COMPILE 1
#endif

// Suppress unused-variable warnings for static arrays in headers
#define SDCCL_MAYBE_UNUSED __attribute__((unused))

#ifdef USE_NVIDIA_ADAPTOR
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
// Compiling with nvcc — full CUDA qualifiers
#define SDCCL_HOST_DECORATOR __host__
#define SDCCL_DEVICE_DECORATOR __device__
#define SDCCL_GLOBAL_DECORATOR __global__
#define SDCCL_DEVICE_INLINE_DECORATOR __forceinline__ __device__
#define SDCCL_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#define SDCCL_DEVICE_CONSTANT_DECORATOR __device__ __constant__
#define SDCCL_DEVICE_THREAD_FENCE __threadfence_system
#define SDCCL_DEVICE_SYNC_THREADS __syncthreads
#define SDCCL_THREAD_IDX_X threadIdx.x
#define SDCCL_BLOCK_IDX_X blockIdx.x
#define SDCCL_BLOCK_DIM_X blockDim.x
#define SDCCL_GRID_DIM_X gridDim.x

// SIMT lockstep width (32 lanes on NVIDIA/CUDA)
#define SDCCL_SIMT_WIDTH 32
#define SDCCL_SHARED __shared__
#else
// Host compiler (g++/clang++) on NVIDIA platform — no CUDA qualifiers
#define SDCCL_HOST_DECORATOR
#define SDCCL_DEVICE_DECORATOR
#define SDCCL_GLOBAL_DECORATOR
#define SDCCL_DEVICE_INLINE_DECORATOR inline
#define SDCCL_HOST_DEVICE_INLINE inline
#define SDCCL_DEVICE_CONSTANT_DECORATOR
#define SDCCL_DEVICE_THREAD_FENCE() ((void)0)
#define SDCCL_DEVICE_SYNC_THREADS() ((void)0)
#define SDCCL_THREAD_IDX_X 0
#define SDCCL_BLOCK_IDX_X 0
#define SDCCL_BLOCK_DIM_X 1
#define SDCCL_GRID_DIM_X 1

// SIMT width (same as device, for template instantiation)
#define SDCCL_SIMT_WIDTH 32
#define SDCCL_SHARED static
#endif // __CUDACC__

// CUDA runtime macros — available from both nvcc and host compiler
#define SDCCL_DEVICE_STREAM_PTR cudaStream_t *

#else
// Non-NVIDIA platform
#define SDCCL_HOST_DECORATOR
#define SDCCL_DEVICE_DECORATOR
#define SDCCL_GLOBAL_DECORATOR
#define SDCCL_DEVICE_INLINE_DECORATOR
#define SDCCL_HOST_DEVICE_INLINE inline
#define SDCCL_DEVICE_CONSTANT_DECORATOR
#define SDCCL_DEVICE_STREAM_PTR
#define SDCCL_DEVICE_THREAD_FENCE() ((void)0)
#define SDCCL_DEVICE_SYNC_THREADS() ((void)0)
#define SDCCL_THREAD_IDX_X 0
#define SDCCL_BLOCK_IDX_X 0
#define SDCCL_BLOCK_DIM_X 1
#define SDCCL_GRID_DIM_X 1
#endif

#endif // SDCCL_ADAPTOR_DEVICE_UTILS_H_
