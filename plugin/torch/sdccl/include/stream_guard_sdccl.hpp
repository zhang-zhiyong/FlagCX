/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/
#pragma once

#include "sdccl.h"

#ifdef USE_NVIDIA_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <cuda_runtime.h>
#elif USE_ASCEND_ADAPTOR
#include "torch_npu/csrc/core/npu/NPUStream.h"
#elif USE_ILUVATAR_COREX_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <cuda_runtime.h>
#elif USE_CAMBRICON_ADAPTOR
#include "framework/core/MLUEvent.h"
#include "framework/core/MLUStream.h"
#include "framework/core/stream_guard.h"
#elif USE_METAX_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <cuda_runtime.h>
#elif USE_MUSA_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <musa_runtime.h>
#include <torch_musa/csrc/core/GuardImpl.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#elif USE_DU_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <cuda_runtime.h>
#elif USE_KUNLUNXIN_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <cuda_runtime.h>
#elif USE_AMD_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/impl/HIPGuardImpl.h>
#include <hip/hip_runtime.h>
#elif USE_TSM_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <tx_runtime.h>
#elif USE_ENFLAME_ADAPTOR
#include <c10/core/impl/InlineStreamGuard.h>
#include <gcu/gcu_guard.h>
#include <gcu/gcu_stream.h>
#include <tops/tops_runtime_api.h>
#endif

namespace c10d {

class sdcclStreamGuard {
public:
  // No default constructor
  explicit sdcclStreamGuard() = delete;
  explicit sdcclStreamGuard(sdcclStream_t stream, const int deviceId)
      : originalStream_(stream), currentStream_(nullptr), deviceId_(deviceId),
#ifdef USE_NVIDIA_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))
#elif USE_ILUVATAR_COREX_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))
#elif USE_CAMBRICON_ADAPTOR
        guard_(
            torch_mlu::getStreamFromExternal(*(cnrtQueue_t *)stream, deviceId))
#elif USE_METAX_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))
#elif USE_MUSA_ADAPTOR
        guard_(
            at::musa::getStreamFromExternal(*(musaStream_t *)stream, deviceId))
#elif USE_DU_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))
#elif USE_KUNLUNXIN_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))
#elif USE_ASCEND_ADAPTOR
        guard_(c10_npu::getNPUStreamFromPool(deviceId))
#elif USE_AMD_ADAPTOR
        guard_(at::hip::getStreamFromExternal(*(hipStream_t *)stream, deviceId))
#elif USE_TSM_ADAPTOR
        guard_(
            torch_txda::getStreamFromExternal(*(txStream_t *)stream, deviceId))
#elif USE_ENFLAME_ADAPTOR
        guard_(
            torch_gcu::getStreamFromExternal(*(topsStream_t *)stream, deviceId))
#endif
  {
  }
  ~sdcclStreamGuard() = default;

  // No copy
  sdcclStreamGuard(const sdcclStreamGuard &) = delete;
  sdcclStreamGuard &operator=(const sdcclStreamGuard &) = delete;

  // No move
  sdcclStreamGuard(sdcclStreamGuard &&) = delete;
  sdcclStreamGuard &operator=(sdcclStreamGuard &&) = delete;

  void reset_stream(sdcclStream_t stream) {
#ifdef USE_NVIDIA_ADAPTOR
    guard_.reset_stream(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId_));
#elif USE_ILUVATAR_COREX_ADAPTOR
    guard_.reset_stream(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId_));
#elif USE_CAMBRICON_ADAPTOR
    guard_.reset_stream(
        torch_mlu::getStreamFromExternal(*(cnrtQueue_t *)stream, deviceId_));
#elif USE_METAX_ADAPTOR
    guard_.reset_stream(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId_));
#elif USE_MUSA_ADAPTOR
    guard_.reset_stream(
        at::musa::getStreamFromExternal(*(musaStream_t *)stream, deviceId_));
#elif USE_DU_ADAPTOR
    guard_.reset_stream(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId_));
#elif USE_KUNLUNXIN_ADAPTOR
    guard_.reset_stream(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId_));
#elif USE_ASCEND_ADAPTOR
    guard_ = c10_npu::getNPUStreamFromPool(deviceId_);
#elif USE_AMD_ADAPTOR
    guard_.reset_stream(
        at::hip::getStreamFromExternal(*(hipStream_t *)stream, deviceId_));
#elif USE_TSM_ADAPTOR
    guard_.reset_stream(
        torch_txda::getStreamFromExternal(*(txStream_t *)stream, deviceId_));
#elif USE_ENFLAME_ADAPTOR
    guard_.reset_stream(
        torch_gcu::getStreamFromExternal(*(topsStream_t *)stream, deviceId_));
#endif
    currentStream_ = stream;
  }

  sdcclStream_t original_stream() const { return originalStream_; }

  sdcclStream_t current_stream() const { return currentStream_; }

private:
  sdcclStream_t originalStream_;
  sdcclStream_t currentStream_;
  int deviceId_;
#ifdef USE_NVIDIA_ADAPTOR
  c10::cuda::CUDAStreamGuard guard_;
#elif USE_ILUVATAR_COREX_ADAPTOR
  c10::cuda::CUDAStreamGuard guard_;
#elif USE_CAMBRICON_ADAPTOR
  torch_mlu::mlu::MLUStreamGuard guard_;
#elif USE_METAX_ADAPTOR
  c10::cuda::CUDAStreamGuard guard_;
#elif USE_MUSA_ADAPTOR
  c10::musa::MUSAStreamGuard guard_;
#elif USE_DU_ADAPTOR
  c10::cuda::CUDAStreamGuard guard_;
#elif USE_KUNLUNXIN_ADAPTOR
  c10::cuda::CUDAStreamGuard guard_;
#elif USE_ASCEND_ADAPTOR
  c10_npu::NPUStream guard_;
#elif USE_AMD_ADAPTOR
  c10::hip::HIPStreamGuard guard_;
#elif USE_TSM_ADAPTOR
  torch_txda::TXDAStreamGuard guard_;
#elif USE_ENFLAME_ADAPTOR
  torch_gcu::GCUStreamGuard guard_;
#endif
};

} // namespace c10d
