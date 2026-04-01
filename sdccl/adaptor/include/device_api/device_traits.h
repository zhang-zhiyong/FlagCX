/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device Traits — Unified compile-time dispatch for device APIs.
 *
 * Architecture:
 *   PlatformTraits<P>         — platform-level: Intrin, Atomic
 *   DeviceTraits<D>           — backend-level:  Window, DevComm, Team, ...
 *   Fallback<PlatformTag>     — common IPC fallback (partial specialization)
 *
 * DeviceTraits pulls in platform capabilities via using-aliases (not
 * inheritance). Vendor specializations wrap vendor types with member
 * functions. The Fallback partial specialization provides IPC-based
 * types that work with any platform.
 *
 * Selection:
 *   NVIDIA + NCCL > 2.28:   DeviceAPI = DeviceTraits<NvidiaVendor>
 *   NVIDIA + fallback:       DeviceAPI = DeviceTraits<Fallback<NvidiaPlatform>>
 *
 * Kernel code uses DeviceAPI::* exclusively, no #ifdef branches.
 ************************************************************************/

#ifndef SDCCL_DEVICE_TRAITS_H_
#define SDCCL_DEVICE_TRAITS_H_

#include "platform_traits.h"
#include <cstddef>
#include <cstdint>

// Primary template — each backend provides a specialization
template <typename Impl>
struct DeviceTraits;

// Fallback tag — parameterized by platform for the partial specialization
template <typename PlatformTag>
struct Fallback {};

// ============================================================
// Action types for one-sided operations (needed by traits Net types).
// Pure POD structs with no device builtins.
// ============================================================
typedef uint32_t sdcclDevNetSignal_t;
typedef uint32_t sdcclDevNetCounter_t;

struct sdcclDevNet_None {};
struct sdcclDevNet_SignalInc {
  sdcclDevNetSignal_t signal;
};
struct sdcclDevNet_SignalAdd {
  sdcclDevNetSignal_t signal;
  uint64_t value;
};
struct sdcclDevNet_CounterInc {
  sdcclDevNetCounter_t counter;
};

// Shared memory descriptor for NIC descriptor optimization.
// Uses void* on all paths; vendor Net casts to native type in toNccl().
struct sdcclDescriptorSmem {
  void *_impl = nullptr;
};

struct sdcclDevNet_DescriptorSmem {
  sdcclDescriptorSmem smem;
};

// Fence level enum — available on all tiers for unified barrier API
enum class sdcclGinFenceLevel { Relaxed };

// ============================================================
// Barrier tag types for DevBarrier<Backend, Tag> dispatch.
// ============================================================
struct sdcclBarrierIntra {};
struct sdcclBarrierInter {};
struct sdcclBarrierWorld {
  struct World {}; // tag for world-barrier ctor
  struct Intra {}; // tag for intra-only ctor
  struct Inter {}; // tag for inter-only ctor
};

// Primary template — each backend provides specializations
template <typename Backend, typename BarrierTag>
struct DevBarrier;

// Common fallback partial specialization (IPC-based, works for any platform)
#include "fallback_device_traits.h"

// Vendor specializations + DeviceAPI selection
#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_device_traits.h"
#endif

// Future:
// #ifdef USE_DU_ADAPTOR
// #include "du_device_traits.h"
// #endif

#endif // SDCCL_DEVICE_TRAITS_H_
