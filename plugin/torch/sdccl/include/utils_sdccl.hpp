#pragma once

#include "sdccl.h"
#include <c10/util/CallOnce.h>
#include <optional>
#include <string>

// Macro to throw on a non-successful Sdccl return value.
#define C10D_SDCCL_CHECK(cmd, failureReason)                                  \
  do {                                                                         \
    sdcclResult_t result = cmd;                                               \
    if (result != sdcclSuccess) {                                             \
      std::string err = "SDCCL error in: " + std::string(__FILE__) + ":" +    \
                        std::to_string(__LINE__) +                             \
                        ", " /*+ sdcclGetErrorWithVersion(result)*/ + "\n" +  \
                        getSdcclErrorDetailStr(result, failureReason);        \
      TORCH_CHECK_WITH(DistBackendError, false, err);                          \
    }                                                                          \
  } while (0)

namespace c10d {

// RAII helper class to manage Sdccl group API.
// The destructor is allowed to throw since this helper class only
// manages group lifetimes.
struct sdcclGroupGuard final {
  sdcclGroupGuard(sdcclComm_t comm) {
    comm_ = comm;
    sdcclGroupStart(comm_);
  }
  ~sdcclGroupGuard() noexcept(false) { sdcclGroupEnd(comm_); }
  sdcclComm_t comm_ = nullptr;
};

std::string getSdcclVersion();

std::string sdcclGetErrorWithVersion(sdcclResult_t error);

// Provides additional detail into Sdccl error codes based on when these are
// thrown in the Sdccl codebase.
std::string getSdcclErrorDetailStr(
    sdcclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

} // namespace c10d