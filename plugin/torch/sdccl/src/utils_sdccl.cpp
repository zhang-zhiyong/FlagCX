#include "utils_sdccl.hpp"

namespace c10d {

std::string getSdcclVersion() {
  static c10::once_flag sdcclGetVersionFlag;
  static std::string versionString;

  c10::call_once(sdcclGetVersionFlag, []() {
    int version = 0;
    sdcclResult_t status = sdcclGetVersion(&version);
    if (status != sdcclSuccess) {
      versionString = "Unknown Sdccl version";
    } else {
      versionString = std::to_string(version);
    }
  });
  return versionString;
}

std::string sdcclGetErrorWithVersion(sdcclResult_t error) {
  return std::string(sdcclGetErrorString(error)) + ", Sdccl version " +
         getSdcclVersion();
}

// Provides additional detail into Sdccl error codes based on when these are
// thrown in the Sdccl codebase.
std::string getSdcclErrorDetailStr(
    sdcclResult_t error,
    std::optional<std::string> processGroupFailureReason /* = std::nullopt */) {
  // Prioritize failure reason provided by PG Sdccl first, as it can abort
  // communicators when it encounters collective timeouts, etc.
  if (processGroupFailureReason != std::nullopt) {
    return *processGroupFailureReason;
  }
  std::string interpret;
  std::string err;
  auto ret = sdcclGetLastError(nullptr);
  if (ret) {
    err = "\nLast error:\n" + std::string(ret);
  } else {
    err = "\nLast error: Unknown Sdccl Error\n";
  }

  switch (error) {
    case sdcclUnhandledDeviceError:
      interpret = "sdcclUnhandledDeviceError: Call to Device function failed.";
      break;
    case sdcclSystemError:
      interpret = "sdcclSystemError: System call (e.g. socket, malloc) or "
                  "external library call failed or device error. ";
      break;
    case sdcclRemoteError:
      interpret = "sdcclRemoteError: A call failed possibly due to a network "
                  "error or a remote process exiting prematurely.";
      break;
    case sdcclInternalError:
      interpret = "sdcclInternalError: Internal check failed.";
      break;
    case sdcclInvalidArgument:
      interpret = "sdcclInvalidArgument: Invalid value for an argument.";
      break;
    case sdcclInvalidUsage:
      interpret = "sdcclInvalidUsage: This usually reflects invalid usage of "
                  "Sdccl library.";
      break;
    default:
      interpret = "Unknown Sdccl error!";
  }
  return interpret + err;
}
} // namespace c10d