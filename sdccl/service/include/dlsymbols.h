/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef SDCCL_DLSYMBOLS_H_
#define SDCCL_DLSYMBOLS_H_

#include "sdccl.h"
#include <dlfcn.h>

extern void *sdcclOpenLib(const char *path, int flags,
                           void (*error_handler)(const char *, int,
                                                 const char *));

// Function pointer types for custom operations
template <typename T, typename... Args>
using sdcclCustomOpFunc_t = T (*)(Args...);
using sdcclLaunchFunc_t = sdcclCustomOpFunc_t<void, sdcclStream_t, void *>;

// Load a custom operation symbol from a shared library
template <typename T>
inline sdcclResult_t loadCustomOpSymbol(const char *path, const char *name,
                                         T *fn) {
  void *handle = sdcclOpenLib(
      path, RTLD_LAZY, [](const char *p, int err, const char *msg) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
      });
  if (!handle)
    return sdcclSystemError;

  void *sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    dlclose(handle);
    return sdcclSystemError;
  }

  *fn = (T)sym;
  return sdcclSuccess;
}

inline sdcclResult_t loadKernelSymbol(const char *path, const char *name,
                                       sdcclLaunchFunc_t *fn) {
  return loadCustomOpSymbol<sdcclLaunchFunc_t>(path, name, fn);
}

#endif // SDCCL_DLSYMBOLS_H_
