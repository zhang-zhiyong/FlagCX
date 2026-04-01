/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/

#include "adaptor.h"
#include "core.h"
#include "net.h"
#include <string.h>

#ifdef USE_NVIDIA_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &ncclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &ncclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &ncclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &cudaAdaptor;
#elif USE_ASCEND_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &hcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &hcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &hcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &cannAdaptor;
#elif USE_ILUVATAR_COREX_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &ixncclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &ixncclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &ixncclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &ixcudaAdaptor;
#elif USE_CAMBRICON_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &cnclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &cnclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &cnclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &mluAdaptor;
#elif USE_METAX_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &mcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &mcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &mcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &macaAdaptor;

#elif USE_MUSA_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &musa_mcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &musa_mcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &musa_mcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &musaAdaptor;

#elif USE_KUNLUNXIN_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &xcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &xcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &xcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &kunlunAdaptor;
#elif USE_DU_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &duncclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &duncclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &duncclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &ducudaAdaptor;

#elif USE_AMD_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &rcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &rcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &rcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &hipAdaptor;

#elif USE_TSM_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &tcclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &tcclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &tcclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &tsmicroAdaptor;

#elif USE_ENFLAME_ADAPTOR
#ifdef USE_BOOTSTRAP_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&bootstrapAdaptor,
                                                      &ecclAdaptor};
#elif USE_GLOO_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&glooAdaptor,
                                                      &ecclAdaptor};
#elif USE_MPI_ADAPTOR
struct sdcclCCLAdaptor *cclAdaptors[NCCLADAPTORS] = {&mpiAdaptor,
                                                      &ecclAdaptor};
#endif
struct sdcclDeviceAdaptor *deviceAdaptor = &topsAdaptor;

#endif

// External adaptor declarations
extern struct sdcclNetAdaptor sdcclNetSocket;
extern struct sdcclNetAdaptor sdcclNetIb;

#ifdef USE_IBUC
extern struct sdcclNetAdaptor sdcclNetIbuc;
#endif

#ifdef USE_UCX
extern struct sdcclNetAdaptor sdcclNetUcx;
#endif

// Unified network adaptor entry point
struct sdcclNetAdaptor *getUnifiedNetAdaptor(int netType) {
  switch (netType) {
    case IBRC:
#ifdef USE_UCX
      // When UCX is enabled, use UCX instead of IBRC
      return &sdcclNetUcx;
#elif USE_IBUC
      // When IBUC is enabled, use IBUC instead of IBRC
      return &sdcclNetIbuc;
#else
      return &sdcclNetIb;
#endif
    case SOCKET:
      return &sdcclNetSocket;
    default:
      return NULL;
  }
}
