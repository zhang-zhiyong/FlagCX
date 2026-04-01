/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "runner.h"

struct sdcclRunner *sdcclRunners[NRUNNERS] = {&homoRunner, &hostRunner,
                                                &hybridRunner, &uniRunner};