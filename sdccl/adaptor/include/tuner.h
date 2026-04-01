#ifndef SDCCL_ADAPTOR_TUNER_H_
#define SDCCL_ADAPTOR_TUNER_H_

#include "debug.h"
#include "sdccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sdccl environment variable types
enum sdcclEnvType {
  SDCCL_ENV_TYPE_CREATION = 0x01, // envs used when creating the communicator
  SDCCL_ENV_TYPE_COLL =
      0x02, // envs should be set for every specific collective at runtime
  SDCCL_ENV_TYPE_ONETIME =
      0x04 // envs take effect only the first time they are used at runtime
};

#define SDCCL_ENV_ENTITY_MAX_LENGTH                                           \
  128 // max length of a single env name or value
struct sdcclEnvEntity {
  uint32_t type; // env type bitmap, OR of SDCCL_ENV_TYPE_*
  char name[SDCCL_ENV_ENTITY_MAX_LENGTH];
  char value[SDCCL_ENV_ENTITY_MAX_LENGTH];
  char defaultValue[SDCCL_ENV_ENTITY_MAX_LENGTH]; // default value used to
                                                   // unset this env
};

#define SDCCL_COMM_TAG_MAX_LENGTH 64 // max length of communicator tag
// A tag to identify a specific communicator configuration
struct sdcclCommTag {
  char tag[SDCCL_COMM_TAG_MAX_LENGTH]; // tag string
};

// Structure of environment list for a specific communicator candidate
// configuration
#define SDCCL_ENV_LIST_MAX_LENGTH 32 // max length of env list per communicator
struct sdcclEnvConfig {
  sdcclCommTag commTag; // communicator tag
  int envCount;          // number of env vars
  struct sdcclEnvEntity envs[SDCCL_ENV_LIST_MAX_LENGTH];
};

#define SDCCL_ENV_CONFIG_MAX_COUNT 4 // max number of communicator configs
// A list of sdcclEnvConfig
struct sdcclEnvConfigList {
  int nConfigs; // number of communicator configs
  struct sdcclEnvConfig configList[SDCCL_ENV_CONFIG_MAX_COUNT];
};

// Used to pair ProfilingStart()/ProfilingStop() calls
#define SDCCL_PROFILE_KEY_MAX_LENGTH 64 // max length of profiling key string
struct sdcclProfileKey {
  char key[SDCCL_PROFILE_KEY_MAX_LENGTH]; // profiling key string
};

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // SDCCL_ADAPTOR_TUNER_H_
