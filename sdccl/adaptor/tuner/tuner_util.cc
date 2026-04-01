#include "tuner_util.h"

// Safely copy std::string to char buffer, ensuring NUL termination and
// truncation
static void safeStrCopy(char *dst, size_t dstSize, const std::string &src) {
  if (dstSize == 0)
    return;
  size_t copyLen = std::min(dstSize - 1, src.size());
  if (copyLen > 0)
    memcpy(dst, src.data(), copyLen);
  dst[copyLen] = '\0';
}

// Generate all combinations and return a vector of sdcclEnvConfig
sdcclResult_t generateCandidate(std::vector<struct sdcclEnvConfig> &cfgList) {

  // Return empty if there are no environment variables
  if (vars.empty()) {
    INFO(SDCCL_INIT, "Invalid number of environment variables: 0");
    return sdcclInvalidArgument;
  }

  // If the number of variables exceeds the structure capacity, truncate
  if (vars.size() > (size_t)SDCCL_ENV_LIST_MAX_LENGTH) {
    INFO(SDCCL_INIT, "The number of environment variables exceeds the maximum "
                      "length defined by SDCCL_ENV_LIST_MAX_LENGTH");
    vars.resize(SDCCL_ENV_LIST_MAX_LENGTH); // Truncate the vars vector
    INFO(SDCCL_INIT,
         "The number of environment variables has been truncated to "
         "SDCCL_ENV_LIST_MAX_LENGTH (%d)",
         SDCCL_ENV_LIST_MAX_LENGTH);
    return sdcclSuccess;
  }

  // Prepare candidate value lists for each variable (at least one empty string
  // to ensure uniform combination logic)
  std::vector<std::vector<std::string>> lists;
  lists.reserve(vars.size());
  for (const auto &v : vars) {
    if (v.choices.empty()) {
      lists.emplace_back(std::vector<std::string>{""});
    } else {
      lists.emplace_back(v.choices);
    }
  }

  // Use an index vector to iterate through the Cartesian product
  // (multi-dimensional counter)
  size_t nvars = lists.size();
  std::vector<size_t> idx(nvars, 0);
  bool done = (nvars == 0);
  unsigned long numCandidate = 0;

  while (!done) {
    // Construct a sdcclEnvConfig and zero-initialize
    sdcclEnvConfig cfg;
    memset(&cfg, 0, sizeof(cfg)); // this zeroes commTag and all fields; adjust
                                  // if you want non-zero defaults

    std::string tagStr = "Config " + std::to_string(numCandidate);
    if (tagStr.size() < sizeof(cfg.commTag.tag)) {
      safeStrCopy(cfg.commTag.tag, sizeof(cfg.commTag.tag), tagStr);
    } else {
      INFO(SDCCL_INIT, "Tag string too long, potential buffer overflow");
      return sdcclInvalidArgument;
    }
    cfg.envCount = 0;

    // Fill envs
    for (size_t i = 0; i < nvars; ++i) {
      sdcclEnvEntity &ent = cfg.envs[i];
      // type
      ent.type = SDCCL_ENV_TYPE_CREATION;
      // name
      safeStrCopy(ent.name, sizeof(ent.name), vars[i].name);
      // value
      const std::string &val = lists[i][idx[i]];
      safeStrCopy(ent.value, sizeof(ent.value), val);
      // defaultValue
      safeStrCopy(ent.defaultValue, sizeof(ent.defaultValue),
                  vars[i].defaultValue);

      cfg.envCount++;
      // Stop if exceeding the maximum allowed envs (should not happen since we
      // truncated vars earlier)
      if (cfg.envCount >= SDCCL_ENV_LIST_MAX_LENGTH)
        break;
    }

    cfgList.push_back(cfg);

    // Increment counter (from least significant to most significant)
    for (int i = (int)nvars - 1; i >= 0; --i) {
      idx[i]++;
      if (idx[i] < lists[i].size())
        break;
      idx[i] = 0;
      if (i == 0)
        done = true;
    }
    numCandidate += 1;
  }

  return sdcclSuccess;
}
