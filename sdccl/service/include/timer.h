/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SDCCL_TIMER_H_
#define SDCCL_TIMER_H_
#if ENABLE_TIMER
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>
static double freq = -1;
static void calibrate() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = -tv.tv_sec * 1E6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i = 0; i < 10000; i++)
    total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec * 1E6 + tv.tv_usec;
  freq = timeCycles / time;
}
static inline double gettime() {
  if (freq == -1)
    calibrate();
  return __rdtsc() / freq;
}
static uint64_t counts[8];
static double times[8];
static double startTimes[8];
#define TIME_START(index)                                                      \
  do {                                                                         \
    counts[index]++;                                                           \
    startTimes[index] = gettime();                                             \
  } while (0);

#define TIME_STOP(index)                                                       \
  do {                                                                         \
    times[index] += gettime() - startTimes[index];                             \
  } while (0);

#define TIME_CANCEL(index)                                                     \
  do {                                                                         \
    counts[index]--;                                                           \
  } while (0);

#define TIME_PRINT(name)                                                       \
  do {                                                                         \
    printf("%s stats", name);                                                  \
    for (int i = 0; i < 8; i++) {                                              \
      if (counts[i])                                                           \
        printf(" [%d] %g/%ld = %g", i, times[i], counts[i],                    \
               times[i] / counts[i]);                                          \
      counts[i] = 0;                                                           \
    }                                                                          \
    printf("\n");                                                              \
  } while (0);
#else
#define TIME_START(index)                                                      \
  while (0)                                                                    \
    ;
#define TIME_STOP(index)                                                       \
  while (0)                                                                    \
    ;
#define TIME_CANCEL(index)                                                     \
  while (0)                                                                    \
    ;
#define TIME_PRINT(name)
#endif

#include <cassert>
#include <pthread.h>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "adaptor.h"
#include "debug.h"
#include "sdccl.h"

constexpr int RECORD_NUM = 2048;

template <typename T>
struct sdcclRecordKey {
  T value;

  sdcclRecordKey() = default;
  sdcclRecordKey(const T &v) : value(v) {}

  bool operator<(const sdcclRecordKey<T> &other) const {
    return value < other.value;
  }
  bool operator==(const sdcclRecordKey<T> &other) const {
    return value == other.value;
  }
};

template <typename T>
struct sdcclRecord {
  sdcclEvent_t beginEvent;
  sdcclEvent_t endEvent;
  sdcclRecordKey<T> recordKey;
  float duration; // ms
  sdcclStream_t stream;

  sdcclRecord();
  sdcclRecord(const sdcclRecord &) = delete;
  sdcclRecord &operator=(const sdcclRecord &) = delete;
  sdcclRecord(sdcclRecord &&) = delete;
  sdcclRecord &operator=(sdcclRecord &&) = delete;

  ~sdcclRecord();
};

template <typename T>
sdcclRecord<T>::sdcclRecord() : duration(0.0f) {
  deviceAdaptor->eventCreate(&beginEvent, sdcclEventDefault);
  deviceAdaptor->eventCreate(&endEvent, sdcclEventDefault);
}

template <typename T>
sdcclRecord<T>::~sdcclRecord<T>() {
  deviceAdaptor->eventDestroy(beginEvent);
  deviceAdaptor->eventDestroy(endEvent);
}

template <typename T>
class sdcclTimer {

public:
  sdcclRecord<T> sdcclRecords[RECORD_NUM];
  pthread_t queryThread;
  bool stopQuery = false;
  std::queue<sdcclRecord<T> *> availableRecords; // NOLINT
  std::queue<sdcclRecord<T> *> usingRecords;     // NOLINT
  std::queue<sdcclRecord<T> *> profilingRecords; // NOLINT
  std::queue<sdcclRecord<T> *> profiledRecords;  // NOLINT
  pthread_mutex_t mutexAvailable{};
  pthread_cond_t condAvailable{};
  pthread_mutex_t mutexProfiling{};
  pthread_cond_t condProfiling{};
  pthread_mutex_t mutexProfiled{};

  void initSyncPrimitives();
  void destroySyncPrimitives();

public:
  sdcclTimer();
  ~sdcclTimer();

  void start();
  void stop();

  sdcclResult_t begin(const sdcclRecordKey<T> &recordKey,
                       sdcclStream_t stream, bool blocking = false);
  sdcclResult_t end(const sdcclRecordKey<T> &recordKey,
                     bool blocking = false);

  float getRecord(const sdcclRecordKey<T> &recordKey, bool blocking = false);
};

template <typename T>
void sdcclTimer<T>::initSyncPrimitives() {
  pthread_mutex_init(&mutexAvailable, nullptr);
  pthread_mutex_init(&mutexProfiling, nullptr);
  pthread_mutex_init(&mutexProfiled, nullptr);

  pthread_cond_init(&condAvailable, nullptr);
  pthread_cond_init(&condProfiling, nullptr);
}

template <typename T>
void sdcclTimer<T>::destroySyncPrimitives() {
  pthread_cond_destroy(&condAvailable);
  pthread_cond_destroy(&condProfiling);

  pthread_mutex_destroy(&mutexAvailable);
  pthread_mutex_destroy(&mutexProfiling);
  pthread_mutex_destroy(&mutexProfiled);
}

template <typename T>
void *sdcclQuery(void *sdcclTimer_) {
  auto *timer = static_cast<sdcclTimer<T> *>(sdcclTimer_);
  sdcclRecord<T> *currRecord = nullptr;

  while (true) {
    currRecord = nullptr;
    pthread_mutex_lock(&timer->mutexProfiling);
    // wait for profilingRecords not empty or stop signal
    while (!timer->stopQuery && timer->profilingRecords.empty()) {
      pthread_cond_wait(&timer->condProfiling, &timer->mutexProfiling);
    }
    // elegant exit
    if (timer->stopQuery && timer->profilingRecords.empty() && !currRecord) {
      pthread_mutex_unlock(&timer->mutexProfiling);
      break;
    }

    if (timer->profilingRecords.empty()) {
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }

    // limitation: process record one by one linearly.
    currRecord = timer->profilingRecords.front();
    if (!currRecord) {
      WARN("profilingRecords front is null, drop this record");
      timer->profilingRecords.pop();
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }
    // INFO(SDCCL_TUNING, "Start to process record %s",
    // currRecord->recordKey.value.toString().c_str());
    sdcclResult_t res = sdcclSuccess;
    res = deviceAdaptor->eventQuery(currRecord->endEvent);
    if (res != sdcclSuccess) {
      if (res != sdcclInProgress) {
        WARN("Cannot query event, drop this record %s",
             currRecord->recordKey.value.toString().c_str());
        timer->profilingRecords.pop();
        pthread_mutex_unlock(&timer->mutexProfiling);
        continue;
      }
      // INFO(SDCCL_TUNING, "Record %s endEvent not ready.",
      // currRecord->recordKey.value.toString().c_str());
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue; // still in progress, try again later
    }
    // when here, both beginEvent and endEvent are recorded
    res = deviceAdaptor->eventElapsedTime(&currRecord->duration,
                                          currRecord->beginEvent,
                                          currRecord->endEvent); // ms
    if (res != sdcclSuccess) {
      WARN("Cannot get elapsed time, drop this record %s",
           currRecord->recordKey.value.toString().c_str());
      timer->profilingRecords.pop();
      pthread_mutex_unlock(&timer->mutexProfiling);
      continue;
    }

    // move currRecord from profilingRecords to profiledRecords
    timer->profilingRecords.pop();
    pthread_mutex_unlock(&timer->mutexProfiling);
    pthread_mutex_lock(&timer->mutexProfiled);
    timer->profiledRecords.push(currRecord);
    pthread_mutex_unlock(&timer->mutexProfiled);
    // INFO(SDCCL_TUNING, "Moving record %s to profiled queue.",
    // currRecord->recordKey.value.toString().c_str());
  }
  return nullptr;
}

template <typename T>
sdcclTimer<T>::sdcclTimer() {
  initSyncPrimitives();
  for (auto &rec : sdcclRecords) {
    this->availableRecords.push(&rec);
  }
}
template <typename T>
sdcclTimer<T>::~sdcclTimer() {
  if (!stopQuery) {
    stop();
  }
  destroySyncPrimitives();
}

template <typename T>
void sdcclTimer<T>::start() {
  pthread_create(&queryThread, NULL, &sdcclQuery<T>, this);
  INFO(SDCCL_TUNING, "sdccl timer start profiling thread");
}

template <typename T>
void sdcclTimer<T>::stop() {
  INFO(SDCCL_TUNING, "stopping timer");
  pthread_mutex_lock(&this->mutexProfiling);
  stopQuery = true;
  pthread_cond_signal(&this->condProfiling);
  pthread_mutex_unlock(&this->mutexProfiling);
  pthread_join(queryThread, NULL);
}

template <typename T>
float sdcclTimer<T>::getRecord(const sdcclRecordKey<T> &recordKey,
                                bool blocking) {
  sdcclRecord<T> *found_record = nullptr;
  int iter = 0;
  do {
    std::queue<sdcclRecord<T> *> remaining_records;
    pthread_mutex_lock(&this->mutexProfiled);
    while (!this->profiledRecords.empty()) {
      sdcclRecord<T> *record = this->profiledRecords.front();
      this->profiledRecords.pop();
      if (found_record == nullptr && record->recordKey == recordKey) {
        found_record = record;
      } else {
        remaining_records.push(record);
      }
    }
    this->profiledRecords.swap(remaining_records);
    pthread_mutex_unlock(&this->mutexProfiled);
    // TODO: add a timeout to avoid infinite loop
    // INFO(SDCCL_TUNING, "Searched %d times for getRecord %s.", iter,
    // recordKey.value.toString().c_str());
    iter++;
  } while (blocking && !found_record);

  if (found_record) {
    float duration = found_record->duration;
    pthread_mutex_lock(&this->mutexAvailable);
    this->availableRecords.push(found_record);
    pthread_cond_signal(&this->condAvailable);
    pthread_mutex_unlock(&this->mutexAvailable);
    return duration;
  }

  return -1.0f; // Indicate that no matching record was found
}

template <typename T>
sdcclResult_t sdcclTimer<T>::begin(const sdcclRecordKey<T> &recordKey,
                                     sdcclStream_t stream_, bool blocking) {
  sdcclRecord<T> *record = nullptr;

  pthread_mutex_lock(&this->mutexAvailable);
  while (availableRecords.empty() && blocking) {
    WARN("sdccl event is empty!");
    pthread_cond_wait(&this->condAvailable, &this->mutexAvailable);
  }
  if (!availableRecords.empty()) {
    record = availableRecords.front();
    availableRecords.pop();
  }
  pthread_mutex_unlock(&this->mutexAvailable);

  if (record) {
    record->recordKey = recordKey;
    record->stream = stream_;
    SDCCLCHECK(deviceAdaptor->eventRecord(record->beginEvent, record->stream));
    usingRecords.push(record);
  } else {
    WARN("no available records");
    return sdcclInternalError;
  }

  return sdcclSuccess;
}

template <typename T>
sdcclResult_t sdcclTimer<T>::end(const sdcclRecordKey<T> &recordKey,
                                   bool blocking) {
  if (usingRecords.empty()) {
    return sdcclInvalidUsage;
  }

  // Find the record with recordKey
  sdcclRecord<T> *record = nullptr;
  std::queue<sdcclRecord<T> *> usingRecordsCopy;

  while (!usingRecords.empty()) {
    record = usingRecords.front();
    usingRecords.pop();
    if (record->recordKey == recordKey) {
      // Record found, update the endEvent and add it back to usingRecords
      SDCCLCHECK(deviceAdaptor->eventRecord(record->endEvent, record->stream));
      break;
    } else {
      // Record not found, keep it in usingRecords
      usingRecordsCopy.push(record);
      record = nullptr;
    }
  }

  // Add the records from usingRecordsCopy to usingRecords
  while (!usingRecords.empty()) {
    usingRecordsCopy.push(usingRecords.front());
    usingRecords.pop();
  }
  usingRecords = usingRecordsCopy;

  if (record == nullptr) {
    WARN("no matching begin for end call");
    return sdcclInvalidUsage;
  }

  if (blocking) {
    SDCCLCHECK(deviceAdaptor->streamSynchronize(record->stream));
  }

  pthread_mutex_lock(&this->mutexProfiling);
  this->profilingRecords.push(record);
  pthread_cond_signal(&this->condProfiling);
  pthread_mutex_unlock(&this->mutexProfiling);

  return sdcclSuccess;
}

#endif
