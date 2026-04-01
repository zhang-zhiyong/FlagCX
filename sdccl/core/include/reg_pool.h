#ifndef SDCCL_REGPOOL_H
#define SDCCL_REGPOOL_H

#include "check.h"
#include "device.h"
#include "sdccl.h"
#include "net.h"
#include "register.h"
#include <map>
#include <unistd.h>

class sdcclRegPool {
public:
  sdcclRegPool();
  ~sdcclRegPool();

  void getPagedAddr(void *data, size_t length, uintptr_t *beginAddr,
                    uintptr_t *endAddr);
  sdcclResult_t addNetHandle(void *comm, sdcclRegItem *reg, void *handle,
                              struct sdcclProxyConnector *proxyConn);
  sdcclResult_t removeRegItemNetHandles(void *comm, sdcclRegItem *reg);
  sdcclResult_t addP2pHandle(void *comm, sdcclRegItem *reg, void *handle,
                              struct sdcclProxyConnector *proxyConn);
  sdcclResult_t removeRegItemP2pHandles(void *comm, sdcclRegItem *reg);
  sdcclResult_t registerBuffer(void *comm, void *data, size_t length);
  sdcclResult_t deregisterBuffer(void *comm, void *handle);
  std::map<uintptr_t, std::map<uintptr_t, sdcclRegItem *>> &getGlobalMap();
  sdcclRegItem *getItem(const void *comm, void *data);
  void dump();

private:
  void mapRegItemPages(uintptr_t commKey, sdcclRegItem *reg);
  std::map<uintptr_t, std::map<uintptr_t, sdcclRegItem *>>
      regMap; // <commPtr, <pageBasePtr, regItemPtr>>
  std::map<uintptr_t, std::list<sdcclRegItem>>
      regPool; // <commPtr, regItemList>
  uintptr_t pageSize;
};

extern sdcclRegPool globalRegPool;

#endif // SDCCL_REGPOOL_H