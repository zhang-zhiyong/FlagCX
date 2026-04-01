#ifndef SDCCL_UNIRUNNER_IMPL_H_
#define SDCCL_UNIRUNNER_IMPL_H_

#include "device.h"
#include "sdccl.h"
#include "sdccl_kernel.h"
#include "sdccl_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include "utils.h"
#include <memory>
#include <pthread.h>

// DAG node types
typedef enum {
  uniRunnerDagNodeTypeP2p = 0,
  uniRunnerDagNodeTypeRed = 1,
  uniRunnerDagNodeTypeCpy = 2
} uniRunnerDagNodeType;

// Single P2P operation data
struct uniRunnerP2pOpData {
  void *addr;                // Buffer address
  size_t count;              // Element count
  int peerRank;              // Peer rank
  sdcclDataType_t datatype; // Data type
  sdcclDevicePrim type;     // Primitive type (send/recv/term/wait)
};

// P2P node data (supports multiple operations in a group)
struct uniRunnerP2pNodeData {
  // Operation information for P2P trigger
  struct uniRunnerP2pOpData *ops; // Array of P2P operations
  int numOps;                     // Number of operations

  // Event for completion tracking
  int eventIdx; // Index of the event in the pool
};

// Reduce node data (operation-specific fields only)
struct uniRunnerRedNodeData {
  // Operation information for reduce trigger
  void *input1;
  void *input2;
  void *output;
  size_t count;
  size_t nthreads;
  sdcclDataType_t datatype;
  sdcclRedOp_t redOp;

  // Trigger and state tracking
  int triggerIdx; // Trigger index in FIFO
};

// Copy node data (operation-specific fields only)
struct uniRunnerCpyNodeData {
  // Operation information for local memcpy
  void *src;
  void *dst;
  size_t count;
  sdcclDataType_t datatype;

  int eventIdx;
};

// Unified DAG node with common DAG structure fields
struct uniRunnerDagNode {
  uniRunnerDagNodeType nodeType; // Discriminator for union

  // Common DAG structure fields (shared by all node types)
  int numParents;                // Number of parent dependencies
  int numChildren;               // Number of children
  int *children;                 // Array of child node indices
  struct uniRunnerDagNode *next; // Queue linkage

  // Union for type-specific operation data
  union {
    struct uniRunnerP2pNodeData p2p;
    struct uniRunnerRedNodeData red;
    struct uniRunnerCpyNodeData cpy;
  } nodeData;
};

// Bitmap for p2pEvent availability
// 1 means in use, 0 means available
typedef struct {
  uint64_t *bits;
  size_t nextIdx;
  size_t size; // total number of events

  // Check if event at index is available
  bool isAvailable(int index);
  // Get first available event index, or -1 if none
  int getAvailable();
  // Mark event at index as in use
  void markInUse(int index);
  // Mark event at index as available
  void markAvailable(int index);
} uniRunnerP2pEventBitmap;

typedef struct {
  pthread_t thread;
  sdcclFifo_t fifo;
  sdcclStream_t commStream;
  sdcclStream_t redStream;
  sdcclStream_t cpyStream;

  // new: DAG and scheduling queues
  struct uniRunnerDagNode *dagNodes; // Array of all DAG nodes
  int numDagNodes;
  int numPendingNodes;
  sdcclIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pReadyQueue;
  sdcclIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redReadyQueue;
  sdcclIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pInflightQueue;
  sdcclIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redInflightQueue;

  uint64_t p2pEventPoolSize;
  uint64_t uniRunnerNSlices;
  uint64_t uniRunnerNThreads;
  uint64_t uniRunnerNBlocks;
  uint64_t uniRunnerNRedSlices;
  uint64_t uniRunnerRedSliceSize;

  // P2P event pool
  sdcclEvent_t *p2pEvents;
  uniRunnerP2pEventBitmap p2pEventMap;

  // get an available event
  int getEvent();
  void resetEvent(int idx);
} sdcclUniRunnerState;

sdcclResult_t initUniRunnerStateDummy(sdcclUniRunnerState *runnerState);
sdcclResult_t initUniRunnerStateLocRed(sdcclUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, sdcclDataType_t datatype,
                                        sdcclRedOp_t op, sdcclComm_t comm);
sdcclResult_t initUniRunnerStateRingAG(sdcclUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, sdcclDataType_t datatype,
                                        sdcclRedOp_t op, sdcclComm_t comm);
sdcclResult_t initUniRunnerStateRingAR(sdcclUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, sdcclDataType_t datatype,
                                        sdcclRedOp_t op, sdcclComm_t comm);
sdcclResult_t initUniRunnerStateSlicedAR(sdcclUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          sdcclDataType_t datatype,
                                          sdcclRedOp_t op, sdcclComm_t comm);
sdcclResult_t initUniRunnerStateRingRS(sdcclUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        sdcclDataType_t datatype,
                                        sdcclRedOp_t op, sdcclComm_t comm);
sdcclResult_t initUniRunnerStateTreeRed(sdcclUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         void *scratchbuff, size_t count,
                                         sdcclDataType_t datatype,
                                         sdcclRedOp_t op, int root,
                                         sdcclComm_t comm);
sdcclResult_t initUniRunner(sdcclComm_t comm, sdcclStream_t stream);
sdcclResult_t cleanupUniRunner(sdcclComm_t comm);
sdcclResult_t runUniRunner(sdcclComm_t comm);
#endif // SDCCL_UNIRUNNER_IMPL_H_
