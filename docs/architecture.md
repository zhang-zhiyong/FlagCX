# FlagCX 架构与流程图 / Architecture & Flow Diagrams

本文用 [Mermaid](https://mermaid.js.org/) 图把 FlagCX 的整体架构与典型执行流程落地到仓库,
方便新贡献者快速建立"代码地图"。所有图都基于仓库现状(`flagcx/core`、`flagcx/adaptor`、
`flagcx/service`、`plugin/`)绘制。

> GitHub 已原生支持在 Markdown 中渲染 Mermaid 代码块,无需额外工具即可查看。

## 目录

- [1. 总体分层架构](#1-总体分层架构)
- [2. 模块/目录映射](#2-模块目录映射)
- [3. Communicator 初始化流程](#3-communicator-初始化流程)
- [4. 同构 AllReduce 调用路径](#4-同构-allreduce-调用路径)
- [5. 异构集合通信(Hetero Proxy)时序](#5-异构集合通信hetero-proxy时序)
- [6. Device-Buffer IPC / RDMA 数据通路](#6-device-buffer-ipc--rdma-数据通路)
- [7. 框架接入(PyTorch / Paddle Plugin)](#7-框架接入pytorch--paddle-plugin)

---

## 1. 总体分层架构

```mermaid
flowchart TB
    subgraph App["上层应用 / Frameworks"]
        PT["PyTorch DDP\n(plugin/torch)"]
        PD["PaddlePaddle\n(adaptor_plugin)"]
        APP["Native C/C++ App\n(test/, runner/)"]
    end

    subgraph API["FlagCX Public API\n(flagcx/include, flagcx.cc)"]
        COLL["Collectives\nallreduce / allgather / alltoall(v) / ..."]
        P2P["P2P\nsend / recv / batch_isend_irecv"]
        MGMT["Group / Comm Mgmt\ngroup.cc, init.cc"]
    end

    subgraph Core["Core Engine (flagcx/core)"]
        INIT["init.cc\nbootstrap + topo + transport"]
        HET["flagcx_hetero.cc\n异构调度 / chunking"]
        PROXY["proxy.cc\n后台代理线程"]
        TOPO["topo.cc / paths.cc / xml.cc\n拓扑发现"]
        NET["net.cc / p2p.cc\n网络与 P2P 抽象"]
        TUNER["flagcx_tuner.cc\ncost_model.cc"]
        KERN["kernels/\nlaunch_kernel.cc"]
    end

    subgraph Service["Service (flagcx/service)"]
        BOOT["bootstrap.cc\nTCP 建链 / OOB"]
        IBV["ibvwrap.cc / ibvsymbols.cc"]
        SHM["shmutils.cc / ipcsocket.cc"]
        DBG["debug.cc / param.cc / utils.cc"]
    end

    subgraph Adaptors["Adaptors (flagcx/adaptor)"]
        CCL["CCL Adaptors\nnccl / cncl / hccl / mccl /\nixnccl / rccl / xccl / duccl /\nmusa / tccl / eccl /\nbootstrap / gloo / mpi"]
        DEV["Device Adaptors\ncuda / hip / cann / mlu /\nmusa / maca / tops /\nixcuda / ducuda / kunlunxin / tsmicro"]
        NETP["Net Adaptors\nibrc / ibrc_p2p / ibuc /\nucx / socket"]
        TUN["Tuner Adaptors"]
    end

    subgraph HW["硬件 / 厂商 SDK"]
        GPU["NV / AMD / Iluvatar / 海光 ..."]
        NPU["昇腾 / 寒武纪 / 沐曦 / 摩尔 / 燧原 / 清微 / 昆仑芯 ..."]
        FAB["RDMA (IB/RoCE) / Ethernet / NVLink / 厂商高速互连"]
    end

    App --> API
    API --> Core
    Core --> Service
    Core --> Adaptors
    Adaptors --> HW
    Service --> HW
```

要点:

- **API 层**(`flagcx/flagcx.cc` + `flagcx/include`)对外暴露统一的 C 接口,语义对齐 NCCL。
- **Core 层**负责 communicator 生命周期、拓扑发现、异构调度、Proxy 后台线程、kernel 启动。
- **Service 层**提供 bootstrap、IB verbs 加载、共享内存/IPC socket、日志参数等基础设施。
- **Adaptor 层**通过 plugin 加载机制(`*_plugin_load.cc`)在运行时绑定到具体的 CCL / Device / Net 后端,实现"一次开发,跨芯运行"。

---

## 2. 模块/目录映射

```mermaid
flowchart LR
    ROOT["FlagCX/"] --> FCX["flagcx/"]
    ROOT --> PLG["plugin/"]
    ROOT --> APLG["adaptor_plugin/"]
    ROOT --> TEST["test/"]
    ROOT --> DOCS["docs/"]
    ROOT --> MK["makefiles/, Makefile"]

    FCX --> CORE["core/\n初始化 / 拓扑 / 异构 / Proxy"]
    FCX --> ADP["adaptor/\nccl, device, net, tuner"]
    FCX --> SVC["service/\nbootstrap, ibv, shm, debug"]
    FCX --> KER["kernels/\n设备侧 reduce/copy kernel"]
    FCX --> INC["include/\n公共头文件"]
    FCX --> RUN["runner/\n内置 runner"]
    FCX --> TOOLS["tools/"]

    PLG --> PT["torch/\nPyTorch backend plugin"]
    PLG --> NCP["nccl/\nNCCL 兼容层"]
    PLG --> ISVC["interservice/"]

    APLG --> PADDLE["Paddle 适配"]
```

---

## 3. Communicator 初始化流程

下图给出 `flagcxCommInitRank` 类调用从应用进入 Core 的关键步骤
(对应 `flagcx/core/init.cc`、`bootstrap.cc`、`topo.cc`、`transport.cc`、`proxy.cc`)。

```mermaid
flowchart TD
    A["App: flagcxCommInitRank(comm, nranks, uniqueId, rank)"] --> B["init.cc\n参数校验 + 选择后端"]
    B --> C{"后端类型"}
    C -- "同构 (单一 xCCL)" --> C1["CCL Adaptor\n例: ncclCommInitRank"]
    C -- "异构 / 多后端" --> D["service/bootstrap.cc\nTCP OOB 建链\n(uniqueId 广播 / AllGather)"]
    D --> E["core/topo.cc + paths.cc + xml.cc\n本地拓扑发现 (PCIe/NVLink/IB)"]
    E --> F["core/p2p_topo.cc\n跨 rank 拓扑交换 + 路径选择"]
    F --> G["core/transport.cc + net.cc\n建立 P2P / Net 通道\n(IB verbs / UCX / Socket)"]
    G --> H["core/proxy.cc\n启动 Proxy 线程\n(管理异步发送/接收)"]
    H --> I["core/flagcx_hetero.cc\n构造异构通信计划\n(chunk / pipeline)"]
    I --> J["返回 flagcxComm_t 句柄"]
    C1 --> J
```

---

## 4. 同构 AllReduce 调用路径

```mermaid
sequenceDiagram
    autonumber
    participant App as App / Framework
    participant API as flagcx.cc (API)
    participant Grp as core/group.cc
    participant Adp as adaptor/ccl/<vendor>_adaptor.cc
    participant CCL as Vendor xCCL (NCCL/CNCL/HCCL/...)
    participant HW as Device + Fabric

    App->>API: flagcxAllReduce(sendbuf, recvbuf, count, dtype, op, comm, stream)
    API->>Grp: groupStart()/groupEnd() 包裹(若在 group 内)
    API->>Adp: adaptor->allReduce(...)
    Adp->>CCL: ncclAllReduce / cnclAllReduce / hcclAllReduce ...
    CCL->>HW: 入队设备侧 kernel + 触发互连
    HW-->>CCL: 完成事件
    CCL-->>Adp: 返回状态
    Adp-->>API: flagcxResult_t
    API-->>App: 返回 (异步,基于 stream)
```

> 同构路径下,FlagCX 主要承担"统一 API + 适配分发"的角色,实际数据搬运由原生 xCCL 完成。

---

## 5. 异构集合通信(Hetero Proxy)时序

异构场景下,跨厂商芯片之间没有统一的 xCCL,FlagCX 用 **Proxy 线程 + Net Adaptor** 在
host 侧编排数据搬运,并把"同侧"部分下沉给本地 xCCL。

```mermaid
sequenceDiagram
    autonumber
    participant App as App
    participant API as flagcx.cc
    participant Het as core/flagcx_hetero.cc
    participant LocalCCL as Local xCCL Adaptor
    participant Proxy as core/proxy.cc (后台线程)
    participant Net as adaptor/net (ibrc / ucx / socket)
    participant Peer as Remote Rank (异构芯片)

    App->>API: flagcxAllReduce(...)
    API->>Het: 构造异构计划(intra-node reduce + inter-node 交换 + broadcast)
    Het->>LocalCCL: 节点内 ReduceScatter / Reduce
    Het->>Proxy: 提交 send/recv 任务 (chunked)
    loop 每个 chunk
        Proxy->>Net: post send (device-buffer RDMA / IPC)
        Net->>Peer: 数据传输
        Peer-->>Net: 数据传输
        Net-->>Proxy: post recv 完成
    end
    Proxy-->>Het: 所有 chunk 完成
    Het->>LocalCCL: 节点内 AllGather / Broadcast
    LocalCCL-->>API: 完成
    API-->>App: 返回
```

---

## 6. Device-Buffer IPC / RDMA 数据通路

FlagCX 的两项原创能力(见 `README.md` About 一节):**device-buffer IPC** 与
**device-buffer RDMA**。它们在异构 P2P 场景下绕过多余的 H2D/D2H 拷贝。

```mermaid
flowchart LR
    subgraph NodeA["Node A (Vendor X)"]
        AppA["Sender App"]
        DevA["Device Buffer A"]
        IPCA["service/ipcsocket.cc\nshmutils.cc"]
        NetA["adaptor/net\nibrc / ibrc_p2p / ucx"]
    end

    subgraph NodeB["Node B (Vendor Y)"]
        AppB["Receiver App"]
        DevB["Device Buffer B"]
        IPCB["service/ipcsocket.cc\nshmutils.cc"]
        NetB["adaptor/net\nibrc / ibrc_p2p / ucx"]
    end

    AppA -->|"intra-node IPC\n(同节点跨进程/跨芯)"| IPCA
    IPCA -. "fd / handle 交换" .-> IPCB
    DevA <-->|"零拷贝映射"| IPCA
    IPCB <-->|"零拷贝映射"| DevB

    DevA -->|"GPU/NPU Direct\n(device-buffer RDMA)"| NetA
    NetA -- "RDMA Write/Read (IB/RoCE)" --> NetB
    NetB -->|"GPU/NPU Direct"| DevB
    AppB --- DevB
```

要点:

- **IPC 路径**:同节点通过 `ipcsocket` 交换 device handle / fd,使对端进程直接映射本地 device buffer。
- **RDMA 路径**:跨节点通过 IB verbs(`service/ibvwrap.cc` + `adaptor/net/ibrc*`)发起 device-direct RDMA,避免在 host 上中转。
- 上层调度由 `core/proxy.cc` 负责,根据拓扑(`paths.cc`)选择最优通道。

---

## 7. 框架接入(PyTorch / Paddle Plugin)

```mermaid
flowchart TB
    subgraph Torch["PyTorch 进程"]
        DDP["torch.distributed (DDP/FSDP)"]
        PG["ProcessGroupFlagCX\n(plugin/torch)"]
    end

    subgraph Paddle["Paddle 进程"]
        PFW["paddle.distributed"]
        PPG["FlagCX Backend\n(adaptor_plugin)"]
    end

    subgraph FlagCXLib["libflagcx.so"]
        APIX["Public C API\n(flagcx.h)"]
        CORE2["Core + Adaptors"]
    end

    DDP --> PG --> APIX
    PFW --> PPG --> APIX
    APIX --> CORE2
    CORE2 --> VENDOR["Vendor xCCL / Net SDK"]
```

要点:

- PyTorch 集成走 `plugin/torch`,实现 `c10d::Backend`,把 `all_reduce / all_gather / send / recv` 等调用转换为 FlagCX C API。
- Paddle 集成位于 `adaptor_plugin/`。
- 两者最终落到 `libflagcx.so` 的统一 API,由 Core + Adaptor 选择具体后端。

---

## 维护说明

- 本文档的图基于仓库当前目录结构绘制;若 `flagcx/core`、`flagcx/adaptor`、`flagcx/service`
  或 `plugin/` 下新增/重命名了重要模块,请同步更新对应 Mermaid 块。
- 如需导出图片,可使用 [mermaid-cli](https://github.com/mermaid-js/mermaid-cli)
  (`mmdc -i architecture.md -o architecture.png`),建议把图片放到 `docs/images/` 下,并在本文中引用。
