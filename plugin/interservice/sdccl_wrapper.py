# SPDX-License-Identifier: Apache-2.0
# reference https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py

import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

# === export types and functions from sdccl to Python ===
# for the original sdccl definition, please check
# https://github.com/FlagOpen/SDCCL/blob/main/sdccl/include/sdccl.h

sdcclResult_t = ctypes.c_int
sdcclDataType_t = ctypes.c_int
sdcclRedOp_t = ctypes.c_int
sdcclMemcpyType_t = ctypes.c_int
sdcclMemType_t = ctypes.c_int
sdcclEventType_t = ctypes.c_int
sdcclIpcMemHandle_t = ctypes.c_void_p

sdcclHandlerGroup_t = ctypes.c_void_p
sdcclComm_t = ctypes.c_void_p
sdcclEvent_t = ctypes.c_void_p
sdcclStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

class sdcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 256)]
sdcclUniqueId_t = ctypes.POINTER(sdcclUniqueId)

DEVICE_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t)
DEVICE_MEMCPY_FUNCTYPE = ctypes.CFUNCTYPE(
    sdcclResult_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    sdcclMemcpyType_t, sdcclStream_t
)
DEVICE_MEMSET_FUNCTYPE = ctypes.CFUNCTYPE(
    sdcclResult_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    sdcclMemType_t, sdcclStream_t
)
DEVICE_MALLOC_FUNCTYPE = ctypes.CFUNCTYPE(
    sdcclResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
    sdcclMemType_t, sdcclStream_t
)
DEVICE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(
    sdcclResult_t, ctypes.c_void_p, sdcclMemType_t, sdcclStream_t
)
SET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.c_int)
GET_DEVICE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(ctypes.c_int))
GET_DEVICE_COUNT_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(ctypes.c_int))
GET_VENDOR_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.c_char_p)
HOST_GET_DEVICE_POINTER_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)

STREAM_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(sdcclStream_t))
STREAM_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclStream_t)
STREAM_COPY_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(sdcclStream_t), ctypes.c_void_p)
STREAM_FREE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclStream_t)
STREAM_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclStream_t)
STREAM_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclStream_t)
STREAM_WAIT_EVENT_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclStream_t, sdcclEvent_t)

EVENT_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(sdcclEvent_t), sdcclEventType_t)
EVENT_DESTROY_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclEvent_t)
EVENT_RECORD_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclEvent_t, sdcclStream_t)
EVENT_SYNCHRONIZE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclEvent_t)
EVENT_QUERY_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclEvent_t)

IPC_MEM_HANDLE_CREATE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.POINTER(sdcclIpcMemHandle_t), ctypes.POINTER(ctypes.c_size_t))
IPC_MEM_HANDLE_GET_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclIpcMemHandle_t, ctypes.c_void_p)
IPC_MEM_HANDLE_OPEN_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclIpcMemHandle_t, ctypes.POINTER(ctypes.c_void_p))
IPC_MEM_HANDLE_CLOSE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, ctypes.c_void_p)
IPC_MEM_HANDLE_FREE_FUNCTYPE = ctypes.CFUNCTYPE(sdcclResult_t, sdcclIpcMemHandle_t)

class sdcclDeviceHandle(ctypes.Structure):
    _fields_ = [
        # Basic functions
        ("deviceSynchronize", DEVICE_SYNCHRONIZE_FUNCTYPE),
        ("deviceMemcpy", DEVICE_MEMCPY_FUNCTYPE),
        ("deviceMemset", DEVICE_MEMSET_FUNCTYPE),
        ("deviceMalloc", DEVICE_MALLOC_FUNCTYPE),
        ("deviceFree", DEVICE_FREE_FUNCTYPE),
        ("setDevice", SET_DEVICE_FUNCTYPE),
        ("getDevice", GET_DEVICE_FUNCTYPE),
        ("getDeviceCount", GET_DEVICE_COUNT_FUNCTYPE),
        ("getVendor", GET_VENDOR_FUNCTYPE),
        ("hostGetDevicePointer", HOST_GET_DEVICE_POINTER_FUNCTYPE),
        # Stream functions
        ("streamCreate", STREAM_CREATE_FUNCTYPE),
        ("streamDestroy", STREAM_DESTROY_FUNCTYPE),
        ("streamCopy", STREAM_COPY_FUNCTYPE),
        ("streamFree", STREAM_FREE_FUNCTYPE),
        ("streamSynchronize", STREAM_SYNCHRONIZE_FUNCTYPE),
        ("streamQuery", STREAM_QUERY_FUNCTYPE),
        ("streamWaitEvent", STREAM_WAIT_EVENT_FUNCTYPE),
        # Event functions
        ("eventCreate", EVENT_CREATE_FUNCTYPE),
        ("eventDestroy", EVENT_DESTROY_FUNCTYPE),
        ("eventRecord", EVENT_RECORD_FUNCTYPE),
        ("eventSynchronize", EVENT_SYNCHRONIZE_FUNCTYPE),
        ("eventQuery", EVENT_QUERY_FUNCTYPE),
        # IpcMemHandle functions
        ("ipcMemHandleCreate", IPC_MEM_HANDLE_CREATE_FUNCTYPE),
        ("ipcMemHandleGet", IPC_MEM_HANDLE_GET_FUNCTYPE),
        ("ipcMemHandleOpen", IPC_MEM_HANDLE_OPEN_FUNCTYPE),
        ("ipcMemHandleClose", IPC_MEM_HANDLE_CLOSE_FUNCTYPE),
        ("ipcMemHandleFree", IPC_MEM_HANDLE_FREE_FUNCTYPE),
    ]
sdcclDeviceHandle_t = ctypes.POINTER(sdcclDeviceHandle)

class sdcclHandlerGroup(ctypes.Structure):
    _fields_ = [
        ("uniqueId", sdcclUniqueId_t),
        ("comm", sdcclComm_t),
        ("devHandle", sdcclDeviceHandle_t),
    ]
sdcclHandlerGroup_t = ctypes.POINTER(sdcclHandlerGroup)


class sdcclDataTypeEnum:
    sdcclInt8 = 0
    sdcclChar = 0
    sdcclUint8 = 1
    sdcclInt32 = 2
    sdcclInt = 2
    sdcclUint32 = 3
    sdcclInt64 = 4
    sdcclUint64 = 5
    sdcclFloat16 = 6
    sdcclHalf = 6
    sdcclFloat32 = 7
    sdcclFloat = 7
    sdcclFloat64 = 8
    sdcclDouble = 8
    sdcclBfloat16 = 9
    sdcclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.sdcclInt8
        if dtype == torch.uint8:
            return cls.sdcclUint8
        if dtype == torch.int32:
            return cls.sdcclInt32
        if dtype == torch.int64:
            return cls.sdcclInt64
        if dtype == torch.float16:
            return cls.sdcclFloat16
        if dtype == torch.float32:
            return cls.sdcclFloat32
        if dtype == torch.float64:
            return cls.sdcclFloat64
        if dtype == torch.bfloat16:
            return cls.sdcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


class sdcclRedOpTypeEnum:
    sdcclSum = 0
    sdcclProd = 1
    sdcclMax = 2
    sdcclMin = 3
    sdcclAvg = 4
    sdcclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.sdcclSum
        if op == ReduceOp.PRODUCT:
            return cls.sdcclProd
        if op == ReduceOp.MAX:
            return cls.sdcclMax
        if op == ReduceOp.MIN:
            return cls.sdcclMin
        if op == ReduceOp.AVG:
            return cls.sdcclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class SDCCLLibrary:
    exported_functions = [
        Function("sdcclHandleInit", sdcclResult_t,
                [ctypes.POINTER(sdcclHandlerGroup_t)]),
        Function("sdcclHandleFree", sdcclResult_t,
                [sdcclHandlerGroup_t]),
        Function("sdcclGetErrorString", ctypes.c_char_p, [sdcclResult_t]),
        Function("sdcclGetVersion", sdcclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        Function("sdcclGetUniqueId", sdcclResult_t,
                [ctypes.POINTER(ctypes.POINTER(sdcclUniqueId))]),
        # Note that sdcclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("sdcclCommInitRank", sdcclResult_t, [
            ctypes.POINTER(sdcclComm_t), ctypes.c_int, ctypes.POINTER(sdcclUniqueId),
            ctypes.c_int
        ]),
        # Note that sdcclStream_t is a pointer type, so the last argument
        # is a pointer
        Function("sdcclAllReduce", sdcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, sdcclDataType_t,
            sdcclRedOp_t, sdcclComm_t, sdcclStream_t
        ]),

        # Note that sdcclStream_t is a pointer type, so the last argument
        # is a pointer
        Function("sdcclReduce", sdcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, sdcclDataType_t,
            sdcclRedOp_t, ctypes.c_int, sdcclComm_t, sdcclStream_t
        ]),

        # Note that sdcclStream_t is a pointer type, so the last argument
        # is a pointer
        Function("sdcclAllGather", sdcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, sdcclDataType_t,
            sdcclComm_t, sdcclStream_t
        ]),

        # Note that sdcclStream_t is a pointer type, so the last argument
        # is a pointer
        Function("sdcclReduceScatter", sdcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, sdcclDataType_t,
            sdcclRedOp_t, sdcclComm_t, sdcclStream_t
        ]),

        Function("sdcclSend", sdcclResult_t, [
            buffer_type, ctypes.c_size_t, sdcclDataType_t, ctypes.c_int,
            sdcclComm_t, sdcclStream_t
        ]),

        Function("sdcclRecv", sdcclResult_t, [
            buffer_type, ctypes.c_size_t, sdcclDataType_t, ctypes.c_int,
            sdcclComm_t, sdcclStream_t
        ]),

        Function("sdcclBroadcast", sdcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, sdcclDataType_t,
            ctypes.c_int, sdcclComm_t, sdcclStream_t
        ]),

        Function("sdcclGroupStart", sdcclResult_t, [
            sdcclComm_t
        ]),

        Function("sdcclGroupEnd", sdcclResult_t, [
            sdcclComm_t
        ]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # sdcclResult_t sdcclCommDestroy(sdcclComm_t comm);
        Function("sdcclCommDestroy", sdcclResult_t, [sdcclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):


        try:
            if so_file not in SDCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                SDCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = SDCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            raise e

        if so_file not in SDCCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in SDCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            SDCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = SDCCLLibrary.path_to_dict_mapping[so_file]

        # init sdccl handler to call device-related apis
        self.handler = sdcclHandlerGroup_t()
        self.SDCCL_CHECK(self._funcs["sdcclHandleInit"](ctypes.byref(self.handler)))
    
    def __del__(self):
        # free sdccl handler
        self.SDCCL_CHECK(self._funcs["sdcclHandleFree"](self.handler))

    def sdcclGetErrorString(self, result: sdcclResult_t) -> str:
        return self._funcs["sdcclGetErrorString"](result).decode("utf-8")

    def SDCCL_CHECK(self, result: sdcclResult_t) -> None:
        if result != 0:
            error_str = self.sdcclGetErrorString(result)
            raise RuntimeError(f"SDCCL error: {error_str}")

    def sdcclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.SDCCL_CHECK(self._funcs["sdcclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def sdcclGetUniqueId(self) -> sdcclUniqueId:
        unique_id = ctypes.POINTER(sdcclUniqueId)()
        self.SDCCL_CHECK(self._funcs["sdcclGetUniqueId"](
            ctypes.byref(unique_id)))
        return unique_id

    def unique_id_from_bytes(self, data: bytes) -> sdcclUniqueId:
        """
        Reconstructs an `ncclUniqueId` object from bytes data.
        Args:
            data: Must be a 128-byte data block (matching NCCL's unique_id).
        Returns:
            ncclUniqueId: The reconstructed NCCL Unique ID object.
        Raises:
            ValueError: If the input data length is not 128 bytes.
        """
        if len(data) != 256:
            raise ValueError(
                f"Expected 256 bytes for ncclUniqueId, got {len(data)} bytes")

        unique_id = sdcclUniqueId()
        ctypes.memmove(ctypes.addressof(unique_id.internal), data, 256)
        return unique_id

    def sdcclCommInitRank(self, world_size: int, unique_id: sdcclUniqueId,
                         rank: int) -> sdcclComm_t:
        comm = sdcclComm_t()
        self.SDCCL_CHECK(self._funcs["sdcclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def sdcclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: sdcclComm_t,
                      stream: sdcclStream_t) -> None:
        # `datatype` actually should be `sdcclDataType_t`
        # and `op` should be `sdcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SDCCL_CHECK(self._funcs["sdcclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def sdcclReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, root: int, comm: sdcclComm_t,
                      stream: sdcclStream_t) -> None:
        # `datatype` actually should be `sdcclDataType_t`
        # and `op` should be `sdcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SDCCL_CHECK(self._funcs["sdcclReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, root, comm,
                                                     stream))

    def sdcclReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, op: int, comm: sdcclComm_t,
                          stream: sdcclStream_t) -> None:
        # `datatype` actually should be `sdcclDataType_t`
        # and `op` should be `sdcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SDCCL_CHECK(self._funcs["sdcclReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def sdcclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: sdcclComm_t,
                      stream: sdcclStream_t) -> None:
        # `datatype` actually should be `sdcclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.SDCCL_CHECK(self._funcs["sdcclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def sdcclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: sdcclComm_t, stream: sdcclStream_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclSend"](sendbuff, count, datatype,
                                                dest, comm, stream))

    def sdcclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: sdcclComm_t, stream: sdcclStream_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def sdcclBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, root: int, comm: sdcclComm_t,
                      stream: sdcclStream_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def sdcclGroupStart(self, comm: sdcclComm_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclGroupStart"](comm))

    def sdcclGroupEnd(self, comm: sdcclComm_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclGroupEnd"](comm))

    def sdcclCommDestroy(self, comm: sdcclComm_t) -> None:
        self.SDCCL_CHECK(self._funcs["sdcclCommDestroy"](comm))

    def adaptor_stream_create(self):
        new_stream = sdcclStream_t()
        self.SDCCL_CHECK(self.handler.contents.devHandle.contents.streamCreate(ctypes.byref(new_stream)))
        return new_stream

    def adaptor_stream_copy(self, old_stream):
        new_stream = sdcclStream_t()
        self.SDCCL_CHECK(self.handler.contents.devHandle.contents.streamCopy(ctypes.byref(new_stream), ctypes.c_void_p(old_stream.cuda_stream)))
        return new_stream

    def adaptor_stream_free(self, stream):
        self.SDCCL_CHECK(self.handler.contents.devHandle.contents.streamFree(stream))

    def adaptor_stream_destroy(self, stream):
        self.SDCCL_CHECK(self.handler.contents.devHandle.contents.streamDestroy(stream))
    
    def sync_stream(self, stream):
        self.SDCCL_CHECK(self.handler.contents.devHandle.contents.streamSynchronize(stream))


__all__ = [
    "SDCCLLibrary", "sdcclDataTypeEnum", "sdcclRedOpTypeEnum", "sdcclUniqueId",
    "sdcclHandlerGroup_t", "sdcclComm_t", "sdcclStream_t", "sdcclEvent_t", "buffer_type"
]
