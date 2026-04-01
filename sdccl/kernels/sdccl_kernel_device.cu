#include "comm.h"
#include "sdccl.h"
#include "sdccl_kernel.h"
#include "device_api/sdccl_device.h"

SDCCL_DEVICE_DECORATOR size_t
getSdcclDataTypeSizeDevice(sdcclDataType_t dtype) {
  switch (dtype) {
    // case sdcclInt8:
    case sdcclChar:
      return sizeof(char); // 1 byte
    case sdcclUint8:
      return sizeof(unsigned char); // 1 byte
    // case sdcclInt32:
    case sdcclInt:
      return sizeof(int); // 4 bytes
    case sdcclUint32:
      return sizeof(unsigned int); // 4 bytes
    case sdcclInt64:
      return sizeof(long long); // 8 bytes
    case sdcclUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case sdcclFloat16:
    case sdcclHalf:
      return 2; // Half precision float is 2 bytes
    // case sdcclFloat32:
    case sdcclFloat:
      return sizeof(float); // 4 bytes
    // case sdcclFloat64:
    case sdcclDouble:
      return sizeof(double); // 8 bytes
    case sdcclBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      return 0;
  }
}
