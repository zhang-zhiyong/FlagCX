CUDA_HOME ?= /usr/local/cuda

INCLUDE_DIRS = \
    . \
    $(CUDA_HOME)/include

CXXFLAGS = $(foreach dir,$(INCLUDE_DIRS),-I$(dir)) \
           -fPIC -fvisibility=hidden -Wall -g

LDFLAGS = -shared \
          -L$(CUDA_HOME)/lib64 -lcudart -lcuda \
          -Wl,-rpath,$(CUDA_HOME)/lib64

build: libsdccl-device-example.so

libsdccl-device-example.so: plugin.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libsdccl-device-example.so
