# NCCL Wrapper for SDCCL
# Produces libnccl.so that wraps NCCL API via SDCCL

SDCCL_HOME ?= $(abspath ../..)
CUDA_HOME   ?= /usr/local/cuda
NCCL_HOME     ?= /usr/local/nccl
REAL_NCCL_LIB ?= $(NCCL_HOME)/lib/libnccl.so.2
BUILDDIR      ?= $(abspath build)

LIBDIR     := $(BUILDDIR)/lib
OBJDIR     := $(BUILDDIR)/obj
TARGET     := libnccl.so

# Source files
SRCS := $(wildcard src/*.cc)
OBJS := $(SRCS:src/%.cc=$(OBJDIR)/%.o)

# Compiler flags
CXX      := g++
CXXFLAGS := -fPIC -fvisibility=default -std=c++11 -Wall -Wno-sign-compare -O2 -g \
            -DREAL_NCCL_PATH=\"$(REAL_NCCL_LIB)\"
INCLUDES := -I$(NCCL_HOME)/include \
            -I$(SDCCL_HOME)/sdccl/include \
            -I$(CUDA_HOME)/include
LDFLAGS  := -shared \
            -L$(SDCCL_HOME)/build/lib -lsdccl \
            -L$(CUDA_HOME)/lib64 -lcudart \
            -ldl \
            -Wl,-rpath,$(SDCCL_HOME)/build/lib \
            -Wl,-rpath,$(CUDA_HOME)/lib64

.PHONY: all clean

all: $(LIBDIR)/$(TARGET)

$(LIBDIR)/$(TARGET): $(OBJS)
	@mkdir -p $(LIBDIR)
	@echo "Linking   $@"
	@$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: src/%.cc
	@mkdir -p $(OBJDIR)
	@echo "Compiling $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

-include $(OBJS:.o=.d)

clean:
	@rm -rf $(BUILDDIR)
