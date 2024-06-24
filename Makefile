# Inspired by https://github.com/ggerganov/llama.cpp/blob/61665277afde2add00c0d387acb94ed5feb95917/Makefile.

.DEFAULT_GOAL := build

SHELL := /bin/bash

MK_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
MK_FLAGS:= $(wordlist 3, $(words $(MAKEFLAGS)), $(MAKEFLAGS))

#
# System flags
#

ifndef UNAME_S
	UNAME_S := $(shell uname -s)
endif
ifndef UNAME_P
	UNAME_P := $(shell uname -p)
endif
ifndef UNAME_M
	UNAME_M := $(shell uname -m)
endif

ifeq ($(origin CC),default)
	CC := cc
endif
ifeq ($(origin CXX),default)
	CXX := c++
endif
ifdef LLAMA_CUDA
	ifdef LLAMA_CUDA_NVCC
		NVCC := $(LLAMA_CUDA_NVCC)
	else
		NVCC := nvcc
	endif
endif

ifndef LLAMA_NO_CCACHE
	CCACHE := $(shell which ccache)
	ifdef CCACHE
		export CCACHE_SLOPPINESS = time_macros
		CC    := $(CCACHE) $(CC)
		CXX   := $(CCACHE) $(CXX)
		NVCC  := $(CCACHE) $(NVCC)
	endif
endif

## Mac OS + Arm can report x86_64
## ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifndef LLAMA_NO_METAL
		LLAMA_METAL := 1
	endif
	LLAMA_NO_OPENMP := 1 # OpenMP is not supported on macOS
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif
ifdef LLAMA_METAL
	MK_FLAGS += " LLAMA_METAL_EMBED_LIBRARY=1"
endif

#
# Compile flags
#

## standard
MK_CPPFLAGS  = -I$(MK_DIR) -I$(MK_DIR)/llama.cpp -I$(MK_DIR)/llama.cpp/common
MK_CFLAGS    = -std=c11 -fPIC
MK_CXXFLAGS  = -std=c++11 -fPIC

## clock_gettime came in POSIX.1b (1993)
## CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
## posix_memalign came in POSIX.1-2001 / SUSv3
## M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
MK_CPPFLAGS += -D_XOPEN_SOURCE=600

## Somehow in OpenBSD whenever POSIX conformance is specified
## some string functions rely on locale_t availability,
## which was introduced in POSIX.1-2008, forcing us to go higher
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -U_XOPEN_SOURCE -D_XOPEN_SOURCE=700
endif

## RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
## and on macOS its availability depends on enabling Darwin extensions
## similarly on DragonFly, enabling BSD extensions is necessary
ifeq ($(UNAME_S),Darwin)
	MK_CPPFLAGS += -D_DARWIN_C_SOURCE
endif
ifeq ($(UNAME_S),DragonFly)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif

## alloca is a non-standard interface that is not visible on BSDs when
## POSIX conformance is specified, but not all of them provide a clean way
## to enable it in such cases
ifeq ($(UNAME_S),FreeBSD)
	MK_CPPFLAGS += -D__BSD_VISIBLE
endif
ifeq ($(UNAME_S),NetBSD)
	MK_CPPFLAGS += -D_NETBSD_SOURCE
endif
ifeq ($(UNAME_S),OpenBSD)
	MK_CPPFLAGS += -D_BSD_SOURCE
endif

## debug or optimization
ifdef LLAMA_DEBUG
	MK_CFLAGS    += -O0 -g
	MK_CXXFLAGS  += -O0 -g
	MK_LDFLAGS   += -g
	ifeq ($(UNAME_S),Darwin)
		MK_CPPFLAGS += -D_GLIBCXX_ASSERTIONS
	endif
else
	MK_CPPFLAGS += -DNDEBUG
	ifdef LLAMA_FAST
		MK_CFLAGS    += -Ofast
		MK_CXXFLAGS  += -Ofast
	else
		MK_CFLAGS    += -O3
		MK_CXXFLAGS  += -O3
	endif
endif

## warning
MK_CFLAGS    += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function \
				-Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int \
				-Werror=implicit-function-declaration
MK_CXXFLAGS  += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function \
 				-Wmissing-declarations -Wmissing-noreturn
ifdef LLAMA_FATAL_WARNINGS
	MK_CFLAGS   += -Werror
	MK_CXXFLAGS += -Werror
endif

## os specific
### thread
ifneq '' '$(filter $(UNAME_S),Linux Darwin FreeBSD NetBSD OpenBSD Haiku)'
	MK_CFLAGS   += -pthread
	MK_CXXFLAGS += -pthread
endif
### windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
	LWINSOCK2 := -lws2_32
endif

## arch specific
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	MK_CFLAGS     += -march=native -mtune=native
	HOST_CXXFLAGS += -march=native -mtune=native

	# Usage AVX-only
	#MK_CFLAGS   += -mfma -mf16c -mavx
	#MK_CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#MK_CFLAGS   += -mssse3
	#MK_CXXFLAGS += -mssse3
endif
ifneq '' '$(findstring mingw,$(shell $(CC) -dumpmachine))'
	# The stack is only 16-byte aligned on Windows, so don't let gcc emit aligned moves.
	# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412
	# https://github.com/ggerganov/llama.cpp/issues/2922
	MK_CFLAGS   += -Xassembler -muse-unaligned-vector-move
	MK_CXXFLAGS += -Xassembler -muse-unaligned-vector-move

	# Target Windows 8 for PrefetchVirtualMemory
	MK_CPPFLAGS += -D_WIN32_WINNT=0x602
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	# Nvidia Jetson
	MK_CFLAGS   += -mcpu=native
	MK_CXXFLAGS += -mcpu=native
	JETSON_RELEASE_INFO = $(shell jetson_release)
	ifdef JETSON_RELEASE_INFO
		ifneq ($(filter TX2%,$(JETSON_RELEASE_INFO)),)
			CC = aarch64-unknown-linux-gnu-gcc
			cxx = aarch64-unknown-linux-gnu-g++
		endif
	endif
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	MK_CFLAGS   += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
	MK_CXXFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	MK_CFLAGS   += -mfp16-format=ieee -mno-unaligned-access
	MK_CXXFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		MK_CFLAGS   += -mcpu=power9
		MK_CXXFLAGS += -mcpu=power9
	endif
endif
ifneq ($(filter ppc64le%,$(UNAME_M)),)
	MK_CFLAGS   += -mcpu=powerpc64le
	MK_CXXFLAGS += -mcpu=powerpc64le
	CUDA_POWER_ARCH = 1
endif
ifneq ($(filter loongarch64%,$(UNAME_M)),)
	MK_CFLAGS   += -mlasx
	MK_CXXFLAGS += -mlasx
endif
ifneq ($(filter riscv64%,$(UNAME_M)),)
	MK_CFLAGS   += -march=rv64gcv -mabi=lp64d
	MK_CXXFLAGS += -march=rv64gcv -mabi=lp64d
endif

## platform specific
### apple metal
ifdef LLAMA_METAL
	MK_LDFLAGS += -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
endif
### cuda
ifdef LLAMA_CUDA
	ifneq ($(wildcard /opt/cuda),)
		CUDA_PATH ?= /opt/cuda
	else
		CUDA_PATH ?= /usr/local/cuda
	endif
	MK_CPPFLAGS  += -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/$(UNAME_M)-linux/include
	MK_LDFLAGS += -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L$(CUDA_PATH)/lib64 -L/usr/lib64 -L$(CUDA_PATH)/targets/$(UNAME_M)-linux/lib -L$(CUDA_PATH)/lib64/stubs -L/usr/lib/wsl/lib
	ifneq ($(filter aarch64%,$(UNAME_M)),)
		ifneq ($(wildcard $(CUDA_PATH)/targets/sbsa-linux),)
			MK_CPPFLAGS += -I$(CUDA_PATH)/targets/sbsa-linux/include
			MK_LDFLAGS  += -L$(CUDA_PATH)/targets/sbsa-linux/lib
		endif
	endif
endif
### hipblas
ifdef LLAMA_HIPBLAS
	ifeq ($(wildcard /opt/rocm),)
		ROCM_PATH ?= /usr
	else
		ROCM_PATH ?= /opt/rocm
	endif
	MK_LDFLAGS += -L$(ROCM_PATH)/lib -Wl,-rpath=$(ROCM_PATH)/lib
	MK_LDFLAGS += -L$(ROCM_PATH)/lib64 -Wl,-rpath=$(ROCM_PATH)/lib64
	MK_LDFLAGS += -lhipblas -lamdhip64 -lrocblas -lrocsolver -lamd_comgr -lhsa-runtime64 -lrocsparse -ldrm -ldrm_amdgpu
endif
### openmp
ifndef LLAMA_NO_OPENMP
	# OpenMP cannot be statically linked.
	MK_CFLAGS   += -fopenmp
	MK_CXXFLAGS += -fopenmp
endif
### openblas
ifdef LLAMA_OPENBLAS
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas)
endif
### openblas64
ifdef LLAMA_OPENBLAS64
	MK_CFLAGS   += $(shell pkg-config --cflags-only-other openblas64)
	MK_LDFLAGS  += $(shell pkg-config --libs openblas64)
endif
### blis
ifdef LLAMA_BLIS
	MK_LDFLAGS  += -lblis -L/usr/local/lib
endif

## get compiler flags
GF_CC := $(CC)
ifdef LLAMA_CUDA
	GF_CC := $(NVCC) -std=c++11 2>/dev/null .c -Xcompiler
endif
include $(MK_DIR)/llama.cpp/scripts/get-flags.mk

## combine build flags with cmdline overrides
override CPPFLAGS  := $(MK_CPPFLAGS) $(CPPFLAGS)
override CFLAGS    := $(CPPFLAGS) $(MK_CFLAGS) $(GF_CFLAGS) $(CFLAGS)
override CXXFLAGS  := $(MK_CXXFLAGS) $(CXXFLAGS) $(GF_CXXFLAGS) $(CPPFLAGS)
override LDFLAGS   := $(MK_LDFLAGS) $(LDFLAGS)

#
# Helper function
#

## BUILD_INFO prints out the build info
define BUILD_INFO
	@echo "I llama-box build info:"
	@echo "I UNAME_S:   $(UNAME_S)"
	@echo "I UNAME_P:   $(UNAME_P)"
	@echo "I UNAME_M:   $(UNAME_M)"
	@echo "I CFLAGS:    $(CFLAGS)"
	@echo "I CXXFLAGS:  $(CXXFLAGS)"
	@echo "I LDFLAGS:   $(LDFLAGS)"
	@echo "I CC:        $(shell $(CC)   --version | head -n 1)"
	@echo "I CXX:       $(shell $(CXX)  --version | head -n 1)"
	@echo
endef

## GET_OBJ_FILE replaces .c, .cpp, and .cu file endings with .o
define GET_OBJ_FILE
	$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(1))))
endef

#
# Main function
#

##
## clean
##

.PHONY: clean
clean:
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "I cleaning llama.cpp"
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	make -C $(MK_DIR)/llama.cpp -j $(MK_FLAGS) clean
	rm -f $(MK_DIR)/llama.cpp/ggml-metal-embed.metal
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "I cleaning llama-box"
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	$(call BUILD_INFO)
	rm -rf $(MK_DIR)/build/bin
	find $(MK_DIR)/llama-box -type f -name "*.o" -delete
	rm -f $(MK_DIR)/llama-box/version.cpp

##
## build
##

llama.cpp/libllama.a:
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "I building llama.cpp"
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	make -C $(MK_DIR)/llama.cpp -j $(MK_FLAGS) libllama.a

llama-box/version.cpp: $(wildcard .git/index) llama-box/scripts/version.sh
	@sh $(MK_DIR)/llama-box/scripts/version.sh > $@.tmp
	@if ! cmp -s $@ $@.tmp; then mv $@.tmp $@; else rm $@.tmp; fi

llama-box/version.o: llama-box/version.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)

llama-box: llama-box/main.cpp llama-box/version.o llama.cpp/libllama.a
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "I building llama.cpp"
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	make -C $(MK_DIR)/llama.cpp -j $(MK_FLAGS) libllama.a
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	@echo "I building llama-box"
	@echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
ifeq ($(_WIN32),1)
SUFFIX := .exe
endif
	$(call BUILD_INFO)
	mkdir -p $(MK_DIR)/build/bin
	$(CXX) $(CXXFLAGS) -c $< -o $(call GET_OBJ_FILE, $<)
	$(CXX) $(CXXFLAGS) $(wildcard llama.cpp/*.o) $(wildcard llama.cpp/ggml-cuda/*.o) $(wildcard llama.cpp/ggml-cuda/template-instances/*.o) $(filter-out %.h %.hpp %.a $<,$^) $(call GET_OBJ_FILE, $<) -o $(MK_DIR)/build/bin/$@$(SUFFIX) $(LDFLAGS) $(LWINSOCK2)

.PHONY: build
build: llama-box
