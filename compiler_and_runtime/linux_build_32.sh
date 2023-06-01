#!/bin/bash
#
# ------------------------------------------------------------------------------
# This script builds linux sample
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
# change YOCTO_BUILD_PATH to yours
export YOCTO_BUILD_PATH=/worktmp/codebase/yocto_0409/build/tmp/work/cortexa7hf-neon-vfpv4-poky-linux-gnueabi/ann/v1-1
export PATH=$YOCTO_BUILD_PATH/recipe-sysroot-native/usr/bin/arm-poky-linux-gnueabi/:$PATH

export SDKTARGETSYSROOT=$YOCTO_BUILD_PATH/recipe-sysroot
export OECORE_NATIVE_SYSROOT=$YOCTO_BUILD_PATH/recipe-sysroot-native

export CXX="arm-poky-linux-gnueabi-clang++ -march=armv7ve -mfpu=neon-vfpv4 -mfloat-abi=hard -mcpu=cortex-a7 --rtlib=compiler-rt --stdlib=libc++ --rtlib=compiler-rt --stdlib=libc++ -mlittle-endian -Wno-error=unused-command-line-argument -Qunused-arguments -Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed --sysroot=$SDKTARGETSYSROOT"
export CC="arm-poky-linux-gnueabi-clang -march=armv7ve -mfpu=neon-vfpv4 -mfloat-abi=hard -mcpu=cortex-a7 --rtlib=compiler-rt --stdlib=libc++ --rtlib=compiler-rt --stdlib=libc++ -mlittle-endian -Wno-error=unused-command-line-argument -Qunused-arguments -Wl,-O1 -Wl,--hash-style=gnu -Wl,--as-needed --sysroot=$SDKTARGETSYSROOT"
# Path to cmake
CMAKE_TOOL=/mtkoss/cmake/cmake-3.13.1-Linux-x86_64/bin/cmake

# Build directory for sample
SAMPLE_BUILD_DIR=yocto_build_32

# Repo directory
SAMPLE_ROOT=$PWD

# ------------------------------------------------------------------------------
# Prepare a build directory for sample
# ------------------------------------------------------------------------------
mkdir $SAMPLE_BUILD_DIR

# ------------------------------------------------------------------------------
# Build sample
# ------------------------------------------------------------------------------
cd $SAMPLE_BUILD_DIR
cmake -DBUILD_OS=linux ../

make

cd $SAMPLE_ROOT


