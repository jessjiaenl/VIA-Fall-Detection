#!/bin/bash
#
# ------------------------------------------------------------------------------
# This script builds linux sample
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
#change the the below paths to yours
#export PATH=/worktmp/codebase/yocto/build/tmp/work/aarch64-poky-linux/ann/v1-1/recipe-sysroot-native/usr/bin/aarch64-poky-linux:$PATH
#export SDKTARGETSYSROOT=/worktmp/codebase/yocto/build/tmp/work/aarch64-poky-linux/ann/v1-1/recipe-sysroot
#export OECORE_NATIVE_SYSROOT=/worktmp/codebase/yocto/build/tmp/work/aarch64-poky-linux/ann/v1-1/recipe-sysroot-native

export CXX="aarch64-linux-gnu-g++"
export CC="aarch64-linux-gnu-gcc"

# Path to cmake
#CMAKE_TOOL=/mtkoss/cmake/cmake-3.13.1-Linux-x86_64/bin/cmake

# Build directory for sample
SAMPLE_BUILD_DIR=yocto_build

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


