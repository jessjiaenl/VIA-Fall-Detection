#!/bin/bash
#
# ------------------------------------------------------------------------------
# This script builds android_ndk sample
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
# Build directory for sample
SAMPLE_BUILD_DIR=ndk_build

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
#Setup 32 bit bit build
#cmake -DBUILD_OS=android_ndk -DTARGET=arm ../
#Setup 64 bit build
cmake -DBUILD_OS=android_ndk -DTARGET=aarch64 ../

make

cd $SAMPLE_ROOT


