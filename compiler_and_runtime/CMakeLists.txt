CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

IF(${BUILD_OS} MATCHES "android_ndk")
    # 3.7.0 required for cross compiling with Android NDK
    #cmake_minimum_required(VERSION 3.7.0)

    PROJECT("neuropilot")
ELSEIF(${BUILD_OS} MATCHES "linux")
    CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

    PROJECT("NeuroPilot-Linux")
ENDIF()


# ------------------------------------------------------------------------------
# Android_NDK build
# ------------------------------------------------------------------------------
IF(${BUILD_OS} MATCHES "android_ndk")
    IF(${TARGET} MATCHES "aarch64")
        SET(NDK_STANDALONE_TOOLCHAIN /proj/mtk40142/neurosw/android-ndk-r18b/ndk/android-r18-toolchain/)
        SET(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-)
        SET(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-clang++)
        SET(SYSROOT ${NDK_STANDALONE_TOOLCHAIN}/sysroot)
        SET(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
        SET(CMAKE_C_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
        SET(CMAKE_CXX_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT} -isystem ${NDK_STANDALONE_TOOLCHAIN}/include/c++/4.9.x")
        LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/libs/arm64)
    ELSEIF(${TARGET} MATCHES "arm")
        SET(NDK_STANDALONE_TOOLCHAIN /proj/mtk40142/neurosw/android-ndk-r18b/ndk/android-r18-toolchain-arm32/)
        SET(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/arm-linux-androideabi-gcc)
        SET(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/arm-linux-androideabi-clang++)
        SET(SYSROOT ${NDK_STANDALONE_TOOLCHAIN}/sysroot)
        SET(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
        SET(CMAKE_C_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
        SET(CMAKE_CXX_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
        LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/libs/arm)
    ENDIF()

    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
ENDIF()


# ------------------------------------------------------------------------------
# Yocto build
# ------------------------------------------------------------------------------
IF(${BUILD_OS} MATCHES "linux")
    SET(CMAKE_C_COMPILER ${CC})
    SET(CMAKE_CXX_COMPILER ${CXX})

    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)
    LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/lib/aarch64-poky-linux)
ENDIF()


# ------------------------------------------------------------------------------
# compiler
# ------------------------------------------------------------------------------
ADD_EXECUTABLE(
    compiler
    api/compiler/sample.cpp)

IF(${BUILD_OS} MATCHES "linux")
    TARGET_LINK_LIBRARIES(compiler dl)
ENDIF()


# ------------------------------------------------------------------------------
# runtime
# ------------------------------------------------------------------------------
ADD_EXECUTABLE(
    runtime
    api/runtime/runtime.cpp)

IF(${BUILD_OS} MATCHES "linux")
    TARGET_LINK_LIBRARIES(runtime dl)
ENDIF()
