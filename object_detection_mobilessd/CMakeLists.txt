CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT("neuropilot")

IF(${BENV} MATCHES "O")
	SET(SYSROOT ${NDK_STANDALONE_TOOLCHAIN}/neuropilot-sysroot)
	add_definitions(-DANDROID_O)
ELSE()
	SET(SYSROOT ${NDK_STANDALONE_TOOLCHAIN}/sysroot)
	add_definitions(-DANDROID_P)
ENDIF()

add_definitions(-DENV=${BENV})
MESSAGE ("-------------Build Android " ${BENV} " version-------------")
MESSAGE ("Tool chain:" ${NDK_STANDALONE_TOOLCHAIN} ", sysroot:" ${SYSROOT})


IF(${TARGET} MATCHES "aarch64")
	SET(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-gcc)
	SET(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/aarch64-linux-android-g++)
	SET(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
	SET(CMAKE_C_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
	SET(CMAKE_CXX_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT} -isystem ${NDK_STANDALONE_TOOLCHAIN}/include/c++/4.9.x")
	INCLUDE_DIRECTORIES(${SYSROOT}/usr/include/tensorflow/contrib/lite/schema)
	INCLUDE_DIRECTORIES(${SYSROOT}/usr/include/)
	LINK_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/libs/arm64)
ELSEIF(${TARGET} MATCHES "arm")
	SET(CMAKE_C_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/arm-linux-androideabi-gcc)
	SET(CMAKE_CXX_COMPILER ${NDK_STANDALONE_TOOLCHAIN}/bin/arm-linux-androideabi-g++)
	SET(CMAKE_FIND_ROOT_PATH ${NDK_STANDALONE_TOOLCHAIN})
	SET(CMAKE_C_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
	SET(CMAKE_CXX_FLAGS "${LINUX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --sysroot=${SYSROOT}")
ENDIF()

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIE -pie")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


ADD_EXECUTABLE(MobilenetSSDDemo Mobilenet_ssd_demo.cpp TFLib.cpp ssd.cpp)

IF(${BENV} MATCHES "O")
	TARGET_LINK_LIBRARIES(MobilenetSSDDemo neuralnetworks log)
ELSE()
	TARGET_LINK_LIBRARIES(MobilenetSSDDemo neuralnetworks log)
ENDIF()
