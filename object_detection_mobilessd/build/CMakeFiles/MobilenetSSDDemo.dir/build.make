# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /proj/mtk09942/object_detection_mobilessd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /proj/mtk09942/object_detection_mobilessd/build

# Include any dependencies generated for this target.
include CMakeFiles/MobilenetSSDDemo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MobilenetSSDDemo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MobilenetSSDDemo.dir/flags.make

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o: CMakeFiles/MobilenetSSDDemo.dir/flags.make
CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o: ../Mobilenet_ssd_demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /proj/mtk09942/object_detection_mobilessd/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o -c /proj/mtk09942/object_detection_mobilessd/Mobilenet_ssd_demo.cpp

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.i"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /proj/mtk09942/object_detection_mobilessd/Mobilenet_ssd_demo.cpp > CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.i

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.s"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /proj/mtk09942/object_detection_mobilessd/Mobilenet_ssd_demo.cpp -o CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.s

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.requires:
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.requires

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.provides: CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/MobilenetSSDDemo.dir/build.make CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.provides.build
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.provides

CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.provides.build: CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o: CMakeFiles/MobilenetSSDDemo.dir/flags.make
CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o: ../TFLib.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /proj/mtk09942/object_detection_mobilessd/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o -c /proj/mtk09942/object_detection_mobilessd/TFLib.cpp

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.i"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /proj/mtk09942/object_detection_mobilessd/TFLib.cpp > CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.i

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.s"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /proj/mtk09942/object_detection_mobilessd/TFLib.cpp -o CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.s

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.requires:
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.requires

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.provides: CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.requires
	$(MAKE) -f CMakeFiles/MobilenetSSDDemo.dir/build.make CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.provides.build
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.provides

CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.provides.build: CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o: CMakeFiles/MobilenetSSDDemo.dir/flags.make
CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o: ../ssd.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /proj/mtk09942/object_detection_mobilessd/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o -c /proj/mtk09942/object_detection_mobilessd/ssd.cpp

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.i"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /proj/mtk09942/object_detection_mobilessd/ssd.cpp > CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.i

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.s"
	../android-ndk-r17b-toolchain-arm64/bin/aarch64-linux-android-g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /proj/mtk09942/object_detection_mobilessd/ssd.cpp -o CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.s

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.requires:
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.requires

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.provides: CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.requires
	$(MAKE) -f CMakeFiles/MobilenetSSDDemo.dir/build.make CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.provides.build
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.provides

CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.provides.build: CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o

# Object files for target MobilenetSSDDemo
MobilenetSSDDemo_OBJECTS = \
"CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o" \
"CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o" \
"CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o"

# External object files for target MobilenetSSDDemo
MobilenetSSDDemo_EXTERNAL_OBJECTS =

MobilenetSSDDemo: CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o
MobilenetSSDDemo: CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o
MobilenetSSDDemo: CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o
MobilenetSSDDemo: CMakeFiles/MobilenetSSDDemo.dir/build.make
MobilenetSSDDemo: CMakeFiles/MobilenetSSDDemo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable MobilenetSSDDemo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MobilenetSSDDemo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MobilenetSSDDemo.dir/build: MobilenetSSDDemo
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/build

CMakeFiles/MobilenetSSDDemo.dir/requires: CMakeFiles/MobilenetSSDDemo.dir/Mobilenet_ssd_demo.cpp.o.requires
CMakeFiles/MobilenetSSDDemo.dir/requires: CMakeFiles/MobilenetSSDDemo.dir/TFLib.cpp.o.requires
CMakeFiles/MobilenetSSDDemo.dir/requires: CMakeFiles/MobilenetSSDDemo.dir/ssd.cpp.o.requires
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/requires

CMakeFiles/MobilenetSSDDemo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MobilenetSSDDemo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/clean

CMakeFiles/MobilenetSSDDemo.dir/depend:
	cd /proj/mtk09942/object_detection_mobilessd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /proj/mtk09942/object_detection_mobilessd /proj/mtk09942/object_detection_mobilessd /proj/mtk09942/object_detection_mobilessd/build /proj/mtk09942/object_detection_mobilessd/build /proj/mtk09942/object_detection_mobilessd/build/CMakeFiles/MobilenetSSDDemo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MobilenetSSDDemo.dir/depend

