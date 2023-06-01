
.__   __.  _______  __    __  .______        ______   .______    __   __        ______   .___________.        _______.     ___      .___  ___. .______    __       _______
|  \ |  | |   ____||  |  |  | |   _  \      /  __  \  |   _  \  |  | |  |      /  __  \  |           |       /       |    /   \     |   \/   | |   _  \  |  |     |   ____|
|   \|  | |  |__   |  |  |  | |  |_)  |    |  |  |  | |  |_)  | |  | |  |     |  |  |  | `---|  |----`      |   (----`   /  ^  \    |  \  /  | |  |_)  | |  |     |  |__
|  . `  | |   __|  |  |  |  | |      /     |  |  |  | |   ___/  |  | |  |     |  |  |  |     |  |            \   \      /  /_\  \   |  |\/|  | |   ___/  |  |     |   __|
|  |\   | |  |____ |  `--'  | |  |\  \----.|  `--'  | |  |      |  | |  `----.|  `--'  |     |  |        .----)   |    /  _____  \  |  |  |  | |  |      |  `----.|  |____
|__| \__| |_______| \______/  | _| `._____| \______/  | _|      |__| |_______| \______/      |__|        |_______/    /__/     \__\ |__|  |__| | _|      |_______||_______|

The package structure is as following:
.
├── api
│   ├── compiler
│   │   ├── CompilerAPI.h
│   │   └── sample.cpp
│   └── runtime
│       ├── RuntimeAPI.h
│       └── runtime.cpp
│
├── build
│   ├── compiler
│   └── runtime
│
├── host
│   ├── bin
│   │   ├── mtknn-hexdump
│   │   ├── mtknn-options
│   │   ├── ncc-estimator
│   │   ├── ncc-simulator
│   │   ├── ncc-tflite
│   │   └── neuronrt
│   └── lib
│       ├── libc++abi.so.1
│       ├── libc++.so.1
│       ├── libncc-compiler.so
│       └── libneuron_runtime.so
└── target
    ├── bin
    │   ├── ncc-tflite
    │   └── neuronrt
    └── lib
        ├── libncc-compiler.so
        └── libneuron_runtime.so

====================================================================================================
1. Directory description
- api
    - compiler
        - CompilerAPI.h
            Provide APIs to convert a tflite model to a offline compiled networks by Neuron compiler collection (i.e., ncc).
        - sample.cpp
            Provide sample code to convert a tflite model to a offline compiled networks by Neuron compiler collection (i.e., ncc).
    - runtime
        - RuntimeAPI.h
            Provide APIs to run a offline compiled networks by Neuron compiler collection (i.e., ncc).
        - runtime.cpp
            Provide sample code to run a offline compiled networks by Neuron compiler collection (i.e., ncc).
- build
    - compiler
        build binary of api/compiler
    - runtime
        build binary of api/runtime
- host
    - bin
        binarys that run on host
    - lib
        libs what binary need
- target
    - bin
        binarys that run on devices
    - lib
        libs what binary need
        libs what sample need

2. Setup the Android NDK standalone toolchain
    - Refer to NeuroPilot website for more information.

3. Build
- android_ndk build
    - In CMakeLists.txt, change NDK_STANDALONE_TOOLCHAIN to correct path
    - auto build
        $./ndk_build.sh
    - manual build
        - Make a build folder. And go to the build folder
            $mkdir build
            $cd build
        - Setup 32 bit / 64 bit build
            - 32bit
            $cmake -DTARGET=arm ../
            - 64bit
            $cmake -DTARGET=aarch64 ../
        - Build the application
            $make
    - Check the output executable(compiler / runtime) in ndk_build folder.
- linux build
    - In linux_build.sh, change build Environment to yours
    - auto build
        $./linux_build.sh
    - Check the output executable(compiler / runtime) in yocto_build folder.

4. Run
- Prepare
    - run push.bat to push input/model/binary/libs to your device
      reminder: modify push.bat for your platform before run, for example mt8168 or mt8183 or ...

- use ncc-tflite to convert tflite model to dla format
    #adb shell "export LD_LIBRARY_PATH=/data/local/tmp/lib; ./data/local/tmp/ncc-tflite -arch vpu /data/local/tmp/model/mobilenet_v1_1.0_224_quant.tflite -v"

    .dla file will be build in /data/local/tmp/model/mobilenet_v2_1.0_224_quant.dla

- use runtime to convert tflite model to dla format
    - Prepare your test image bin file.
        - Use the python script to convert the image file.(BMP and JPG files are acceptable.)
            Resize the image to width:300, height:300, and convert the resized image into a byte array with dimension: [1, 300, 300, 3]
            $python process_input_image.py voc_boat_pi.jpg 300 300
        - You will get a bin file converted from the image file.
            voc_boat_pi.jpg_300300.bin
    - Prepare your TFLite model file.

    - Run inference with the converted bin file.
        - Available options
            -m, dla path
            -l: labels path
            -i, input bin
        - Example
        Inference with mobilenet_v1_1.0_224_quant.dla
        #adb shell "export LD_LIBRARY_PATH=/data/local/tmp/lib; ./data/local/tmp/runtime -m /data/local/tmp/model/mobilenet_v1_1.0_224_quant.dla -l /data/local/tmp/model/labels_imagenet_slim.txt -i /data/local/tmp/input/grace_hopper.bmp_224224.bin"

        You can see the log as follow
        The required size of the input buffer is 150528
        The required size of the output buffer is 1001
        Dump the profiled QoS Data:
        executing boost value = 100
        SubCmd[0][0]:
        execution time = 16244923
        boost value = 100
        bandwidth = 0
        The top index is 653: military uniform

5. host & target:
    Executables and libraries for x86-64 and aarch64 (created by Android NDK-18) respectively.
    We provide all of our tools, executables, and libraries in host build. For target build, on the
    other hand, only a part of them are provided (such as tool for evaluting model performance).

6. Executables:
    6.1 mtknn-hexdump:
        Like a objdump tool, this helps dump a MDLA pattern[1].
        Usage:
            $ mtknn-hexdump PATTERN.hex

    6.2 mtknn-options:
        Since Neuron softwares provide many environment variable to specify special usage, this tool
        helps user to clearly see the status of the environment variables.
        Please see http://wiki.mediatek.inc/display/CIT/MTKNN+Software+Configurable+API for further
        description about the environment variables.
        Usage:
            $ mtknn-options  # no arguments needed

    6.3 ncc-estimator:
        For estimating performance carried out by MDLA, including hardware utilization rate and
        memory requirments, ...etc.
        Usage:
            $ ncc-estimator MODEL.tflite

    6.4 ncc-simulator:
        Tool for compiling a tflite model into MDLA commands and execute them on the CModel.
        You should also have libCModel.so which is provided by CModel team and export
        MTKNN_MDLA_CMODEL_LIB_PATH to the location of libCModel.so.
        (e.g., $ export MTKNN_MDLA_CMODEL_LIB_PATH=PATH_TO_CMODEL/libCModel.so)
        Usage: ncc-simulator [OPTION] model_name.tflite
             -arch <target>          : Specify the architecture to build for
                                       Use -arch=? to show available targets
                                       Use -arch=<target1>,<target2>,... to specify multiple targets
             -e                      : Enable execution mode
             -i <file>               : Specify an input bin file
             -o <file>               : Specify an output bin file or DLA file
             -R                      : Rewrite IO data type to FP16 for FP32 networks
             -s                      : Show tensors and nodes in the tflite model
             -v                      : Verbose mode
             --show-exec-plan        : Show execution plan
             --show-supported-layers : Show supported layers by target
             --show-memory-summary   : Show memory allocation summary
             --relax-fp32            : Hint compiler to compute fp32 models using fp16
             --symmetric-int8        : Hint compiler to use symmetric 8-bit mode
             --l1-size-kb            : Hint compiler with the size of L1 memory
             --l2-size-kb            : Hint compiler with the size of L2 memory
                                       (--l1-size-kb and --l2-size-kb override env var.)
             --help                  : Show this usage

        Example:
            $ ./ncc-simulator -i input.bin -o output.bin -e <model.tflite> # For single input and output
            $ ./ncc-simulator -i input1.bin -i input2.bin -i input3.bin -o output1.bin \
                              -o output2.bin -e <model.tflite> # For multiple inputs and outputs

    6.5 ncc-tflite:
        Tool for generating static-time compiled networks (.dla, Deep Learning Archieve, file) and
        MDLA pattern[1].
        Usage: ncc-tflite [OPTION] model_name.tflite
             -arch <target>          : Specify the architecture to build for
                                       Use -arch=? to show available targets
                                       Use -arch=<target1>,<target2>,... to specify multiple targets
             -e                      : Enable execution mode
             -i <file>               : Specify an input bin file
             -o <file>               : Specify an output bin file or DLA file
             -R                      : Rewrite IO data type to FP16 for FP32 networks
             -s                      : Show tensors and nodes in the tflite model
             -v                      : Verbose mode
             --show-exec-plan        : Show execution plan
             --show-supported-layers : Show supported layers by target
             --show-memory-summary   : Show memory allocation summary
             --relax-fp32            : Hint compiler to compute fp32 models using fp16
             --symmetric-int8        : Hint compiler to use symmetric 8-bit mode
             --l1-size-kb            : Hint compiler with the size of L1 memory
             --l2-size-kb            : Hint compiler with the size of L2 memory
                                       (--l1-size-kb and --l2-size-kb override env var.)
             --help                  : Show this usage

        Example:
            $ ./ncc-tflite -i input.bin -o output.bin -e <model.tflite> # For single input and output
            $ ./ncc-tflite -i input1.bin -i input2.bin -i input3.bin -o output1.bin \
                           -o output2.bin -e <model.tflite> # For multiple inputs and outputs

    6.6 neuronrt:
        Tool for executing static-time compiled networks (.dla file).
        neuronrt [-m<device>] -a<aotfile> [-d] [-k<id>] -i<inputbin> [-k<id>] -o<outputbin> [-c<num>]
         -m : Specify which device will be used to execute MDLA commands
              <device> can be: null/cmodel/hw, default is null
              'hw' is only available in NDK build or Android build.
              If 'cmodel' is chosen, users need to further set MTKNN_CMODEL_LIB_PATH to libCModel.so
              and set MTKNN_MDLA_EXECUTE_CMODEL=1.
         -a : Specify the compiled network
         -d : Show I/O id-shape mapping table.
         -k : Specify which tensor the following files will be used as input/output for.
              Note that -k affect all the following files, no matter it is specified with
              -i or -o. For example, -k 3 -i input3.bin -o output.bin causes output.bin
              be mapped to handle 3, which is usually not the user intended. To fix this,
              specify -k 3 -i input3.bin -k 0 -o output.bin
         -i : Specify an input bin file.
         -o : Specify an output bin file.
              Note that user can specify the sequence '-k<id> -[i|o] file' multiple times.
         -c : Repeat the inference <num> times. It can be used for profiling.

    6.7 toco:
        toco tool generated from TFLite source.

7. Libraries:
    7.1 libneuron.a:
        For users that would like to leverage neuron runtime framework, you can link this library
        in your application.

    7.2 libneuron_runtime.so:
        Necessary dynamic library executing neuronrt.

    7.3 libtensorflow_framework.so:
        Necessary dynamic library for toco.

    7.4 libneuron_platform.so:
        Library that contains platform related APIs.

====================================================================================================
