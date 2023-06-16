adb wait-for-device

adb root
adb remount
::adb shell "setenforce 0"


:: FP32
::adb shell "/vendor/bin/hw/android.hardware.neuralnetworks@1.1-service-armnn >/dev/null 2>&1 &"
::adb shell setprop config.neuropilot.gpu_nn_fp_mode 32

:: FP16
::adb shell "/vendor/bin/hw/android.hardware.neuralnetworks@1.1-service-armnn -f >/dev/null 2>&1 &"
::adb shell setprop config.neuropilot.gpu_nn_fp_mode 16

::Restart GPU service
::adb shell "stop neuralnetworks_hal_service_armnn"
::adb shell "killall android.hardware.neuralnetworks@1.1-service-armnn"
::adb shell "start neuralnetworks_hal_service_armnn"

::Restart APU service
::adb shell "stop neuralnetworks_hal_service_apunn"
::adb shell "killall android.hardware.neuralnetworks@1.1-service-apunn"
::adb shell "start neuralnetworks_hal_service_apunn"


adb push MobilenetSSDDemo /data/local/tmp/MobilenetSSDDemo


:: Big core only
adb shell "echo 0 4 > /proc/ppm/policy/ut_fix_core_num"

:: Full speed
adb shell "echo 0 0 > /proc/ppm/policy/ut_fix_freq_idx"

:: GPU full speed
adb shell "echo 800000 > /proc/gpufreq/gpufreq_opp_freq"

:: DRAM full speed
adb shell "echo kir_emi 0 > /sys/devices/platform/10012000.dvfsrc_top/helio-dvfsrc/dvfsrc_debug"



:: Run Test
::adb shell "pidof android.hardware.neuralnetworks@1.1-service-apunn"
adb shell "cd /data/local/tmp;chmod +x MobilenetSSDDemo;./MobilenetSSDDemo"
:: Confirm APU service crash not happened
::adb shell "pidof android.hardware.neuralnetworks@1.1-service-apunn"

pause
