adb wait-for-device
adb root
adb remount

adb push build/. /data/
adb push input /data/
adb push model /data/
adb push mt8168/lib /data/
adb push mt8168/bin/. /data/

adb shell "chmod +x /data/compiler"
adb shell "chmod +x /data/runtime"
adb shell "chmod +x /data/ncc-tflite"
adb shell "chmod +x /data/neuronrt"
