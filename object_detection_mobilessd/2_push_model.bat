:::::: First Push
::adb root
::adb disable-verity
::adb reboot
::adb wait-for-device
::adb wait-for-device
::adb root
::adb remount

:::::: PUSH ARMNN
::adb push nnservice/android.hardware.neuralnetworks@1.0-service-armnn /vendor/bin/hw/

:::::: PUSH libneuropilot.so
::adb push lib/32/libneuropilot.so /vendor/lib/
::adb push lib/64/libneuropilot.so /vendor/lib64/
::adb wait-for-device
::adb root
::adb remount


adb push model/mobilenet_coco.tflite /data/local/tmp/mobilenet_coco.tflite
adb push model/mobilenet_coco_quant.tflite /data/local/tmp/mobilenet_coco_quant.tflite

adb push model/mobilenet_ssd_pascal.tflite /data/local/tmp/mobilenet_ssd_pascal.tflite
adb push model/mobilenet_ssd_pascal_quant.tflite /data/local/tmp/mobilenet_ssd_pascal_quant.tflite
