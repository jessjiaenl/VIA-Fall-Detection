/*
 * Copyright (c) 2019, MediaTek Inc. All rights reserved.
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws.
 * The information contained herein is confidential and proprietary to
 * MediaTek Inc. and/or its licensors.
 * Except as otherwise provided in the applicable licensing terms with
 * MediaTek Inc. and/or its licensors, any reproduction, modification, use or
 * disclosure of MediaTek Software, and information contained herein, in whole
 * or in part, shall be strictly prohibited.
 */


package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.mediatek.neuropilot.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


/**
 * Detect images with Tensorflow Lite.
 */
public class PoseDetector {

    private static final String LOG_TAG = PoseDetector.class.getSimpleName();

    /**
     * Name of the model file stored in Assets.
     */
    private static final String MODEL_PATH = "openpose_mobilenetv0.75_quant_1x368x368x3.tflite";
    /**
     * Dimensions of batch size of the model.
     */
    private static final int DIM_BATCH_SIZE = 1;
    /**
     * Dimensions of inputs.
     */
    static final int DIM_IMG_SIZE_X = 368;
    static final int DIM_IMG_SIZE_Y = 368;
    private static final int DIM_PIXEL_SIZE = 3;
    /**
     * Dimensions of outputs from Tensorflow Lite.
     */
    public static final int DIM_OUTPUT_W = 46;
    public static final int DIM_OUTPUT_H = 46;
    static final int DIM_OUTPUT_NUM = 57;
    /**
     * Dimensions of data we generated from outputs. (will show on PoseView).
     */
    static final int FINAL_OUTPUT_DIM_NUM = 18;
    static final int FINAL_OUTPUT_DIM_POINT = 2;
    /**
     * Parameters for output normalization (INT8 -> FLOAT32).
     * Different for different model, can get this info from model.
     */
    private static final float OUTPUT_RATIO = 0.008926701731979847f;
    private static final int OUTPUT_BIAS = 126;
    /**
     * Threshold for output's confidence;
     * We will drop the node if its confidence is less than this threshold.
     */
    private static final float THRESHOLD_CONFIDENCE = 0.2f;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter interpreter;
    /**
     * Arrays to store pixel information of bitmap.
     */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    /**
     * ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs.
     */
    protected ByteBuffer imgData = null;
    /**
     * Arrays to hold outputs from Tensorflow Lite.
     */
    private byte[][][][] outputArrayByte = null;
    /**
     * Arrays to store data we generated from outputs.
     */
    public float[][] nodePosition = null;
    public float[][] positionsToDraw = null;


    /**
     * Constructor for PoseDetector. Will init necessary components.
     * @param activity
     * @throws IOException
     */
    PoseDetector(Activity activity) throws IOException {
        interpreter = new Interpreter(loadModelFile(activity, MODEL_PATH));
        interpreter.setUseNNAPI(true);

        imgData = ByteBuffer.allocateDirect(
                DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE * 1);
        imgData.order(ByteOrder.nativeOrder());

        outputArrayByte = new byte[DIM_BATCH_SIZE][DIM_OUTPUT_W][DIM_OUTPUT_H][DIM_OUTPUT_NUM];
        nodePosition = new float[FINAL_OUTPUT_DIM_NUM][FINAL_OUTPUT_DIM_POINT];
        for (int i = 0; i < FINAL_OUTPUT_DIM_NUM; i++) {
            for (int j = 0; j < FINAL_OUTPUT_DIM_POINT; j++) {
                nodePosition[i][j] = -1;
            }
        }
    }

    /**
     * Do pose detection for certain bitmap.
     * @param bitmap
     * @return The time consume of this detect.
     */
    String detectFrame(Bitmap bitmap) {
        if (interpreter == null) {
            Log.e(LOG_TAG, "Image detector has not been initialized; Skipped.");
            return "Uninitialized PoseDetector";
        }

        convertBitmapToByteBuffer(bitmap, intValues, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
        long startTime = SystemClock.uptimeMillis();
        interpreter.run(imgData, outputArrayByte);
        long endTime = SystemClock.uptimeMillis();


        



        float maxConfidence = 0F;
        for (int i = 0; i < FINAL_OUTPUT_DIM_NUM; i++) {
            maxConfidence = 0F;
            for (int j = 0; j < DIM_OUTPUT_H; j++) {
                for (int k = 0; k < DIM_OUTPUT_W; k++) {
                    float confidence = dequantizeConfidence(outputArrayByte[0][j][k][i]);
                    if (maxConfidence < confidence) {
                        maxConfidence = confidence;
                        nodePosition[i][0] = (float) k;
                        nodePosition[i][1] = (float) j;
                    }
                }
            }
            if (maxConfidence < THRESHOLD_CONFIDENCE) {      // remove node with low confidence.
                nodePosition[i][0] = -1;
                nodePosition[i][1] = -1;
            }
        }
        positionsToDraw = nodePosition.clone();

        return Long.toString(endTime - startTime) + "ms";
    }

    /**
     * Closes interpreter to release resources.
     */
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_PATH) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private void convertBitmapToByteBuffer(Bitmap bitmap, int[] intValues, int DIM_IMG_SIZE_X, int DIM_IMG_SIZE_Y) {
        if (imgData == null || bitmap == null) {
            return;
        }
        imgData.rewind();

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // fill in buffer with bgr format.
                imgData.put((byte) (val & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) ((val >> 16) & 0xFF));
            }
        }
     }

    /**
     * Change output's type from INT8 to FLOAT32.
     * @param data
     * @return
     */
    private float dequantizeConfidence(byte data) {
        int iclass = data;
        if (iclass < 0) {   // data in byte is unsigned int. So here need change int back to unsigned int.
            iclass = iclass + 256;
        }
        return (iclass - OUTPUT_BIAS) * OUTPUT_RATIO;
    }
}
