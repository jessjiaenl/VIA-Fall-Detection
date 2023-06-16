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

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.util.AttributeSet;
import android.view.View;

import java.util.Arrays;

/**
 * This is for pose view.
 */
public class PoseView extends View {

    private static final String LOG_TAG = PoseView.class.getSimpleName();

    private static final float STRIKE_WIDTH_TEXT = 8f;
    private static final float STRIKE_WIDTH_LINE = 3f;
    private static final int TOTAL_POINTS = 18;
    private static final int COLOR_HEAD = Color.argb(180, 0, 0, 255);
    private static final int COLOR_BODY = Color.argb(180, 255, 255, 0);
    private static final int COLOR_FOOT = Color.argb(180, 0, 255, 0);
    private static final int COLOR_BACKGROUND = Color.WHITE;
    private static final int COLOR_LINE = Color.argb(180, 0x88, 0x88, 0x88);
    private static final int COLOR_TEXT = Color.RED;
    /**
     * Definition of output's index.
     */
    private static final int[] POINT_COLORS = {
            COLOR_HEAD, // 0. nose
            COLOR_BODY, // 1. neck
            COLOR_BODY, // 2. R shoulder
            COLOR_BODY, // 3. R elbow
            COLOR_BODY, // 4. R wrist
            COLOR_BODY, // 5. L shoulder
            COLOR_BODY, // 6. R elbow
            COLOR_BODY, // 7. R wrist
            COLOR_FOOT, // 8. R hip
            COLOR_FOOT, // 9. R knee
            COLOR_FOOT, // 10. R ankle
            COLOR_FOOT, // 11. L hip
            COLOR_FOOT, // 12. L knee
            COLOR_FOOT, // 13. L ankle
            COLOR_HEAD, // 14. R eye
            COLOR_HEAD, // 15. L eye
            COLOR_HEAD, // 16. R ear
            COLOR_HEAD, // 17. L ear
            COLOR_BACKGROUND, // 18. background
    };
    private static final int[][] LINE_POINTS = {
            {0, 1},
            {1, 2},
            {2, 3},
            {3, 4},
            {1, 5},
            {5, 6},
            {6, 7},
            {1, 8},
            {8, 9},
            {9, 10},
            {1, 11},
            {11, 12},
            {12, 13},
            {0, 14},
            {0, 15},
            {14, 16},
            {15, 17},
    };
    private static final String[] faceIndex = {"0", "14", "15","16","17"};

    private PointF[] points;
    private Paint paint;

    /**
     * init the face view.
     *
     * @param context the activity context.
     * @param attrs   view AttributeSet.
     */
    public PoseView(Context context, AttributeSet attrs) {
        super(context, attrs);
        paint = new Paint();
        paint.setTextSize(30f);
        points = new PointF[TOTAL_POINTS];
    }

    /**
     * Record latest data, which will be shown in the next call of onDraw() triggered by invalidate().
     * @param nodePosition
     */
    public void track(float[][] nodePosition) {
        for (int i = 0; i < TOTAL_POINTS; ++i) {
            if (i < nodePosition.length && nodePosition[i][0] >= 0) {
                // Resize output based on size of the view.
                points[i] = new PointF(
                        nodePosition[i][0] / PoseDetector.DIM_OUTPUT_W * getWidth(),
                        nodePosition[i][1] / PoseDetector.DIM_OUTPUT_H * getHeight());
            } else {
                points[i] = null;
            }
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        // Draw points.
        paint.setStrokeWidth(STRIKE_WIDTH_TEXT);
        for (int i = 0; i < TOTAL_POINTS; ++i) {
            PointF point = points[i];
            if (point != null) {
                paint.setColor(POINT_COLORS[i]);
                boolean isFacePoint = Arrays.asList(faceIndex).contains(String.valueOf(i));
                canvas.drawCircle(point.x, point.y, (isFacePoint ? 8 : 12), paint);
                paint.setColor(COLOR_TEXT);
                paint.setTextSize(isFacePoint ? 20f : 30f);
                canvas.drawText(String.valueOf(i), point.x, point.y, paint);
            }
        }

        // Draw lines.
        paint.setColor(COLOR_LINE);
        paint.setStrokeWidth(STRIKE_WIDTH_LINE);
        for (int[] linePoint : LINE_POINTS) {
            int startIndex = linePoint[0];
            int endIndex = linePoint[1];
            PointF startPoint = points[startIndex];
            PointF endPoint = points[endIndex];
            if (startPoint != null && endPoint != null) {
                canvas.drawLine(startPoint.x, startPoint.y, endPoint.x, endPoint.y, paint);
            }
        }
        super.onDraw(canvas);
    }
}
