/*
 * Copyright (C) 2007 The Android Open Source Project Licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the License. You may obtain
 * a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable
 * law or agreed to in writing, software distributed under the License is distributed on an "AS IS"
 * BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
 * for the specific language governing permissions and limitations under the License.
 *
 * ~ Modified by Chris McClanahan for GTRI ~
 */

package com.fptd.sensorlogger;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.hardware.SensorListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.view.ContextMenu;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ContextMenu.ContextMenuInfo;

/**
 * Application that displays the values of the acceleration, magnetic field, and orientation sensors
 * graphically.
 */
@SuppressWarnings("deprecation")
public class Sensors extends Activity {
    public class GraphView extends View implements SensorListener { // SensorEventListener {
        private Bitmap mBitmap;
        private final Paint mPaint = new Paint();
        private final Canvas mCanvas = new Canvas();
        private final Path mPath = new Path();
        private final RectF mRect = new RectF();
        private final float mLastValues[] = new float[3 * 2];
        private final float mZeroValues[] = new float[3 * 2];
        private final float mOrientationValues[] = new float[3];
        private final int mColors[] = new int[3 * 2];
        private float mLastX;
        private final float mScale[] = new float[2];
        private float mYOffset;
        private float mMaxX;
        private final float mDeltaX = 1.0f;
        private float mWidth;
        private boolean mInit = false;
        private final float mStrokeWidth = 2.0f;
        private final float mAccThreshGui[] = new float[6];

        public GraphView(final Context context) {
            super(context);
            mColors[0] = Color.argb(220, 255, 000, 000); // red
            mColors[1] = Color.argb(220, 000, 255, 000); // green
            mColors[2] = Color.argb(220, 000, 000, 255); // blue
            mColors[3] = Color.argb(220, 000, 255, 255); // cyan
            mColors[4] = Color.argb(220, 255, 000, 255); // magenta
            mColors[5] = Color.argb(220, 255, 255, 000); // yellow

            mPaint.setFlags(Paint.ANTI_ALIAS_FLAG);
            mPaint.setStrokeWidth(mStrokeWidth);
            mRect.set(-0.5f, -0.5f, 0.5f, 0.5f);
            mPath.arcTo(mRect, 0, 180);
        }

        public void onAccuracyChanged(final int sensor, final int accuracy) {
            // TODO Auto-generated method stub

        }

        @Override
        protected void onDraw(final Canvas canvas) {
            synchronized (this) {
                if (mBitmap != null) {

                    final Paint paint = mPaint;
                    final Path path = mPath;
                    final int outer = 0xFFC0C0C0; // grey
                    final int inner = 0xFFFF7711; // orange
                    final int otext = 0xFFCC22AA; // purple

                    // restart graph when edge is reached
                    if (mLastX >= mMaxX) {
                        mLastX = 0;
                        final Canvas cavas = mCanvas;
                        final float yoffset = mYOffset;
                        final float maxx = mMaxX;
                        // final float oneG = SensorManager.STANDARD_GRAVITY * mScale[0];
                        // final float halfheight = mHeight/2; // mYOffset is already half
                        cavas.drawColor(0xFFFFFFFF);
                        // center gray line
                        paint.setColor(outer);
                        paint.setStrokeWidth(mStrokeWidth + 1.0f);
                        cavas.drawLine(0, yoffset, maxx, yoffset, paint);
                        paint.setStrokeWidth(mStrokeWidth);
                        // thresholds from seek bars
                        if (Store.isThresholding()) {
                            float xthresh1 = mAccThreshGui[0];
                            float ythresh1 = mAccThreshGui[1];
                            float zthresh1 = mAccThreshGui[2];
                            float xthresh2 = mAccThreshGui[3];
                            float ythresh2 = mAccThreshGui[4];
                            float zthresh2 = mAccThreshGui[5];
                            paint.setColor(mColors[0]);
                            cavas.drawLine(0, xthresh1, maxx, xthresh1, paint);
                            cavas.drawLine(0, xthresh2, maxx, xthresh2, paint);
                            paint.setColor(mColors[1]);
                            cavas.drawLine(0, ythresh1, maxx, ythresh1, paint);
                            cavas.drawLine(0, ythresh2, maxx, ythresh2, paint);
                            paint.setColor(mColors[2]);
                            cavas.drawLine(0, zthresh1, maxx, zthresh1, paint);
                            cavas.drawLine(0, zthresh2, maxx, zthresh2, paint);
                        }
                    }

                    //
                    canvas.drawBitmap(mBitmap, 0, 0, null);

                    // orientation circles
                    float[] values = mOrientationValues;
                    float w0 = mWidth * 0.333333f;
                    float w = w0 - 32;
                    float x = w0 * 0.5f;
                    float y = w * 0.5f + 4.0f;
                    for (int i = 0; i < 3; ++i) {
                        // each circle
                        canvas.save(Canvas.MATRIX_SAVE_FLAG);
                        canvas.translate(x, y);
                        canvas.save(Canvas.MATRIX_SAVE_FLAG);
                        // outer circle
                        paint.setColor(outer);
                        canvas.scale(w, w);
                        canvas.drawOval(mRect, paint);
                        canvas.restore();
                        // inner half circle
                        paint.setColor(inner);
                        canvas.scale(w - 6, w - 6);
                        canvas.rotate(-values[i]);
                        canvas.drawPath(path, paint);
                        canvas.restore();
                        // text value
                        String t = ((Float)(values[i])).toString();
                        paint.setColor(otext);
                        canvas.drawText(t, x, y, paint);
                        // next circle
                        x += w0;
                    }

                } // end if (mBitmap != null)

            } // end synchronized

        }

        public void onSensorChanged(final int sensor, final float[] values) {
            // Log.d(TAG, "sensor: " + sensor + ", x: " + values[0] + ", y: " + values[1] + ", z: "
            // + values[2]);
            synchronized (this) {
                if (mBitmap != null) {

                    final Canvas canvas = mCanvas;
                    final Paint paint = mPaint;
                    float newX = mLastX + mDeltaX;
                    int j; // 0 -> Accelerometer , 1 -> Magnetic Field

                    switch (sensor) {

                    case SensorManager.SENSOR_ACCELEROMETER:
                        j = 0;
                        for (int i = 0; i < 3; ++i) {
                            // plot
                            int k = i + j * 3;
                            final float v = mYOffset + values[i] * mScale[j] - mZeroValues[k];
                            paint.setColor(mColors[k]);
                            canvas.drawLine(mLastX, mLastValues[k], newX, v, paint);
                            mLastValues[k] = v;
                            // threshold notification
                            if (Store.isThresholding()) {
                                if (Math.abs(v) > mAccThreshGui[k]) {
                                    paint.setColor(0xFFFF7711); // orange
                                    canvas.drawCircle(newX, v, 4.0f, paint);
                                } else if (Math.abs(v) < mAccThreshGui[k + 3]) {
                                    paint.setColor(0xFFFF7711); // orange
                                    canvas.drawCircle(newX, v, 4.0f, paint);
                                }
                            }
                        }
                        // mLastX = newX; // acceleration sensor updates fastest
                        break;

                    case SensorManager.SENSOR_MAGNETIC_FIELD:
                        j = 1;
                        for (int i = 0; i < 3; ++i) {
                            // plot
                            int k = i + j * 3;
                            final float v = mYOffset + values[i] * mScale[j] - mZeroValues[k];
                            paint.setColor(mColors[k]);
                            canvas.drawLine(mLastX, mLastValues[k], newX, v, paint);
                            mLastValues[k] = v;
                        }
                        mLastX = newX; // magnetic field sensor updates slowest
                        break;

                    case SensorManager.SENSOR_ORIENTATION:
                        // drawn above in orientation circles section
                        for (int i = 0; i < 3; ++i) {
                            mOrientationValues[i] = values[i];
                        }
                        break;

                    }

                    // refresh screen
                    invalidate();
                }
            }
        }

        @Override
        protected void onSizeChanged(final int w, final int h, final int oldw, final int oldh) {
            if (mInit == false) {
                // see AndroidManifest.xml - forced to portrait mode
                if (w < h) {
                    // vertical orientation setup
                    mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);
                    mCanvas.setBitmap(mBitmap);
                    mCanvas.drawColor(0xFFFFFFFF);
                    mYOffset = h * 0.5f;
                    mScale[0] = -(h * 0.5f * (1.0f / (SensorManager.STANDARD_GRAVITY * Store.getNumGs()))); // accelerometer
                    mScale[1] = -(h * 0.5f * (1.0f / (SensorManager.MAGNETIC_FIELD_EARTH_MAX))); // magnetic
                    // field
                    mWidth = w;
                    mMaxX = w;
                    mInit = true; // stay vertical
                    mLastX = mMaxX; // set to max to trigger check in onDraw()

                    if (Store.isThresholding()) {
                        mAccThreshGui[0] = mYOffset + mYOffset * Store.getXthreshAcc() / 100;
                        mAccThreshGui[1] = mYOffset + mYOffset * Store.getYthreshAcc() / 100;
                        mAccThreshGui[2] = mYOffset + mYOffset * Store.getZthreshAcc() / 100;
                        mAccThreshGui[3] = mYOffset - mYOffset * Store.getXthreshAcc() / 100;
                        mAccThreshGui[4] = mYOffset - mYOffset * Store.getYthreshAcc() / 100;
                        mAccThreshGui[5] = mYOffset - mYOffset * Store.getZthreshAcc() / 100;
                    }
                }
                super.onSizeChanged(w, h, oldw, oldh);
            }
        }

        public void setReferenceZero() {
            // XXX
            for (int i = 0; i < mZeroValues.length; ++i) { mZeroValues[i] = mLastValues[i] - mYOffset; }
        }

        public void resetToZero() {
            for (int i = 0; i < mZeroValues.length; ++i) { mZeroValues[i] = 0; }
        }

    } // end class

    private SensorManager mSensorManager;

    private GraphView mGraphView;

    /**
     * Initialization of the Activity after it is first created. Must at least call
     * {@link android.app.Activity#setContentView setContentView()} to describe what is to be
     * displayed in the screen.
     */
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        // Be sure to call the super class.
        super.onCreate(savedInstanceState);

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mGraphView = new GraphView(this);
        mGraphView.resetToZero();
        setContentView(mGraphView);
    }

    @Override
    protected void onPause() {
        mSensorManager.unregisterListener(mGraphView);
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(mGraphView, SensorManager.SENSOR_ACCELEROMETER | SensorManager.SENSOR_MAGNETIC_FIELD | SensorManager.SENSOR_ORIENTATION, SensorManager.SENSOR_DELAY_FASTEST);
        // mSensorManager.registerListener(mGraphView,
        // mSensorManager.getDefaultSensor(SensorManager.SENSOR_ACCELEROMETER),
        // SensorManager.SENSOR_DELAY_FASTEST);
        // mSensorManager.registerListener(mGraphView,
        // mSensorManager.getDefaultSensor(SensorManager.SENSOR_MAGNETIC_FIELD),
        // SensorManager.SENSOR_DELAY_FASTEST);
        // mSensorManager.registerListener(mGraphView,
        // mSensorManager.getDefaultSensor(SensorManager.SENSOR_ORIENTATION),
        // SensorManager.SENSOR_DELAY_FASTEST);

    }

    @Override
    protected void onStop() {
        mSensorManager.unregisterListener(mGraphView);
        super.onStop();
    }


    private static final int MENU_ZERO_SET = 0;
    private static final int MENU_RESET_ZERO = 1;

    /**
     * */
    @Override
    public boolean onCreateOptionsMenu(final Menu menu) {
        menu.add(0, MENU_ZERO_SET, MENU_ZERO_SET, "Set Zero");
        menu.add(0, MENU_RESET_ZERO, MENU_RESET_ZERO, "Reset Zero");
        return super.onCreateOptionsMenu(menu);
    }

    /**
     * */
    @Override
    public boolean onOptionsItemSelected(final MenuItem item) {
        switch (item.getItemId()) {
        case MENU_ZERO_SET:
            mGraphView.setReferenceZero();
            break;
        case MENU_RESET_ZERO:
            mGraphView.resetToZero();
            break;
        }
        return super.onOptionsItemSelected(item);
    }



}
