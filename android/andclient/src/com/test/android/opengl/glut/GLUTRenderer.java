/*
 * Copyright (C) 2008
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.test.android.opengl.glut;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.opengles.GL10;

import android.app.Activity;
import android.content.Context;
import android.opengl.GLU;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;

import com.test.android.opengl.glut.GL_View.EventListener;
import com.test.android.opengl.glut.GL_View.FpsPrinter;

@SuppressWarnings("unused")
public class GLUTRenderer implements GL_View.Renderer, EventListener, FpsPrinter {

    public static final int CUBE = 0;

    public static final int XYZ = 1;

    public static void updateSoapData() {

        canDraw = false;

        synchronized (Store.getXyzSoapData()) {
            final float[] xyzdata = Store.getXyzSoapData();
            GLUTDraw.XYZ.setXyz(xyzdata.length, xyzdata);
        }
        synchronized (Store.getBondsSoapData()) {
            final int[] bonddata = Store.getBondsSoapData();
            GLUTDraw.Bonds.setBonds(bonddata.length / 2, bonddata);
        }

        canDraw = true;
    }

    long lastDraw;

    protected Context mContext;

    int mode;

    private float mPreviousX = 0.0f;

    private float mPreviousY = 0.0f;

    private float mZoom = -40.0f;

    protected int rotateX;

    protected int rotateY;

    private final float TOUCH_SCALE_FACTOR = 0.25f;

    FpsPrinter printer;

    int[] vPort = new int[4];

    protected boolean wireFrame = false;

    protected int xChange = 10;

    protected int yChange = 10;

    protected static boolean canDraw = true;

    static int hackcount = 0;

    public static Thread recvT = new Thread(new SOAP_Activity.recvLoop());

    public GLUTRenderer(Activity context) {
        mContext = context;
        setFpsPrinter(this);
    }

    public void actionCenter() {
        rotateX = 0;
        rotateY = 0;
        wireFrame = !wireFrame;
    }

    public void actionDown() {
        rotateY -= yChange;
        if (rotateY < 0) {
            rotateY += 360;
        }
    }

    public void actionLeft() {
        rotateX -= xChange;
        if (rotateX < 0) {
            rotateX += 360;
        }
    }

    public void actionRight() {
        rotateX += xChange;
        if (rotateX > 360) {
            rotateX -= 360;
        }
    }

    public void actionUp() {
        rotateY += yChange;
        if (rotateY > 360) {
            rotateY -= 360;
        }
    }

    public void display(GL10 gl) {

        gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);

        //
        //  Set view matrix
        //
        gl.glMatrixMode(GL10.GL_MODELVIEW);
        gl.glLoadIdentity();
        gl.glTranslatef(0.0f, 0.0f, mZoom);
        GLU.gluLookAt(gl, 0.0f, 0.0f, 1.0f, /* position of eye */
                      0.0f, 0.0f, 0.0f, /* at, where pointing at */
                      0.0f, 1.0f, 0.0f /* up vector of the camera */
                     );
        gl.glRotatef(rotateY, 1.0f, 0.0f, 0.0f);
        gl.glRotatef(rotateX, 0.0f, 1.0f, 0.0f);

        //
        //  Draw
        //
        if (canDraw == true) {
            displayObjects(gl);
        }

    }

    public void displayObjects(GL10 gl) {
        if (mode == CUBE) {
            if (wireFrame) {
                GLUTDraw.glutWireCube(gl, 1.0f);
            } else {
                GLUTDraw.glutSolidCube(gl, 1.0f);
            }
        } else if (mode == XYZ) {
            if (wireFrame) {

            } else {
                GLUTDraw.glutXYZ(gl);
                GLUTDraw.glutXYZBonds(gl);
            }
        }
    }

    public void drawFrame(GL10 gl) {
        display(gl);

    }

    // XXX random xyz and bonds data
    public void generateFakeData(int natoms, int nbonds) {

        canDraw = false;

        final float[] xyzdata = makeRandomXyzData(natoms);
        GLUTDraw.XYZ.setXyz(natoms * 3, xyzdata);

        final int[] bonddata = makeRandomBonds(nbonds, natoms);
        GLUTDraw.Bonds.setBonds(nbonds, bonddata);

        canDraw = true;

    }

    // XXX random xyz and bonds data
    public void generateFakeSoapData(int natoms, int nbonds) {

        canDraw = false;

        final float[] xyzdata = Store.getXyzSoapData();
        natoms = xyzdata.length / 3;
        GLUTDraw.XYZ.setXyz(natoms * 3, xyzdata);

        final int[] bonddata = makeRandomBonds(nbonds, natoms);
        GLUTDraw.Bonds.setBonds(nbonds, bonddata);

        canDraw = true;

    }

    public int[] getConfigSpec() {
        final int[] configSpec = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_NONE }; // 24
        return configSpec;
    }

    public Context getContext() {
        return mContext;
    }

    public int getMode() {
        return mode;
    }

    public void init(GL10 gl) {
        /*  */
        gl.glClearColor(0.99f, 0.99f, 0.99f, 1.0f); // Set background
        gl.glShadeModel(GL10.GL_SMOOTH); // Smooth transitions between edges
        /* light properties */
        final float light_position1[] = { 1.0f, 1.0f, 1.0f, 0.0f }; // Define light source position
        final float ambient[] = { 0.5f, 0.5f, 0.5f, 1.0f }; // Define ambient lightning
        final float whiteDiffuse[] = { 0.85f, 0.85f, 0.85f, 1.0f }; // Define diffuse lighting
        gl.glLightModelfv(GL10.GL_LIGHT_MODEL_AMBIENT, GL_Utils.toFloatBufferPositionZero(ambient));
        gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_DIFFUSE, GL_Utils.toFloatBufferPositionZero(whiteDiffuse)); // Set light1 properties
        gl.glLightfv(GL10.GL_LIGHT0, GL10.GL_POSITION, GL_Utils.toFloatBufferPositionZero(light_position1)); // Set light1 properties
        /* enable */
        gl.glEnable(GL10.GL_COLOR_MATERIAL);// Enable color
        gl.glEnable(GL10.GL_LIGHTING); // Enable lighting for surfaces
        gl.glEnable(GL10.GL_LIGHT0); // Enable light source
        gl.glEnable(GL10.GL_RESCALE_NORMAL); // Generates the normals to the surfaces
        gl.glEnable(GL10.GL_NORMALIZE); // Keep lighting stable when scaling (can be slower)
        gl.glEnable(GL10.GL_CULL_FACE); // Enabling backface culling
        gl.glEnable(GL10.GL_DEPTH_TEST); // check the Z-buffer before placing pixels onto the screen.
        /* misc */
        gl.glDepthMask(true); // place depth values into the Z-buffer.
        gl.glDepthFunc(GL10.GL_LEQUAL); // valid depth check
        gl.glClearDepthf(1.0f); // 0 is near, 1 is far
        gl.glHint(GL10.GL_LINE_SMOOTH_HINT, GL10.GL_FASTEST); // Fastest rendering chosen
        gl.glHint(GL10.GL_POINT_SMOOTH_HINT, GL10.GL_FASTEST);
        gl.glHint(GL10.GL_POLYGON_SMOOTH_HINT, GL10.GL_FASTEST);
        gl.glHint(GL10.GL_PERSPECTIVE_CORRECTION_HINT, GL10.GL_FASTEST);
        /* size of bonds and spheres */
        gl.glLineWidth(2);
        gl.glPointSize(6);
    }

    // XXX random xyz data
    public int[] makeRandomBonds(int nbonds, int natoms) {
        final int N = nbonds * 2;
        final int[] nbdata = new int[N];
        int temp;
        for (int i = 0; i < N; ++i) {
            nbdata[i] = 0;
        }
        for (int i = 0; i < N; ++i) {
            temp = (int)(Math.random() * natoms);
            nbdata[i] = temp;
        }
        return nbdata;
    }

    // XXX random xyz data
    public float[] makeRandomXyzData(int natoms) {
        final int N = natoms * 3;
        final float low = -4.0f;
        final float high = 4.0f;
        final float[] xyz0 = new float[N];
        double temp;
        for (int i = 0; i < N - 2; i += 3) {
            temp = Math.sin(i * N / ((N) + 1.0)) * (high - low) + low;
            xyz0[i + 0] = (float) temp;
            temp = Math.cos(i * N / ((N) + 1.0)) * (high - low) + low;
            xyz0[i + 1] = (float) temp;
            temp = Math.tan(Math.random() / ((1.0) + 1.0)) * (high - low) + low;
            xyz0[i + 2] = (float) temp;
        }
        return xyz0;
    }

    public boolean onKeyDown(int keyCode, KeyEvent event) {

        if (keyCode == KeyEvent.KEYCODE_MENU) {
            wireFrame = !wireFrame;
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_UP) {
            //actionUp();
            //generateFakeData(GLUTDraw.XYZ.natoms + 100, GLUTDraw.Bonds.nbonds); // increase natoms
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_DOWN) {
            //actionDown();
            //generateFakeData(GLUTDraw.XYZ.natoms - 100, GLUTDraw.Bonds.nbonds); // decrease natoms
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_RIGHT) {
            //actionRight();
            //generateFakeData(GLUTDraw.XYZ.natoms, GLUTDraw.Bonds.nbonds + 10); // increase bonds
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_LEFT) {
            //actionLeft();
            //generateFakeData(GLUTDraw.XYZ.natoms, GLUTDraw.Bonds.nbonds - 10); // decrease bonds
        } else if (keyCode == KeyEvent.KEYCODE_DPAD_CENTER) {
            actionCenter();
        }

        return true;
    }

    public boolean onTouchEvent(MotionEvent e) {

        final float x1 = e.getX();
        final float y1 = e.getY();
        final boolean isMultitouch = (e.getPointerCount() > 1);

        if (isMultitouch) {
            final float x2 = e.getX(1);
            final float y2 = e.getY(1);
            final float dx = Math.abs(x1 - x2);
            final float dy = Math.abs(y1 - y2);
            mZoom = -(vPort[2] + vPort[3]) / (dx + dy) / TOUCH_SCALE_FACTOR;
        } else {
            switch (e.getAction()) {
            case MotionEvent.ACTION_MOVE:
                final float dx = x1 - mPreviousX;
                final float dy = y1 - mPreviousY;
                rotateX += dx * TOUCH_SCALE_FACTOR;
                rotateY += dy * TOUCH_SCALE_FACTOR;
            }
            mPreviousX = x1;
            mPreviousY = y1;
        }

        return true;
    }

    public void setFpsPrinter(FpsPrinter printer) {
        this.printer = printer;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public void showFps(String fps) {
        Log.d("GLUTRenderer", "fps " + fps);
    }

    public void sizeChanged(GL10 gl, int w, int h) {

        vPort[2] = w;
        vPort[3] = h;
        gl.glViewport(0, 0, w, h);
        gl.glMatrixMode(GL10.GL_PROJECTION);
        gl.glLoadIdentity();

        //
        //  set perspective
        //
        GLU.gluPerspective(gl, 65.0f, // deg FOV
                           (float) w / (float) h, // aspect ratio
                           0.1f, // z near
                           100.0f // z far
                          );

    }

    public void surfaceCreated(GL10 gl) {
        init(gl);

        /** FIRST DISPLAY */
        if (Store.isPollServer()) {
            Log.d("GLUTRenderer", "starting recvT");
            if (!GLUTRenderer.recvT.isAlive()) {
                GLUTRenderer.recvT.start();
            }
            Log.d("GLUTRenderer", " recvT started");
            //updateSoapData();
        } else {
            generateFakeData(3333, 69);
        }

    }

}
