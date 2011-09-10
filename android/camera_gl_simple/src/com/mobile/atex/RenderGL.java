package com.mobile.atex;


import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import com.mobile.atex.camera.NativePreviewer;

import android.content.Context;
import android.opengl.GLU;
import android.opengl.GLSurfaceView.Renderer;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;

/**
 *
 */
public class RenderGL implements Renderer, OnTouchListener  {

    /** Cube instance */
    private Cube cube;

    /* Rotation values for all axis */
    private float xrot;				//X Rotation
    private float yrot;				//Y Rotation
    private float zrot;				//Z Rotation

    /** The Activity Context ( NEW ) */
    private Context context;

    /**
     * Instance the Cube object and set
     * the Activity Context handed over
     */
    public RenderGL(Context context) {
        this.context = context;

        cube = new Cube();
    }

    /**
     * Instance the Cube object and set
     * the Activity Context handed over
     * AND CAMERA
     */
    private NativePreviewer mPreview = null;
    public RenderGL(Context context, NativePreviewer camera) {
        this.context = context;
        this.mPreview = camera;
        cube = new Cube();
    }

    /**
     * The Surface is created/init()
     */
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        // jni_init(...)

        //Load the texture for the cube once during Surface creation
        cube.loadGLTexture(gl, this.context);

        gl.glEnable(GL10.GL_TEXTURE_2D);			//Enable Texture Mapping ( NEW )
        gl.glShadeModel(GL10.GL_SMOOTH); 			//Enable Smooth Shading
        gl.glClearColor(0.0f, 0.0f, 0.0f, 0.5f); 	//Black Background
        gl.glClearDepthf(1.0f); 					//Depth Buffer Setup
        gl.glEnable(GL10.GL_DEPTH_TEST); 			//Enables Depth Testing
        gl.glDepthFunc(GL10.GL_LEQUAL); 			//The Type Of Depth Testing To Do

        //Really Nice Perspective Calculations
        gl.glHint(GL10.GL_PERSPECTIVE_CORRECTION_HINT, GL10.GL_NICEST);
    }

    /**
     * Here we do our drawing
     */
    public void onDrawFrame(GL10 gl) {

        //Clear Screen And Depth Buffer
        gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);
        gl.glLoadIdentity();			//Reset The Current Modelview Matrix

        //Drawing
        gl.glTranslatef(0.0f, 0.0f, mZoom);		//Pinch to zoom
        gl.glScalef(0.8f, 0.8f, 0.8f); 			//Scale the Cube to 80 percent

        //Rotate around the axis based on the rotation matrix (rotation, x, y, z)
        gl.glRotatef(xrot, 1.0f, 0.0f, 0.0f);	//X  - touch spin
        gl.glRotatef(yrot, 0.0f, 1.0f, 0.0f);	//Y  - touch spin
        gl.glRotatef(zrot, 0.0f, 0.0f, 1.0f);	//Z

        if (mPreview == null) { cube.draw(gl); }							//Draw the Cube
        else { cube.draw(gl, mPreview.getCameraData()); }				//Draw the Cube

        //Change rotation factors (nice rotation)
        xrot += 0.3f;
        yrot += 0.2f;
        zrot += 0.4f;

        /* RENDER */
        // jni_render(...)
        Native.renderNative(drawWidth, drawHeight, true, ((mPreview == null) ? null : mPreview.getCameraData()));

    }

    /**
     * If the surface changes, reset the view
     */
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        // jni_surface_changed(...)

        if (height == 0) { 						//Prevent A Divide By Zero By
            height = 1; 						//Making Height Equal One
        }

        gl.glViewport(0, 0, width, height); 	//Reset The Current Viewport
        gl.glMatrixMode(GL10.GL_PROJECTION); 	//Select The Projection Matrix
        gl.glLoadIdentity(); 					//Reset The Projection Matrix

        //Calculate The Aspect Ratio Of The Window
        GLU.gluPerspective(gl, 45.0f, (float)width / (float)height, 0.1f, 100.0f);

        gl.glMatrixMode(GL10.GL_MODELVIEW); 	//Select The Modelview Matrix
        gl.glLoadIdentity(); 					//Reset The Modelview Matrix

        /* save for render() */
        drawWidth = width;
        drawHeight = height;
        Native.surfaceChangedNative(width, height);

    }


    /**
     * ===========================================
     */
    private float mPreviousX = 0.0f;
    private float mPreviousY = 0.0f;
    private final float TOUCH_SCALE_FACTOR = 0.25f;
    private float mZoom = -10.0f;
    int drawWidth = 320;
    int drawHeight = 240;

    @Override
    public boolean onTouch(View v, MotionEvent e) {
        final float x1 = e.getX();
        final float y1 = e.getY();
        final boolean isMultitouch = (e.getPointerCount() > 1);
        if (isMultitouch) {
            // jni_multitouch(...)
            final float x2 = e.getX(1);
            final float y2 = e.getY(1);
            final float dx = Math.abs(x1 - x2);
            final float dy = Math.abs(y1 - y2);
            mZoom = -(drawWidth + drawHeight) / (dx + dy) / TOUCH_SCALE_FACTOR;
        } else {
            switch (e.getAction()) {
            case MotionEvent.ACTION_MOVE:
                final float dx = x1 - mPreviousX;
                final float dy = y1 - mPreviousY;
                xrot += dx * TOUCH_SCALE_FACTOR;
                yrot += dy * TOUCH_SCALE_FACTOR;
                // jni_on_move
            }
            mPreviousX = x1;
            mPreviousY = y1;
        }
        return true;
    }
    /**
     * ===========================================
     */

}
