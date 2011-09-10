package com.mobile.atex;

import com.mobile.atex.camera.NativePreviewer;

import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.view.ViewGroup.LayoutParams;
import android.view.Gravity;


import android.app.Activity;
import android.opengl.GLSurfaceView;
import android.os.Bundle;


/**
 * The initial Android Activity, setting and initiating
 * the OpenGL ES Renderer Class @see RenderGL.java
 *
 * @author
 */
public class Run extends Activity  {

    /** The OpenGL View */
    private GLSurfaceView glSurface;

    /** The Camera */
    private NativePreviewer mPreview;


    /**
     * Initiate the OpenGL View and set our own
     * Renderer (@see RenderGL.java)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        glSurface = new GLSurfaceView(this);
        final RenderGL render = new RenderGL(this);
        glSurface.setRenderer(render);
        glSurface.setOnTouchListener(render);


        //////////////////////////////////
        int camWidth = 320;
        int camHeight = 240;
        //////////////////////////////////
        FrameLayout frame = new FrameLayout(this);
        mPreview = new NativePreviewer(getApplication(), camWidth, camHeight);
        LayoutParams params = new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT);
        params.height = getWindowManager().getDefaultDisplay().getHeight();
        params.width = (int)(params.height * 4.0 / 2.88);
        LinearLayout vidlay = new LinearLayout(getApplication());
        vidlay.setGravity(Gravity.CENTER);
        vidlay.addView(mPreview, params);
        frame.addView(vidlay);
        mPreview.setZOrderMediaOverlay(false);
        ///////////////////////////////////
        glSurface.setZOrderMediaOverlay(true);
        glSurface.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT, LayoutParams.FILL_PARENT));
        frame.addView(glSurface);
        setContentView(frame);
        ///////////////////////////////////



        //setContentView(glSurface);
    }

    /**
     * Remember to resume the glSurface
     */
    @Override
    protected void onResume() {
        super.onResume();
        glSurface.onResume();
    }

    /**
     * Also pause the glSurface
     */
    @Override
    protected void onPause() {
        super.onPause();
        glSurface.onPause();
    }


}