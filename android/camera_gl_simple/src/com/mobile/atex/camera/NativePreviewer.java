package com.mobile.atex.camera;

import java.io.IOException;
import java.util.Date;
import java.util.List;

import android.content.Context;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.Handler;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

//import com.opencv.camera.NativeProcessor.NativeProcessorCallback;
//import com.opencv.camera.NativeProcessor.PoolCallback;

public class NativePreviewer extends SurfaceView implements
    SurfaceHolder.Callback, Camera.PreviewCallback {
    SurfaceHolder mHolder;
    Camera mCamera;

    private int preview_width, preview_height;
    private int pixelformat;
    private PixelFormat pixelinfo;

    ////////////////////////////////////
    //private CameraErrorCallback mErrorCallback;
    private CameraPreviewCallback mPreviewCallback;
    //private CameraAutoFocusCallback mAutoFocusCallback;
    //private CameraPictureCallback mPictureCallback;
    ////////////////////////////////////

    public NativePreviewer(Context context, AttributeSet attributes) {
        super(context, attributes);
        //		listAllCameraMethods();
        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        this.preview_width = attributes.getAttributeIntValue("opencv", "preview_width", 600);
        this.preview_height = attributes.getAttributeIntValue("opencv", "preview_height", 600);

        setZOrderMediaOverlay(false);
    }
    public NativePreviewer(Context context, int preview_width,
                           int preview_height) {
        super(context);

        //		listAllCameraMethods();
        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        this.preview_width = preview_width;
        this.preview_height = preview_height;

        setZOrderMediaOverlay(false);

    }
    Handler camerainiter = new Handler();
    void initCamera(SurfaceHolder holder) throws InterruptedException {
        if (mCamera == null) {
            // The Surface has been created, acquire the camera and tell it where
            // to draw.
            int i = 0;
            while (i++ < 5) {
                try {
                    mCamera = Camera.open();
                    break;
                } catch (RuntimeException e) {
                    Thread.sleep(200);
                } catch (Exception e) {
                    Log.e("camera", "fail", e);
                }
            }
            try {
                mCamera.setPreviewDisplay(holder);
            } catch (IOException ex) {
                mCamera.release();
                mCamera = null;
                Log.e("camera", "stacktrace io", ex);
            } catch (RuntimeException e) {
                Log.e("camera", "stacktrace rt", e);
            } catch (Exception e) {
                Log.e("camera", "fail", e);
            }
        }
    }
    void releaseCamera() {
        if (mCamera != null) {
            // Surface will be destroyed when we return, so stop the preview.
            // Because the CameraDevice object is not a shared resource, it's very
            // important to release it when the activity is paused.
            try {
                mCamera.stopPreview();
                mCamera.release();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        mCamera = null;

    }

    public void surfaceCreated(SurfaceHolder holder) {

    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        releaseCamera();
    }

    public  byte[] getCameraData() {
        return mPreviewCallback.getCameraData();
    }

    private boolean hasAutoFocus = false;
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {

        try {
            initCamera(mHolder);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return;
        } catch (Exception e) {
            Log.e("camera", "fail", e);
            return;
        }

        // Now that the size is known, set up the camera parameters and begin
        // the preview.

        Camera.Parameters parameters = mCamera.getParameters();
        List<Camera.Size> pvsizes = mCamera.getParameters().getSupportedPreviewSizes();
        int best_width = 1000000;
        int best_height = 1000000;
        for (Size x: pvsizes) {
            if (x.width - preview_width >= 0 && x.width <= best_width) {
                best_width = x.width;
                best_height = x.height;
            }
        }
        preview_width = best_width;
        preview_height = best_height;
        List<String> fmodes = mCamera.getParameters().getSupportedFocusModes();

        int idx = fmodes.indexOf(Camera.Parameters.FOCUS_MODE_INFINITY);
        if (idx != -1) {
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_INFINITY);
        } else if (fmodes.indexOf(Camera.Parameters.FOCUS_MODE_FIXED) != -1) {
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_FIXED);
        }

        if (fmodes.indexOf(Camera.Parameters.FOCUS_MODE_AUTO) != -1) {
            hasAutoFocus  = true;
        }

        List<String> scenemodes = mCamera.getParameters().getSupportedSceneModes();
        if (scenemodes != null)
            if (scenemodes.indexOf(Camera.Parameters.SCENE_MODE_STEADYPHOTO) != -1) {
                parameters.setSceneMode(Camera.Parameters.SCENE_MODE_STEADYPHOTO);
            }

        parameters.setPreviewSize(preview_width, preview_height);
        mCamera.setParameters(parameters);

        pixelinfo = new PixelFormat();
        pixelformat = mCamera.getParameters().getPreviewFormat();
        PixelFormat.getPixelFormatInfo(pixelformat, pixelinfo);

        Size preview_size = mCamera.getParameters().getPreviewSize();
        preview_width = preview_size.width;
        preview_height = preview_size.height;

        int bufSize = preview_width * preview_height * pixelinfo.bitsPerPixel
                      / 8;

        Log.d("NATIVE", "w " + preview_width + " h " + preview_height + " bpp " + pixelinfo.bitsPerPixel);


        /////////////////////////////////////////////
        try {
            //mErrorCallback = new CameraErrorCallback();
            //mCamera.setErrorCallback(mErrorCallback);
            //mPictureCallback = new CameraPictureCallback();
            mPreviewCallback = new CameraPreviewCallback(preview_size, bufSize);
            mCamera.setPreviewCallback(mPreviewCallback);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return;
        }
        /////////////////////////////////////////////

        mCamera.startPreview();

        //postautofocus(0);
    }
    public void postautofocus(int delay) {
        if (hasAutoFocus) {
            handler.postDelayed(autofocusrunner, delay);
        }

    }
    private Runnable autofocusrunner = new Runnable() {

        @Override
        public void run() {
            mCamera.autoFocus(autocallback);

        }
    };

    Camera.AutoFocusCallback autocallback = new Camera.AutoFocusCallback() {

        @Override
        public void onAutoFocus(boolean success, Camera camera) {
            if (!success) {
                postautofocus(1000);
            }
        }
    };
    Handler handler = new Handler();

    /* /\** */
    /*  * This method will list all methods of the android.hardware.Camera class, */
    /*  * even the hidden ones. With the information it provides, you can use the */
    /*  * same approach I took below to expose methods that were written but hidden */
    /*  * in eclair */
    /*  *\/ */
    /* private void listAllCameraMethods() { */
    /* 	try { */
    /* 		Class<?> c = Class.forName("android.hardware.Camera"); */
    /* 		Method[] m = c.getMethods(); */
    /* 		for (int i = 0; i < m.length; i++) { */
    /* 			Log.d("NativePreviewer", "  method:" + m[i].toString()); */
    /* 		} */
    /* 	} catch (Exception e) { */
    /* 		// TODO Auto-generated catch block */
    /* 		Log.e("NativePreviewer", e.toString()); */
    /* 	} */
    /* } */



    Date start;
    int fcount = 0;
    boolean processing = false;
    public static byte[] rawdata = null;

    /**
     * Demonstration of how to use onPreviewFrame. In this case I'm not
     * processing the data, I'm just adding the buffer back to the buffer queue
     * for re-use
     */
    public void onPreviewFrame(byte[] data, Camera camera) {
        if (start == null) {
            start = new Date();
        }
        //		processor.post(data, preview_width, preview_height,
        //		pixelformat, System.nanoTime(),	this);
        rawdata = data;
        fcount++;
        if (fcount % 100 == 0) {
            double ms = (new Date()).getTime() - start.getTime();
            Log.i("NativePreviewer", "fps:" + fcount / (ms / 1000.0));
            start = new Date();
            fcount = 0;
        }
    }

    /* @Override */
    /* public void onDoneNativeProcessing(byte[] buffer) { */
    /* 	addCallbackBuffer(buffer); */
    /* } */

    /* public void addCallbackStack(LinkedList<PoolCallback> callbackstack) { */
    /* 	processor.addCallbackStack(callbackstack); */
    /* } */

    /**This must be called when the activity pauses, in Activity.onPause
     * This has the side effect of clearing the callback stack.
     *
     */
    public void onPause() {
        releaseCamera();
        //		addCallbackStack(null);
        //		processor.stop();
    }

    public void onResume() {
        //		processor.start();
    }

}
