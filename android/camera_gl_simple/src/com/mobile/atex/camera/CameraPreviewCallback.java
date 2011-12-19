/*
   Copyright [2011] [Chris McClanahan]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
  Used the project MobileEye as a reference for
  getting the camera data and overlaying the camera bitmap.
*/

package com.mobile.atex.camera;

import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.hardware.Camera.Size;
import android.util.Log;


public class CameraPreviewCallback implements PreviewCallback {
    private static boolean mProcessing = true;
    private static boolean getnewframe = true;
    private final String TAG = "CameraPreviewCallback";
    private static Size mPreviewSize;
    private static int mBuffSize = 320 * 240 * 12 / 8;
    // buffsize =  w*h*12/8 : (12 bits per pixel / 8 bits per byte)
    public static byte[] cameradata = new byte[320 * 240 * 12 / 8];

    public CameraPreviewCallback(Size previewSize, int buffsize) {
        mPreviewSize = previewSize;
        mBuffSize = buffsize;
        cameradata = new byte[buffsize];
    }

    public void onPreviewFrame(byte[] data, Camera camera) {
        Log.d(TAG, "=======previewframe==========");
        if (getnewframe) {
            System.arraycopy(data, 0, cameradata, 0, data.length);
            getnewframe = false;
        }
    }

    public byte[] getCameraData() {
        getnewframe = true;
        return cameradata;
    }

    public void doProcessing() {
        mProcessing = !mProcessing;
    }

}
