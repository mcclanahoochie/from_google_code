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

import javax.microedition.khronos.opengles.GL;

import android.app.Activity;
import android.opengl.GLDebugHelper;
import android.os.Bundle;

public class CubeActivity extends Activity {

    /**
     * Set to true to enable checking of the OpenGL error code after every OpenGL call.
     * Set to false for faster code.
     */
    private final static boolean DEBUG_CHECK_GL_ERROR = false;

    private GL_View mGLView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTitle("Cube");

        mGLView = new GL_View(this);
        setContentView(mGLView);

        if (DEBUG_CHECK_GL_ERROR) {
            mGLView.setGLWrapper(new GL_View.GLWrapper() {
                public GL wrap(GL gl) {
                    return GLDebugHelper.wrap(gl, GLDebugHelper.CONFIG_CHECK_GL_ERROR, null);
                }
            });
        }
        final GLUTRenderer render = new GLUTRenderer(this);
        render.setMode(GLUTRenderer.CUBE);
        mGLView.setRenderer(render);
        mGLView.setEventListener(render);
        mGLView.requestFocus();
    }

    @Override
    protected void onPause() {
        super.onPause();
        mGLView.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mGLView.onResume();
    }
}
