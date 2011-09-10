/*
 * This class is based glutes_shape.c
 *
 *      http://glutes.sourceforge.net/
 */

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

import java.nio.FloatBuffer;

import javax.microedition.khronos.opengles.GL10;

public class GLUTDraw {
    public static class Bonds {
        public static int nbonds;

        public static FloatBuffer bPosvBuffer = null;

        public static void draw(GL10 gl) {

            if (bPosvBuffer == null) { return; }
            final int A = nbonds;

            //
            // Draw bonds
            //
            gl.glVertexPointer(3, GL10.GL_FLOAT, 0, bPosvBuffer);
            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            for (int atom = 0; atom < A; atom++) {
                gl.glDrawArrays(GL10.GL_LINES, atom * 2, 2);
            }
            gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
        }

        private static FloatBuffer loadBondsPosv(int nbonds, int[] nblist) {
            if (bPosvBuffer != null) {
                bPosvBuffer.clear();
            }

            int _N = nbonds * 2 * 3; // size of nblist xyz coordinate pair buffer
            bPosvBuffer = GL_Utils.allocateFloatBuffer(_N * 4);

            final int DIM_NUM = 3;
            //int N = XYZ.natoms * DIM_NUM - 2;
            final int A = nbonds * 2 - 1;
            _N = XYZ.xyzvBuffer.capacity() - 2;

            int idx1, idx2;
            float x1, y1, z1, x2, y2, z2;
            for (int atom = 0; atom < A; atom += 2) {

                idx1 = nblist[atom] * DIM_NUM;
                idx2 = nblist[atom + 1] * DIM_NUM;
                //Log.i("GLUTD ", "atom "+atom+" = "+idx1+","+idx2);
                if ((idx1 < _N) && (idx1 >= 0) && (idx2 < _N) && (idx2 >= 0)) { // ??
                    x1 = XYZ.xyzvBuffer.get(idx1); //xyz[idx1  ];
                    y1 = XYZ.xyzvBuffer.get(idx1 + 1); //xyz[idx1+1];
                    z1 = XYZ.xyzvBuffer.get(idx1 + 2); //xyz[idx1+2];
                    x2 = XYZ.xyzvBuffer.get(idx2); //xyz[idx2  ];
                    y2 = XYZ.xyzvBuffer.get(idx2 + 1); //xyz[idx2+1];
                    z2 = XYZ.xyzvBuffer.get(idx2 + 2); //xyz[idx2+2];

                    bPosvBuffer.put(x1);
                    bPosvBuffer.put(y1);
                    bPosvBuffer.put(z1);
                    bPosvBuffer.put(x2);
                    bPosvBuffer.put(y2);
                    bPosvBuffer.put(z2);
                }
            }
            bPosvBuffer.position(0);
            return bPosvBuffer;
        }

        public static void setBonds(int _nbonds, int[] nblist) {
            nbonds = _nbonds;
            bPosvBuffer = loadBondsPosv(nbonds, nblist);
        }
    }

    public static class SolidCube {
        static float cuben[] = { 0, 0, 1f, /* front */
                                 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, -1f, /* back */
                                 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, -1f, 0, 0, /* left */
                                 -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, 1f, 0, 0, /* right */
                                 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 0, 1f, 0, /* top */
                                 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, 1f, 0, 0, -1f, 0, /* bottom */
                                 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0, 0, -1f, 0,
                               };

        private static FloatBuffer cubenBuffer;

        public static float cubev[] = { -1f, -1f, 1f, /* front */
                                        1f, -1f, 1f, -1f, 1f, 1f, 1f, -1f, 1f, 1f, 1f, 1f, -1f, 1f, 1f, -1f, 1f, -1f, /* back */
                                        1f, -1f, -1f, -1f, -1f, -1f, -1f, 1f, -1f, 1f, 1f, -1f, 1f, -1f, -1f, -1f, -1f, -1f, /* left */
                                        -1f, -1f, 1f, -1f, 1f, -1f, -1f, -1f, 1f, -1f, 1f, 1f, -1f, 1f, -1f, 1f, -1f, 1f, /* right */
                                        1f, -1f, -1f, 1f, 1f, 1f, 1f, -1f, -1f, 1f, 1f, -1f, 1f, 1f, 1f, -1f, 1f, 1f, /* top */
                                        1f, 1f, 1f, -1f, 1f, -1f, 1f, 1f, 1f, 1f, 1f, -1f, -1f, 1f, -1f, -1f, -1f, -1f, /* bottom */
                                        1f, -1f, -1f, -1f, -1f, 1f, 1f, -1f, -1f, 1f, -1f, 1f, -1f, -1f, 1f,
                                      };

        private static FloatBuffer cubevBuffer;

        private static float param;

        public static float v[] = new float[108]; // 108 = 6*18

        public static void draw(GL10 gl, float size) {
            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glEnableClientState(GL10.GL_NORMAL_ARRAY);
            if (cubevBuffer != null) {
                if (param != size) {
                    cubevBuffer = null;
                    cubenBuffer = null;
                    gl.glVertexPointer(3, GL10.GL_FLOAT, 0, GL_Utils.allocateFloatBuffer(0));
                    gl.glNormalPointer(GL10.GL_FLOAT, 0, GL_Utils.allocateFloatBuffer(0));
                }
            }
            if (cubenBuffer == null) {
                cubevBuffer = loadCubev(size);
                cubenBuffer = loadCuben();
                param = size;
            }
            gl.glVertexPointer(3, GL10.GL_FLOAT, 0, cubevBuffer);
            gl.glNormalPointer(GL10.GL_FLOAT, 0, cubenBuffer);
            gl.glDrawArrays(GL10.GL_TRIANGLES, 0, 36);
            gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glDisableClientState(GL10.GL_NORMAL_ARRAY);
        }

        private static FloatBuffer loadCuben() {
            cubenBuffer = GL_Utils.allocateFloatBuffer(108 * 4);
            for (int i = 0; i < 108; i++) {
                cubenBuffer.put(cuben[i]);
            }
            cubenBuffer.position(0);
            return cubenBuffer;
        }

        private static FloatBuffer loadCubev(float size) {
            size /= 2;
            cubevBuffer = GL_Utils.allocateFloatBuffer(108 * 4);
            for (int i = 0; i < 108; i++) {
                cubevBuffer.put(cubev[i] * size);
            }
            cubevBuffer.position(0);
            return cubevBuffer;
        }
    }

    public static class WireCube {
        static float cuben[] = { 0f, 0f, 1.0f, /* front */
                                 0f, 0f, 1.0f, 0f, 0f, 1.0f, 0f, 0f, 1.0f, 0f, 0f, -1.0f, /* back */
                                 0f, 0f, -1.0f, 0f, 0f, -1.0f, 0f, 0f, -1.0f, -1.0f, 0f, 0f, /* left */
                                 -1.0f, 0f, 0f, -1.0f, 0f, 0f, -1.0f, 0f, 0f, 1.0f, 0f, 0f, /* right */
                                 1.0f, 0f, 0f, 1.0f, 0f, 0f, 1.0f, 0f, 0f, 0f, 1.0f, 0f, /* top */
                                 0f, 1.0f, 0f, 0f, 1.0f, 0f, 0f, 1.0f, 0f, 0f, -1.0f, 0f, /* bottom */
                                 0f, -1.0f, 0f, 0f, -1.0f, 0f, 0f, -1.0f, 0f,
                               };

        private static FloatBuffer cubenBuffer;

        static float cubev[] = { // 72 = 3*6*4
            -1.0f, -1.0f, 1.0f, /* front */
            1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, /* back */
            1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, /* left */
            -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, /* right */
            1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, /* top */
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, /* bottom */
            1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
        };

        private static FloatBuffer cubevBuffer;

        private static float param;

        static float v[] = new float[72];

        public static void draw(GL10 gl, float size) {
            if (cubevBuffer != null) {
                if (param != size) {
                    cubevBuffer = null;
                    cubenBuffer = null;
                    gl.glVertexPointer(3, GL10.GL_FLOAT, 0, GL_Utils.allocateFloatBuffer(0));
                    gl.glNormalPointer(GL10.GL_FLOAT, 0, GL_Utils.allocateFloatBuffer(0));
                }
            }
            if (cubenBuffer == null) {
                cubevBuffer = loadCubev(size);
                cubenBuffer = loadCuben();
                param = size;
            }
            gl.glVertexPointer(3, GL10.GL_FLOAT, 0, cubevBuffer);
            gl.glNormalPointer(GL10.GL_FLOAT, 0, cubenBuffer);
            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glEnableClientState(GL10.GL_NORMAL_ARRAY);
            for (int i = 0; i < 6; i++) {
                gl.glDrawArrays(GL10.GL_LINE_LOOP, 4 * i, 4);
            }
            gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glDisableClientState(GL10.GL_NORMAL_ARRAY);
        }

        private static FloatBuffer loadCuben() {
            if (cubenBuffer == null) {
                cubenBuffer = GL_Utils.allocateFloatBuffer(72 * 4);
                for (int i = 0; i < 72; i++) {
                    cubenBuffer.put(cuben[i]);
                }
                cubenBuffer.position(0);
            }
            return cubenBuffer;
        }

        private static FloatBuffer loadCubev(float size) {
            if (cubevBuffer == null) {
                size /= 2;
                cubevBuffer = GL_Utils.allocateFloatBuffer(72 * 4);
                for (int i = 0; i < 72; i++) {
                    cubevBuffer.put(cubev[i] * size);
                }
                cubevBuffer.position(0);
            }
            return cubevBuffer;
        }
    }

    public static class XYZ {
        public static int natoms;

        public static FloatBuffer xyzvBuffer = null;

        public static void draw(GL10 gl) {

            if (xyzvBuffer == null) { return; }

            //
            // Draw atoms
            //
            gl.glVertexPointer(3, GL10.GL_FLOAT, 0, xyzvBuffer);
            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            for (int i = 0; i < natoms; i++) {
                gl.glDrawArrays(GL10.GL_POINTS, i, 1);
            }
            gl.glDisableClientState(GL10.GL_VERTEX_ARRAY);
        }

        private static FloatBuffer loadXyzv(int N, float[] xyz) {
            xyzvBuffer = GL_Utils.allocateFloatBuffer(N * 4);
            for (int i = 0; i < N; i++) {
                xyzvBuffer.put(xyz[i] * scale);
            }
            xyzvBuffer.position(0);
            return xyzvBuffer;
        }

        public static void setXyz(int N, float[] xyz) {
            xyzvBuffer = loadXyzv(N, xyz);
            natoms = N / 3;
        }
    }

    public static float scale = 0.60f;

    public static void glutSolidCube(GL10 gl, float size) {
        gl.glColor4f(0.9f, 0.1f, 0.1f, 1.0f);
        SolidCube.draw(gl, size);
    }

    public static void glutWireCube(GL10 gl, float size) {
        gl.glColor4f(0.9f, 0.1f, 0.1f, 1.0f);
        WireCube.draw(gl, size);
    }

    public static void glutXYZ(GL10 gl) {
        gl.glColor4f(0.9f, 0.1f, 0.1f, 1.0f);
        XYZ.draw(gl);
    }

    public static void glutXYZBonds(GL10 gl) {
        gl.glColor4f(0.1f, 0.1f, 0.9f, 1.0f);
        Bonds.draw(gl);
    }

}
