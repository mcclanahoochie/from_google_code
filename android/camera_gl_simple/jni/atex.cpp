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


#include <jni.h>
#include <string.h>
#include <math.h>
#include "Native.h"


/**
 *
 */
#define MSG(msg,...) do { __android_log_print(ANDROID_LOG_DEBUG, "gpgpu", __FILE__":%d(%s) " msg "\n", __LINE__, __FUNCTION__, ##__VA_ARGS__); } while (0)




/**
 *
 */
JNIEXPORT jint JNICALL
Java_com_mobile_jacket_Native_surfaceChangedNative
(JNIEnv* env, jobject jc, jint width, jint height) {

    MSG("jni surface changed %d %d", width, height);


    return (jint)1;
}



/**
 *
 */
JNIEXPORT jint JNICALL
Java_com_mobile_jacket_Native_renderNative
(JNIEnv* env, jobject jc, jint drawWidth, jint drawHeight, jboolean forceRedraw, jbyteArray _yuv420sp) {

    //MSG("%d %d",drawWidth,drawHeight);

    jbyte* yuv420sp = (env)->GetByteArrayElements(_yuv420sp, NULL);

    (env)->ReleaseByteArrayElements((jbyteArray) _yuv420sp, yuv420sp, JNI_ABORT);


    return (jint)1;
}

