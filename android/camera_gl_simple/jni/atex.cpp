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
(JNIEnv* env, jobject jc, jint width, jint height){

	MSG("jni surface changed %d %d",width,height);


	return (jint)1;
}



/**
 *
 */
JNIEXPORT jint JNICALL
Java_com_mobile_jacket_Native_renderNative
(JNIEnv* env, jobject jc, jint drawWidth, jint drawHeight, jboolean forceRedraw, jbyteArray _yuv420sp){

	//MSG("%d %d",drawWidth,drawHeight);

	jbyte* yuv420sp = (env)->GetByteArrayElements( _yuv420sp, NULL);

	(env)->ReleaseByteArrayElements( (jbyteArray) _yuv420sp, yuv420sp, JNI_ABORT);


	return (jint)1;
}

