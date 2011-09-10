#include <jni.h>
#include <android/log.h>

//#ifndef _Included_com_mobile_jacket_Native
//#define _Included_com_mobile_jacket_Native

#ifdef __cplusplus
extern "C" {
#endif


    /**
     *
     */
    JNIEXPORT jint JNICALL
    Java_com_mobile_jacket_Native_surfaceChangedNative
    (JNIEnv* , jobject , jint, jint);


    /**
     *
     */
    JNIEXPORT jint JNICALL
    Java_com_mobile_jacket_Native_renderNative
    (JNIEnv* , jobject , jint, jint, jboolean, jbyteArray);



#ifdef __cplusplus
}
#endif

//#endif // included
