#
#
#
#
LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_LDLIBS := -L$(SYSROOT)/usr/lib -llog

LOCAL_MODULE    := atex

LOCAL_SRC_FILES := atex.cpp

include $(BUILD_SHARED_LIBRARY)
