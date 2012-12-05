CPPFLAGS += `pkg-config --cflags opencv`
LDFLAGS  += `pkg-config --libs opencv` 

#LDFLAGS  +=  -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video 
#CPPFLAGS +=  -I/usr/local/include/opencv/ 

AF_PATH ?= ../..
# if elsewhere: AF_PATH := /opt/local/arrayfire
BIN := webcam_demo_19
include $(AF_PATH)/examples/common.mk
LDFLAGS += -lafGFX
