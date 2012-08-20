CPPFLAGS += `pkg-config --cflags opencv`
LDFLAGS  += `pkg-config --libs opencv` 

AF_PATH ?= ../..
# if elsewhere: AF_PATH := /opt/local/arrayfire
BIN := webcam_demo
include $(AF_PATH)/examples/common.mk
LDFLAGS += -lafGFX
