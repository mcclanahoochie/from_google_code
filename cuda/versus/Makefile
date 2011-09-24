# requires installation of latest opencv
# compiled from source with gpu support

CPPFLAGS += `pkg-config --cflags opencv`
LDFLAGS  += `pkg-config --libs opencv` -lopencv_gpu

BIN := versus
LDFLAGS+=-ljacketGFX
include ../Makefile.common
