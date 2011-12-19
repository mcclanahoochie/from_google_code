####################################################################### 

HOME    = $(PWD)
CC      = g++
CPP	    = g++
CFLAGS  = -Wall -Wextra -pedantic -O3 -Wno-long-long
COMMON_INC =

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
INCL_P     = -I$(HOME)/inc -I/usr/local/cuda/include  $(COMMON_INC)
LIBS       = -lm -lOpenCL -lrt
INCL_AMD   = -I$(HOME)/inc -I$(AMDAPPSDKROOT)/include  $(COMMON_INC)
LIBS_AMD   = -L$(AMDAPPSDKROOT)/lib/x86_64 $(LIBS)
CFLAGS_AMD = $(CFLAGS) -DATI_OS_LINUX
endif
ifeq ($(UNAME), Darwin)
INCL_P     = -I/System/Library/Frameworks/OpenCL.framework/Headers #$(HOME)/inc
INCL_P    += -I/Developer/GPU\ Computing/OpenCL/common/inc/  $(COMMON_INC)
LIBS       = -framework OpenCL
INCL_AMD   = $(INCL_P) 
LIBS_AMD   = $(LIBS)
CFLAGS    += -DMAC
CFLAGS_AMD = $(CFLAGS) -DMAC
endif

CFLAGS += `pkg-config --cflags opencv`
LIBS   += `pkg-config --libs opencv` 

SOURCES = simpleCL.c main.cpp
BIN = cell_hist_test

all:
	$(CC) $(CFLAGS) $(INCL_P) -c $(SOURCES)
	$(CC) $(CFLAGS) *.o  -o $(BIN) $(LIBS)

amd:
	$(CC) $(CFLAGS_AMD) $(INCL_AMD) -c $(SOURCES)
	$(CC) $(CFLAGS_AMD) *.o  -o $(BIN) $(LIBS_AMD)

cpp:
	$(CPP) $(CFLAGS) $(INCL_P) -c $(SOURCES)

cppAMD:
	$(CPP) $(CFLAGS_AMD) $(INCL_AMD) -c $(SOURCES)

clean:
	rm -f *.o $(BIN)

