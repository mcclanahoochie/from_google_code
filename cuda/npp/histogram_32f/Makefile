IFILES := $(shell ls *.cu)
OFILES := $(IFILES:.cu=)

CUDADIR := /usr/local/cuda
CUDALIB := $(CUDADIR)/lib
CUDAINC := $(CUDADIR)/include

INCDIRS := -I$(CUDAINC)
LIBDIRS := -L$(CUDALIB)

%LIBS := -lcuda -lcudart -lcurand -lnpp -lcublas
LIBS := -lcuda -lcudart -lnpp
CC := $(CUDADIR)/bin/nvcc
all:
	for IN in $(OFILES); do $(CC) $$IN.cu -w $(INCDIRS) $(LIBDIRS) $(LIBS) -o $$IN; done
debug:
	for IN in $(OFILES); do $(CC) $$IN.cu -w $(INCDIRS) $(LIBDIRS) $(LIBS) -o $$IN -D DEBUG; done
clean:
	for IN in $(OFILES); do rm $$IN; done
