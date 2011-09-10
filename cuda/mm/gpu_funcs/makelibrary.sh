#!/bin/sh

NAME="libcudafuncs"
SOURCES="gpu_bonds.cu gpu_bonds_kernel.cu gpu_autocorrelation.cu gpu_autocorrelation_kernel.cu"
INCLUDES="-I/usr/local/cuda/include -I../thrust/"
LIBS="-L/usr/local/cuda/lib64 -lcudart"
OPTS="-Xcompiler -fPIC -arch sm_12"

echo "making library: " $NAME " ... "

nvcc --shared -o $NAME.so $SOURCES $INCLUDES $LIBS $OPTS

#echo "copying to lib folder ..."
#sudo cp $NAME.so /usr/lib/

echo "done"


# usage:
#
# #include "gpu_functions.h"
# gcc with -lcudafuncs
#
