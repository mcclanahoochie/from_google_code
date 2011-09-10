#!/bin/sh

NAME="libcudafuncscpu"
SOURCES="cpu_bonds.cu cpu_autocorrelation.cu cpu_autocorrelation_kernel.cu"
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
# #include "cpu_functions.h"
# gcc with -lcudafuncscpu
#
