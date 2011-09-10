#!/bin/sh
#in order to compile correctly you need to 
#1)move  this old libraries in a directory named somehow (eg old);
#2)install opencv

#this should only be necessary on matlab2008 and older.
#mkdir $MATLAB_HOME/sys/os/glnx86/old
#cd $MATLAB_HOME/sys/os/glnx86/old
#mv $MATLAB_HOME/sys/os/glnx86/libgcc_s.so.1 
#mv $MATLAB_HOME/sys/os/glnx86/libstdc++.so.6  
#mv $MATLAB_HOME/sys/os/glnx86/libstdc++.so.6.0.8

#sudo ln -sf /usr/lib64/libstdc++.so.6 /usr/local/matlab/sys/os/glnxa64/libstdc++.so.6

#install opencv (ubuntu)
#sudo apt-get install libcv1 libcv-dev libhighgui1 libhighgui-dev libcvaux1 libcvaux-dev

#on ubuntu just do this:
LIBS=`pkg-config --libs opencv` 
CFLAGS=`pkg-config --cflags opencv`
 
# matlab compile
mex read_video.cpp video.cpp abstractcapturesource.cpp $LIBS $CFLAGS
mex read_from_camera.cpp camera.cpp abstractcapturesource.cpp $LIBS $CFLAGS 


