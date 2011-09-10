##### TEMPLATE #####

#
# USE GCC 4.1 
# USE G++ 4.1
#

# Fill in the name of the output binary here
target    := cam-cuda 

# List of sources, with .c, .cu, and .cc extensions
sources   := sobel_kernel.cu sobel4_kernel.cu main.cpp CVcam.cpp

# Other things that need to be built, e.g. .cubin files
extradeps := 

# path to thrust folder
#THRUST  := -I../thrust/

# opencv
#opencv_includes := -I/usr/include/opencv -I/usr/local/include/opencv 
opencv_includes := -I/opt/local/include/opencv -I/opt/local/include/opencv2 -I/opt/local/include/
opencv_includes += -L/opt/local/lib/
opencv_libs := -lopencv_core -lopencv_highgui -lopencv_imgproc

# path to NVIDIA CUDA SDK
#NVIDIASDK := $(HOME)/NVIDIA_CUDA_SDK/


# Flags common to all compilers. You can set these on the comamnd line, e.g:
# $ make opt="" dbg="-g" warn="-Wno-deptrcated-declarations -Wall -Werror"

opt  ?= -O2 -arch sm_11    # use -arch sm_12 for shared atomics
dbg  ?= 
warn ?= -Wall -Werror -fno-strict-aliasing -O2


# This is where the cuda runtime libraries and includes reside.
cudaroot  := /usr/local/cuda


#----- C compilation options ------ 
gcc        := /usr/bin/gcc
cflags     :=  $(dbg) $(warn)
clib_paths := -L$(cudaroot)/lib
cinc_paths := -I$(cudaroot)/include $(opencv_includes)
clibraries := -lcuda -lcudart  $(opencv_libs)


#----- C++ compilation options ------
gpp         := /usr/bin/g++
ccflags     :=  $(dbg) $(warn)
cclib_paths := -L$(cudaroot)/lib 
ccinc_paths := -I$(cudaroot)/include $(opencv_includes)
cclibraries := -lcuda -lcudart  -lm  -lpthread $(opencv_libs)

#----- CUDA compilation options -----
nvcc        := $(cudaroot)/bin/nvcc
cuflags     := $(opt) $(dbg)   
culib_paths := -L$(cudaroot)/lib 
cuinc_paths := -I$(cudaroot)/include   $(THRUST)
culibraries := -lcuda -lcudart 

#-----
lib_paths   := $(culib_paths) $(cclib_paths) $(clib_paths) $(opencv_includes)
libraries   := $(culibraries) $(cclibraries) $(clibraries) 


#----- Generate source file and object file lists
# This code separates the source files by filename extension into C, C++,
# and Cuda files.

csources  := $(filter %.c, $(sources)) 
ccsources := $(filter %.cpp, $(sources)) 
#             $(filter %.cc,$(sources)) 
cusources := $(filter %.cu, $(sources))

# This code generates a list of object files by replacing filename extensions

objects := $(patsubst %.cu, %.o,$(cusources)) \
           $(patsubst %.c,  %.o,$(csources) ) \
           $(patsubst %.cpp,%.o,$(ccsources))
#           $(patsubst %.cc, %.o,$(ccsources)) \

#----- Build rules ------

$(target): $(extradeps)


$(target): $(objects)
	$(gcc) -o $@ $(lib_paths) $(libraries) $(objects)

%.o: %.cu
	$(nvcc) -c $^ $(cuflags) $(cuinc_paths) -o $@

%.cubin: %.cu
	$(nvcc) -cubin $(cuflags) $(cuinc_paths) $^

%.o: %.cc
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@

%.o: %.cpp
	$(gpp) -c $^ $(ccflags) $(ccinc_paths) -o $@	

%.o: %.c
	$(gcc) -c $^ $(cflags) $(cinc_paths) -o $@

clean:
	rm -f *.o $(target) makefile.*dep 


#----- Dependency Generation -----
#
# If a particular set of sources is non-empty, then have rules for
# generating the necessary dep files.
#

#ccdep := ccdep.mk
#cdep  := cdep.mk
#cudep := cudep.mk
ccdep := 
cdep  := 
cudep := 


depfiles = 

ifneq ($(ccsources),)

depfiles += $(ccdep)
$(ccdep): $(ccsources)
	$(gpp) -MM $(ccsources) > $(ccdep)

else

$(ccdep):

endif

ifneq ($(cusources),)

depfiles += $(cudep)
$(cudep):
	$(gpp) -MM -x c++ $(cusources) > $(cudep)

else

$(cudep):

endif

ifneq ($(csources),)

depfiles += $(cdep)
$(cdep): $(csources)
	$(gcc) -MM -x c $(csources) > $(cdep)

else

$(cdep):

endif

.PHONY: dep
dep: $(depfiles)


ifneq ($(MAKECMDGOALS),dep)
 ifneq ($(MAKECMDGOALS),clean)
  ifneq ($(ccsources),)
   include $(ccdep)
  endif
  ifneq ($(cusources),)
   include $(cudep)
  endif
  ifneq ($(csources),)
   include $(cdep)
  endif
 endif
endif
