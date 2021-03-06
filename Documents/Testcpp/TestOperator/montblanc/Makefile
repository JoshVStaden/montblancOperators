# Tensorflow includes and defines
TF_INC=$(shell python -c 'from __future__ import print_function; import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_CUDA=$(shell python -c 'from __future__ import print_function; import tensorflow as tf; print(int(tf.test.is_built_with_cuda()))')

TF_FLAGS=-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D_GLIBCXX_USE_CXX11_ABI=0

# Dependencies
DEPDIR:=.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS=-MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

# Define our sources, compiling CUDA code if it's enabled
ifeq ($(TF_CUDA), 1)
    SOURCES=$(wildcard *.cpp *.cu)
else
    SOURCES=$(wildcard *.cpp)
endif

# Define objects and shared_library
OBJECTS=$(addsuffix .o, $(basename $(SOURCES)))
LIBRARY=montblanc.so

# Compiler flags
INCLUDES= -I $(TF_INC)
CPPFLAGS=-std=c++11 $(TF_FLAGS) $(INCLUDES) -fPIC -fopenmp -O2 -march=native -mtune=native


LDFLAGS = -fPIC -fopenmp

# Compiler directives
COMPILE.cpp = g++ $(DEPFLAGS) $(CPPFLAGS) -c

all : $(LIBRARY)

%.o : %.cpp
	$(COMPILE.cpp) $<


clean :
	rm -f $(OBJECTS) $(LIBRARY)

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) $(LDFLAGS)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))
