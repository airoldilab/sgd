#
# 'make'        build executable file
# 'make clean'  removes all .o and executable files
#

# Executables to build
EXEC += unittest

# Get the shell name to determine the OS
UNAME := $(shell uname)

# Define the C++ compiler to use
CXX := $(shell which clang++) -std=c++11

# Dependency directory and flags
DEPSDIR := $(shell mkdir -p .deps; echo .deps)
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = $(DEPSDIR)/$(notdir $*.d)
DEPSFLAGS = -MD -MF $(DEPSFILE) -MP

# Define any directories containing header files
#   To include directories use -Ipath/to/files
INCLUDES += -I.
#INCLUDES += -I/home/ky/MTL-4/usr/include

# Define CXX compile flags
CXXFLAGS += -O3 -funroll-loops -fopenmp -W -Wall -Wextra #-Wfatal-errors

# Define any directories containing libraries
#   To include directories use -Lpath/to/files
LDFLAGS +=

# Define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX
LDLIBS += -lSDL -lGL -lGLU

-ferror-limit=1

##################
# The following part of the makefile defines generic rules; it can be used to
# build any executable just by changing the definitions above.
#
#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule
##################

# 'make' - default rule
all: $(EXEC)

# Default rule for creating an exec of $(EXEC) from a .o file
$(EXEC): % : %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Default rule for creating a .o file from a .cpp file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPSFLAGS) -c -o $@ $<

# Extra dependencies for executables
#   Nothing here

# 'make clean' - deletes all .o files, exec, and dependency files
clean:
	-$(RM) *.o $(EXEC)
	$(RM) -r $(DEPSDIR)

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include $(wildcard $(DEPSDIR)/*.d)
