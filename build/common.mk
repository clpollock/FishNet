Mode = Release
OS = $(shell uname)

CXXFLAGS = -std=c++14 -pthread -Wall -Wno-dangling-else

ifeq ($(Mode),Debug)
    CXXFLAGS += -g -D_DEBUG
else ifeq ($(Mode),Release)
    CXXFLAGS += -O3
else
    $(error Mode must be either Debug or Release.)
endif

ifeq ($(OS),FreeBSD)
    CXX := /usr/local/bin/clang++
else
    CXX := /usr/bin/clang++
endif

MKDIR	:= mkdir -p

OutputDir = $(RootDir)/$(Mode)
DepDir = $(OutputDir)/depend/$(Project)
ObjDir = $(OutputDir)/obj/$(Project)
LibDir = $(OutputDir)/lib
BinDir = $(OutputDir)/bin

Objects = $(addprefix $(ObjDir)/, $(Sources:.cpp=.o))

IncludePath = $(addprefix -I$(RootDir)/, $(Dependencies)) $(addprefix -I, $(Include))
