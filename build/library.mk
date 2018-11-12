
RootDir := $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..)

include $(RootDir)/build/common.mk

OutputFile = $(LibDir)/lib$(Project).a

AR	:= /usr/bin/ar rc

.PHONY: all clean
all: $(ObjDir) $(LibDir) $(OutputFile)

$(OutputFile): $(Objects)
	$(AR) $(OutputFile) $(Objects)

$(LibDir):
	@test -d $(LibDir) || $(MKDIR) $(LibDir)

include $(RootDir)/build/compile.mk
