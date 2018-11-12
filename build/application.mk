
RootDir := $(abspath $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..)

include $(RootDir)/build/common.mk

LibPath                :=-L$(LibDir)
ifeq ($(OS),FreeBSD)
    LibPath += -L/usr/local/lib
endif
LinkerName             :=$(CXX)
LinkOptions            :=
OutputFile             :=$(BinDir)/$(Project)
Libs                   := $(addprefix -l, $(Dependencies) $(Libraries))

.PHONY: all clean
all: $(ObjDir) $(BinDir) $(OutputFile)

$(OutputFile): $(Objects)
	@$(MKDIR) $(@D)
	$(LinkerName) -o$(OutputFile) $(Objects) $(LibPath) $(Libs) $(LinkOptions)

$(BinDir):
	@test -d $(BinDir) || $(MKDIR) $(BinDir)

include $(RootDir)/build/compile.mk
