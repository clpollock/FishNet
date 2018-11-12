clean:
	$(RM) $(ObjDir)/*
	$(RM) $(OutputFile)

$(ObjDir):
	@test -d $(ObjDir) || $(MKDIR) $(ObjDir)

$(shell mkdir -p $(DepDir) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DepDir)/$*.Td

COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(IncludePath) -c
POSTCOMPILE = mv -f $(DepDir)/$*.Td $(DepDir)/$*.d ; touch $@

%.o : %.cpp
$(ObjDir)/%.o : %.cpp
$(ObjDir)/%.o : %.cpp $(DepDir)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(DepDir)/%.d: ;
.PRECIOUS: $(DepDir)/%.d

-include $(patsubst %,$(DepDir)/%.d,$(basename $(Sources)))
