all:
	$(MAKE) -C Utils
	$(MAKE) -C FishNet
	$(MAKE) -C Classifier

clean:
	$(MAKE) -C Utils clean
	$(MAKE) -C FishNet clean
	$(MAKE) -C Classifier clean
