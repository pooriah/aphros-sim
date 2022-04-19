# mesh size
m =64 64 64
bs=8 8 8
# number of processors
np = 8
# job time limit in minutes
tl = 600

include $(shell ap.makesim)

.PHONY: small

small:
	$(MAKE) cleanrun m='32 32 32' np=1
