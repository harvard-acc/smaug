#include common/Makefile.arch
#include common/Makefile.tracer
#include common/Makefile.gem5

native:
	@$(MAKE) -f common/Makefile.native --no-print-directory native

dma-trace-binary:
	@$(MAKE) -f common/Makefile.tracer --no-print-directory dma-trace-binary

gem5-cpu:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-cpu

gem5-accel:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-accel

clean-trace:
	@$(MAKE) -f common/Makefile.tracer --no-print-directory clean-trace

clean-gem5:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory clean-gem5

clean-native:
	@$(MAKE) -f common/Makefile.native --no-print-directory clean-native

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native
