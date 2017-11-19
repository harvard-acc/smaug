.PHONY: all help native eigen dma-trace-binary \
			  gem5-cpu gem5-accel \
				clean-trace clean-gem5 clean-native clean-eigen clean gem5
all:
	@echo "Usage: make [target] ARCH=[ARCH]"
	@echo ""
	@echo "Available targets:"
	@echo "  native, eigen, dma-trace-binary, gem5"
	@echo "  gem5: gem5-cpu, gem5-accel (aggregate target)"
	@echo ""
	@echo "Architectures:"
	@echo "  MONOLITHIC, COMPOSABLE, SMIV, EIGEN"
	@echo "  Default: MONOLITHIC"

help: all

native:
	@$(MAKE) -f common/Makefile.native --no-print-directory native

debug:
	@$(MAKE) -f common/Makefile.native --no-print-directory debug

debug-verbose:
	@$(MAKE) -f common/Makefile.native --no-print-directory debug-verbose

eigen:
	@$(MAKE) -f common/Makefile.eigen --no-print-directory eigen ARCH=EIGEN

dma-trace-binary:
	@$(MAKE) -f common/Makefile.tracer --no-print-directory dma-trace-binary

gem5-cpu:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-cpu

gem5-accel:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-accel

perftests:
	@$(MAKE) -f common/Makefile.perftests --no-print-directory all

run-perftest:
	@$(MAKE) -f common/Makefile.perftests --no-print-directory run

clean-trace:
	@$(MAKE) -f common/Makefile.tracer --no-print-directory clean-trace

clean-gem5:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory clean-gem5

clean-native:
	@$(MAKE) -f common/Makefile.native --no-print-directory clean-native

clean-eigen:
	@$(MAKE) -f common/Makefile.eigen --no-print-directory clean-eigen ARCH=EIGEN

clean-perftests:
	@$(MAKE) -f common/Makefile.perftests --no-print-directory clean-perftests

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native
