.PHONY: all help native eigen dma-trace-binary \
			  gem5-cpu gem5-accel \
				new new-test new-clean \
				clean-trace clean-gem5 clean-native clean-eigen clean gem5

## SMAUG v2 BUILD RULES ##

new:
	@$(MAKE) -f common/Makefile.common all
new-test:
	@$(MAKE) -f common/Makefile.common tests
new-test-run:
	@$(MAKE) -f common/Makefile.common run-tests
new-clean:
	@$(MAKE) -f common/Makefile.common clean

## SMAUG v1 BUILD RULES ##

all:
	@echo "Usage: make [target] ARCH=[ARCH]"
	@echo ""
	@echo "Available targets:"
	@echo "  native: For execution on the host."
	@echo "  dma-trace-binary: Dynamic trace generation."
	@echo "  gem5-accel: Simulation on gem5 with HW acceleration."
	@echo "  gem5-cpu: Simulation on gem5 without HW acceleration."
	@echo "  eigen: For execution on the host, using the Eigen backend."
	@echo "  eigen-gem5: For simulation on gem5, using the Eigen backend."
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

eigen-gem5:
	@$(MAKE) -f common/Makefile.eigen --no-print-directory eigen-gem5 ARCH=EIGEN

mkl:
	@$(MAKE) -f common/Makefile.mkl --no-print-directory mkl ARCH=MKLDNN

mkl-gem5:
	@$(MAKE) -f common/Makefile.mkl --no-print-directory mkl-gem5 ARCH=MKLDNN

dma-trace-binary:
	@$(MAKE) -f common/Makefile.tracer --no-print-directory dma-trace-binary

gem5-cpu:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-cpu

gem5-accel:
	@$(MAKE) -f common/Makefile.gem5 --no-print-directory gem5-accel

perftests:
	@$(MAKE) -f common/Makefile.perftests --no-print-directory

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

clean-mkl:
	@$(MAKE) -f common/Makefile.mkl --no-print-directory clean-mkl ARCH=MKLDNN

clean-perftests:
	@$(MAKE) -f common/Makefile.perftests --no-print-directory clean-perftests

gem5: gem5-cpu gem5-accel

clean: clean-trace clean-gem5 clean-native clean-perftests
