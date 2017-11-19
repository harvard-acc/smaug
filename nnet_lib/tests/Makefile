# Run integration tests with this Makefile.
#
# Usage:
# 	make [target]
#
# where target is one of: all, smiv, monolithic, composable.

.PHONY: tests build smiv monolithic composable

smiv: 			ARCH=SMIV
monolithic: ARCH=MONOLITHIC
composable: ARCH=COMPOSABLE
eigen:      ARCH=EIGEN

build:
	@echo "Building test..."
	@$(MAKE) -C .. --no-print-directory clean-native
	@$(MAKE) -C .. --no-print-directory native ARCH=$(ARCH)

smiv: build
	@python run_tests.py ../build/nnet-smiv-native

monolithic: build
	@python run_tests.py ../build/nnet-monolithic-native

composable: build
	@python run_tests.py ../build/nnet-composable-native

eigen:
	@$(MAKE) -C .. --no-print-directory clean-eigen
	@$(MAKE) -C .. --no-print-directory eigen ARCH=$(ARCH)
	@python run_tests.py ../build/nnet-eigen-native

all:
	@$(MAKE) -C . --no-print-directory monolithic
	@$(MAKE) -C . --no-print-directory composable
	@$(MAKE) -C . --no-print-directory smiv
	@$(MAKE) -C . --no-print-directory eigen
