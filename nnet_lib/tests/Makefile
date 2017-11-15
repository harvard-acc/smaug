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

build:
	@echo "Building test..."
	@$(MAKE) -C .. --no-print-directory clean-native
	@$(MAKE) -C .. --no-print-directory native ARCH=$(ARCH)

smiv: build
	@mv ../build/nnet-native ../build/nnet-native-smiv
	@python run_tests.py ../build/nnet-native-smiv

monolithic: build
	@mv ../build/nnet-native ../build/nnet-native-monolithic
	@python run_tests.py ../build/nnet-native-monolithic

composable: build
	@mv ../build/nnet-native ../build/nnet-native-composable
	@python run_tests.py ../build/nnet-native-composable

all:
	@$(MAKE) -C . --no-print-directory monolithic
	@$(MAKE) -C . --no-print-directory composable
	@$(MAKE) -C . --no-print-directory smiv
