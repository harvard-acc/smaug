.PHONY: help all test test-run clean tracer

help:
	@echo "Usage: make [option]"
	@echo ""
	@echo "Available targets:"
	@echo "  all: For execution on the host and gem5 simulation."
	@echo "  tracer: Instrumented binary for dynamic trace generation."
	@echo "  test: Compile all the tests."
	@echo "  test-run: Run all the tests."
	@echo "  clean: Clean up the build directory."

all:
	@$(MAKE) -f make/Makefile.native --no-print-directory all
test:
	@$(MAKE) -f make/Makefile.native --no-print-directory tests
test-run:
	@$(MAKE) -f make/Makefile.native --no-print-directory run-tests
clean:
	@$(MAKE) -f make/Makefile.native --no-print-directory clean
tracer:
	@$(MAKE) -f make/Makefile.tracer --no-print-directory dma-trace-binary
clean-trace:
	@$(MAKE) -f make/Makefile.tracer --no-print-directory clean-trace
