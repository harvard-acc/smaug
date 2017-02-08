# Workaround for the forwarding rules in multiple translation units.
SRCS = activation_functions.c \
			 nnet_fwd.c \
			 utility.c \
			 init_data.c

CFLAGS = -Wno-unused-but-set-variable
ACCEL_NAME = nnet
BMARK_SPECIFIC_CFLAGS = -DDMA_INTERFACE_V2
export WORKLOAD=nnet_fwd

include common/Makefile.tracer
include common/Makefile.gem5
