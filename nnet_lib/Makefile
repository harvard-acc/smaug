# Workaround for the forwarding rules in multiple translation units.

CORE_SRCS = core/activation_functions.c \
						core/convolution.c \
						core/matrix_multiply.c \
						core/zeropad.c \
						core/pooling.c

UTILITY_SRCS = utility/init_data.c \
							 utility/utility.c \
							 utility/model_conf_params.c \
							 utility/read_model_conf.c

SRCS = $(CORE_SRCS) $(UTILITY_SRCS) nnet_fwd.c

CFLAGS = -Wno-unused-but-set-variable
ACCEL_NAME = nnet
BMARK_SPECIFIC_CFLAGS = -DDMA_INTERFACE_V2
export WORKLOAD=nnet_fwd

include common/Makefile.tracer
include common/Makefile.gem5
