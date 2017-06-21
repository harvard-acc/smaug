# Workaround for the forwarding rules in multiple translation units.

CORE_SRCS = core/activation_functions.c \
						core/convolution.c \
						core/matrix_multiply.c \
						core/zeropad.c \
						core/pooling.c \
						core/flatten.c \
						core/smiv_core.c

UTILITY_SRCS = utility/init_data.c \
							 utility/utility.c \
							 utility/model_conf_params.c \
							 utility/read_model_conf.c

ARCH_SRCS = arch/monolithic.c \
	 					arch/composable.c \
						arch/smiv.c \
					  arch/common.c

SRCS = $(CORE_SRCS) $(UTILITY_SRCS) $(ARCH_SRCS) nnet_fwd.c

CFLAGS = -Wno-unused-but-set-variable
ACCEL_NAME = nnet
BMARK_SPECIFIC_CFLAGS = -DDMA_INTERFACE_V2
# export WORKLOAD=nnet_fwd_hw
export WORKLOAD=inner_product_layer_hw,convolution_layer_hw,max_pooling_layer_hw,activation_hw

include common/Makefile.tracer
include common/Makefile.gem5

clean: clean-native clean-trace
