# Workaround for the forwarding rules in multiple translation units.

CORE_SRCS = core/activation_functions.c \
						core/convolution.c \
						core/matrix_multiply.c \
						core/zeropad.c \
						core/pooling.c \
						core/flatten.c \
						core/smiv.c \
						core/smiv/convolution.c \
						core/smiv/convolution_simd.c \
						core/smiv/matrix_multiply.c \
						core/smiv/reduction.c

UTILITY_SRCS = utility/init_data.c \
							 utility/utility.c \
							 utility/model_conf_params.c \
							 utility/read_model_conf.c

ARCH_SRCS = arch/monolithic.c \
	 					arch/composable.c \
						arch/smiv.c \
					  arch/common.c

SRCS = $(CORE_SRCS) $(UTILITY_SRCS) $(ARCH_SRCS) nnet_fwd.c

ACCEL_NAME = nnet

include common/Makefile.arch
include common/Makefile.tracer
include common/Makefile.gem5

clean: clean-native clean-trace clean-gem5
