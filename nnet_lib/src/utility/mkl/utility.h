#ifndef _MKL_UTILITY_H_
#define _MKL_UTILITY_H_

#include "arch/nnet_mkl.h"

namespace nnet_mkl {

MklSession* get_session(device_t* device);

}  // namespace nnet_mkl

#endif
