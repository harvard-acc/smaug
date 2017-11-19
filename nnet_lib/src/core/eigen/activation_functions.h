#ifndef _EIGEN_ACTIVATION_FUNCTIONS_H_
#define _EIGEN_ACTIVATION_FUNCTIONS_H_

#ifdef EIGEN_ARCH_IMPL

namespace nnet_eigen {

void activation_fun(float* inputs,
                    int size,
                    activation_type function,
                    float* sigmoid_table,
                    float* result);

};  // namespace nnet_eigen

#endif

#endif
