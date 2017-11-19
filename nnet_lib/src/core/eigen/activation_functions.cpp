#ifdef EIGEN_ARCH_IMPL

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"
#include "core/eigen/activation_functions.h"
#include "utility/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

void activation_fun(float* inputs,
                    int size,
                    activation_type function,
                    float* sigmoid_table,
                    float* result) {
  TensorMap<Tensor<float, 1>> input_tensor(inputs, size);
  TensorMap<Tensor<float, 1>> result_tensor(result, size);
  if (function == SIGMOID) {
    result_tensor = input_tensor.sigmoid();
  }
}

}  // namespace nnet_eigen

#endif
