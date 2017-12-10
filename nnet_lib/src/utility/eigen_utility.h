#ifndef _EIGEN_UTILITY_H_
#define _EIGEN_UTILITY_H_

#include "unsupported/Eigen/CXX11/Tensor"

namespace nnet_eigen {

using namespace ::Eigen;

// Print a 4D tensor.
//
// The dimensions are assumed to be:
// 1: depth (input image)
// 2. channels
// 3. rows
// 4. cols
void print_debug4d(TensorMap<Tensor<float, 4>>& tensor);

}  // namespace nnet_eigen

#endif
