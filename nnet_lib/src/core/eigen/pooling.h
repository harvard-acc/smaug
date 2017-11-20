#ifdef EIGEN_ARCH_IMPL

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"

namespace nnet_eigen {

using namespace ::Eigen;

void max_pooling_rowmajor(float* activations, float* result, layer_t curr_layer);
void max_pooling_colmajor(float* activations, float* result, layer_t curr_layer);

// Default implementation is RowMajor.
void max_pooling(float* activations, float* result, layer_t curr_layer);

}  // namespace nnet_eigen

#endif
