#ifdef EIGEN_ARCH_IMPL

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"

namespace nnet_eigen {

using namespace ::Eigen;

void max_pooling(float* activations, float* result, layer_t curr_layer);

}  // namespace nnet_eigen

#endif
