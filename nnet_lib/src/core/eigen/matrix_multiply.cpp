#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "Eigen/Dense"

#include "core/nnet_fwd_defs.h"
#include "core/eigen/matrix_multiply.h"
#include "utility/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

using RowMajorMatrix = Matrix<float, Dynamic, Dynamic, RowMajor>;
using ColMajorMatrix = Matrix<float, Dynamic, Dynamic, ColMajor>;
using RowMajorMap = Map<RowMajorMatrix>;
using ColMajorMap = Map<ColMajorMatrix>;

using RowVectorType = Matrix<float, 1, Dynamic, RowMajor>;
using ColVectorType = Matrix<float, Dynamic, 1, RowMajor>;
using RowVectorMap = Map<RowVectorType>;
using ColVectorMap = Map<ColVectorType>;

// Multiply matrices a and b, assuming the last row of b are biases.
//
// Args:
//   a_height = height of A matrix.
//   b_height = height of the B matrix, which is also the width of the A matrix
//     + 1.
//   b_width = width of the B matrix.
void matrix_multiply_with_bias(float* __restrict__ a,
                               float* __restrict__ b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* __restrict__ result) {
    int a_width = b_height - 1;
    int num_weights = a_width * b_width;
    RowMajorMap a_map(a, a_height, a_width);
    RowMajorMap b_map(b, a_width, b_width);
    RowVectorMap bias(b + num_weights, b_width);

#if DEBUG_LEVEL == 2
    std::cout << "B matrix:\n" << b_map << std::endl;
#endif

    RowMajorMap result_map(result, a_height, b_width);
    result_map.noalias() = a_map * b_map;
    result_map.rowwise() += bias;
}

// Multiply the matrices a and b, where b is stored columnwise. The last
// logical row of b are still the biases.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the B matrix + 1.
//   b_width = width of the B matrix.
void matrix_multiply_with_bias_transpose(float* __restrict__ a,
                                         float* __restrict__ b,
                                         int a_height,
                                         int b_height,
                                         int b_width,
                                         float* __restrict__ result) {
    int a_width = b_height - 1;
    RowMajorMap a_map(a, a_height, a_width);
    // For column major matrix, we need to construct the map with the entire
    // matrix and then cut off the last row of biases before doing the GEMM.
    ColMajorMap b_with_bias_map(b, b_height, b_width);
    auto b_block = b_with_bias_map.topLeftCorner(a_width, b_width);
    auto bias = b_with_bias_map.row(b_height - 1);

#if DEBUG_LEVEL == 2
    std::cout << "B matrix transpose:\n" << b_block << std::endl;
#endif

    RowMajorMap result_map(result, a_height, b_width);
    result_map = a_map * b_block;
    result_map.rowwise() += bias;
}

}  // namespace nnet_eigen

#endif
