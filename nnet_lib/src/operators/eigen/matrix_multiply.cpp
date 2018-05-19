#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "Eigen/Dense"

#include "core/eigen/matrix_multiply.h"
#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

using RowMajorMatrix = Matrix<float, Dynamic, Dynamic, RowMajor>;
using ColMajorMatrix = Matrix<float, Dynamic, Dynamic, ColMajor>;
using RowMajorMap = Map<RowMajorMatrix>;
using ColMajorMap = Map<ColMajorMatrix>;

// Even though all data in Eigen is initialized with a ColMajor matrix, a row
// vector cannot be colmajor or it could only have size (1,1).
using RowVectorType = Matrix<float, 1, Dynamic, RowMajor>;
using ColVectorType = Matrix<float, Dynamic, 1, ColMajor>;
using RowVectorMap = Map<RowVectorType>;
using ColVectorMap = Map<ColVectorType>;

namespace internal {

// A x B[0:-1, :] + B[-1, :] = C.
//
// Args:
//   A, C: An Eigen Map<Matrix>, which can be any StorageOrder.
//   B: An Eigen Map<Matrix, ColMajor> NxM matrix, where the last row contains
//     the biases.
template <typename InputMapType>
void gemm_bias_impl(InputMapType& inputs,
                    ColMajorMap& weights,
                    InputMapType& results) {
    // For column major matrix, we need to construct the map with the entire
    // matrix and then cut off the last row of biases before doing the
    // multiply.
    auto weights_block =
            weights.topLeftCorner(weights.rows() - 1, weights.cols());
    auto bias = weights.row(weights.rows() - 1);

#if DEBUG_LEVEL == 2
    std::cout << "weights:\n" << weights_block << std::endl;
    std::cout << "biases:\n" << bias << std::endl;
    std::cout << "vector multiply result (without bias):\n"
              << inputs * weights_block << std::endl;
#endif

    results.noalias() = (inputs * weights_block).rowwise() + bias;
}

// Vector-matrix multiply.
//
// Arguments:
//   vector: row vector.
//   matrix: column major matrix.
void vector_matrix_multiply_bias(float* __restrict__ vector,
                                 float* __restrict__ matrix,
                                 int v_width,
                                 int matrix_width,
                                 float* __restrict__ result) {
    RowVectorMap vector_map(vector, v_width);
    ColMajorMap matrix_with_bias_map(matrix, v_width + 1, matrix_width);
    RowVectorMap result_map(result, matrix_width);
    gemm_bias_impl(vector_map, matrix_with_bias_map, result_map);
}

void matrix_matrix_multiply_bias(float* __restrict__ a,
                                 float* __restrict__ b,
                                 int a_height,
                                 int b_height,
                                 int b_width,
                                 float* __restrict__ result) {
    int a_width = b_height - 1;
    ColMajorMap a_map(a, a_height, a_width);
    ColMajorMap b_with_bias_map(b, b_height, b_width);
    ColMajorMap result_map(result, a_height, b_width);
    gemm_bias_impl(a_map, b_with_bias_map, result_map);
}

}  // namespace internal.

// Multiply the matrices a and b, where b is stored columnwise. The last
// logical row of b are the biases.
//
// Args:
//   a_height = height of the A matrix.
//   b_height = height of the B matrix + 1 (for the biases).
//   b_width = width of the B matrix.
void matrix_multiply_with_bias(float* __restrict__ a,
                               float* __restrict__ b,
                               int a_height,
                               int b_height,
                               int b_width,
                               float* __restrict__ result) {
    if (a_height == 1) {
        internal::vector_matrix_multiply_bias(
                a, b, b_height - 1, b_width, result);
    } else {
        internal::matrix_matrix_multiply_bias(
                a, b, a_height, b_height, b_width, result);
    }
}

}  // namespace nnet_eigen

#endif
