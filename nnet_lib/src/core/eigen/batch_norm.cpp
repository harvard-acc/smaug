#ifdef EIGEN_ARCH_IMPL

#include "Eigen/Dense"

#include "batch_norm.h"

namespace nnet_eigen {

using namespace ::Eigen;

using RowMajorMatrixDyn = Matrix<float, Dynamic, Dynamic, RowMajor>;
using RowMajorMap = Map<RowMajorMatrixDyn, Aligned64>;
using RowVector = Matrix<float, 1, Dynamic, RowMajor>;
using RowVectorMap = Map<RowVector, Aligned8>;

void batch_norm(float* __restrict__ inputs,
                float* __restrict__ weights,
                int input_size,
                int batch_size,
                float* __restrict__ result) {
    // The weights are divided into four blocks.
    enum {
        MEAN,
        VARIANCE,
        GAMMA,
        BETA
    };
    RowMajorMap inputs_map(inputs, batch_size, input_size);
    RowMajorMap results_map(result, batch_size, input_size);
    RowVectorMap means(weights + MEAN * input_size, input_size);
    RowVectorMap variances(weights + VARIANCE * input_size, input_size);
    RowVectorMap gamma(weights + GAMMA * input_size, input_size);
    RowVectorMap beta(weights + BETA * input_size, input_size);

    // The 1.0/sqrt(var + eps) is precomputed, so it just turns into a
    // multiply.
    results_map.noalias() = ((inputs_map.rowwise() - means) *
                             variances.cwiseProduct(gamma).asDiagonal())
                                    .rowwise() +
                            beta;
}

}  // namespace nnet_eigen

#endif
