#ifndef _EIGEN_BATCH_NORM_H_
#define _EIGEN_BATCH_NORM_H_

namespace nnet_eigen {

void batch_norm(float* __restrict__ inputs,
                float* __restrict__ weights,
                int input_size,
                int batch_size,
                float* __restrict__ result);

}  // namespace nnet_eigen

#endif
