#ifndef _EIGEN_UTILITY_H_
#define _EIGEN_UTILITY_H_

#include <iostream>

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
template <int MapOptions>
void print_debug4d(TensorMap<Tensor<float, 4>, MapOptions>& tensor) {
    Tensor<float, 2>::Dimensions print_dims;
    auto input_dims = tensor.dimensions();
    print_dims[0] = input_dims[2];
    print_dims[1] = input_dims[3];
    for (int n = 0; n < input_dims[0]; n++) {
        std::cout << "Depth " << n << "\n";
        for (int h = 0; h < input_dims[1]; h++) {
            std::cout << "Channel " << h << "\n";
            array<long int, 4> offset = { n, h, 0, 0 };
            array<long int, 4> extent = { 1, 1, print_dims[0], print_dims[1] };
            std::cout << tensor.slice(offset, extent).reshape(print_dims)
                      << std::endl;
        }
    }
}

template <int MapOptions>
Tensor<float, 1> compute_hard_targets(
        TensorMap<Tensor<float, 2>, MapOptions>& soft_targets) {
    auto dims = soft_targets.dimensions();
    Tensor<float, 1> argmax(dims[0]);
    for (int i = 0; i < dims[0]; i++) {
        int index = 0;
        float max = soft_targets(i, 0);
        for (int j = 1; j < dims[1]; j++) {
            if (soft_targets(i, j) > max) {
                index = j;
                max = soft_targets(i, j);
            }
        }
        argmax[i] = index;
    }
    return argmax;
}

float compute_errors(float* network_soft_pred,
                     int* correct_labels,
                     int batch_size,
                     int num_classes);

void write_output_labels(const char* fname,
                         float* network_soft_pred,
                         int batch_size,
                         int num_classes);

}  // namespace nnet_eigen

#endif
