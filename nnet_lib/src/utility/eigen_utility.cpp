#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "utility/eigen_utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

void print_debug4d(TensorMap<Tensor<float, 4>>& tensor) {
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

}  // namespace nnet_eigen

#endif
