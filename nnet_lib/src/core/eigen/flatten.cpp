#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"
#include "utility/eigen_utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

// Takes a ColMajor input tensor and runs im2row on it, returning it still in
// ColMajor format.
result_buf flatten_input(float* activations,
                         layer_t* prev_layer,
                         int num_images,
                         float* result) {
    typedef TensorMap<Tensor<float, 4>, Aligned64> ImageTensorMap;
    typedef TensorMap<Tensor<float, 2>, Aligned64> FlattenedTensorMap;
    ImageTensorMap input_map(
            activations,
            num_images,
            prev_layer->outputs.height,
            prev_layer->outputs.rows,
            prev_layer->outputs.cols + prev_layer->outputs.align_pad);
    FlattenedTensorMap result_map(
            result,
            num_images,
            prev_layer->outputs.height * prev_layer->outputs.rows *
                    (prev_layer->outputs.cols + prev_layer->outputs.align_pad));

    Eigen::array<ptrdiff_t, 4> preshuffle_idx({3,2,1,0});
    Eigen::array<ptrdiff_t, 2> postshuffle_idx({1,0});
    Tensor<float, 2>::Dimensions reshape_dims;
    reshape_dims[0] = num_images;
    reshape_dims[1] = result_map.dimensions()[1];

#if DEBUG_LEVEL >= 1
    Tensor<float, 4, RowMajor> shuffled =
            input_map.swap_layout().shuffle(preshuffle_idx);
    print_debug4d(shuffled);

    Tensor<float, 2, RowMajor> reshaped = shuffled.reshape(reshape_dims);
    std::cout << "reshaped:\n" << reshaped << std::endl;

    Tensor<float, 2, ColMajor> reswapped = reshaped.swap_layout();
    std::cout << "reswapped:\n" << reswapped << std::endl;

    Tensor<float, 2, ColMajor> final_result = reswapped.shuffle(postshuffle_idx);
    std::cout << "final:\n" << final_result << std::endl;

    result_map = final_result;

#else

    result_map = input_map.swap_layout()
                         .shuffle(preshuffle_idx)
                         .reshape(reshape_dims)
                         .swap_layout()
                         .shuffle(postshuffle_idx);

#endif

    return result;
}

}  // nnet_eigen

#endif
