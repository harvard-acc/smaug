#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"
#include "core/eigen/pooling.h"
#include "utility/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

void max_pooling(float* activations, float* result, layer_t curr_layer) {
    int in_rows = curr_layer.inputs.rows;
    int in_cols = curr_layer.inputs.cols;
    int in_pad = curr_layer.inputs.align_pad;
    int out_rows = curr_layer.outputs.rows;
    int out_cols = curr_layer.outputs.cols;
    int out_pad = curr_layer.outputs.align_pad;
    int hgt = curr_layer.inputs.height;
    int stride = curr_layer.field_stride;
    int size = curr_layer.weights.cols;

    // Input activations are always in NCHW format, RowMajor.
    typedef Tensor<float, 4, RowMajor> InputTensorType;
    typedef Tensor<float, 4, ColMajor> ColInputTensorType;
    TensorMap<InputTensorType> inputs_map(
            activations, NUM_TEST_CASES, hgt, in_rows, in_cols + in_pad);
    TensorMap<ColInputTensorType> results_map(
            result, out_cols + out_pad, out_rows, hgt, NUM_TEST_CASES);

#if DEBUG_LEVEL >= 2
    for (int n = 0; n < NUM_TEST_CASES; n++) {
        std::cout << "Input image " << n << "\n";
        for (int h = 0; h < hgt; h++) {
            std::cout << "Channel " << h << "\n";
            for (int r = 0; r < in_rows; r++) {
                for (int c = 0; c < in_cols + in_pad; c++) {
                    printf("%4.6f ", inputs_map(n, h, r, c));
                }
                std::cout << "\n";
            }
        }
    }
#endif

    // This implementation is based off of the TensorFlow implementation that
    // Eigen (they have another one that is more homebrew and supposedly better
    // performing).
    //
    // High level idea: Take the image, and extract patches out of it such that
    // each patch is a vector representing all pixels in the region to be
    // pooled. Reshape this into a suitable format so that it can be reduced
    // vector-wise with maximum(), then reshape it back into the final expected
    // output shape.

    // Postreduce dims = final output shape.
    Tensor<float, 4>::Dimensions postreduce_dims;
    static const int idxBatch = 0;
    static const int idxChannel = 1;
    static const int idxRow = 2;
    static const int idxCol = 3;
    postreduce_dims[idxBatch] = NUM_TEST_CASES;
    postreduce_dims[idxChannel] = hgt;
    postreduce_dims[idxRow] = out_rows;
    postreduce_dims[idxCol] = out_cols + out_pad;

    // Shape suitable for reduction. Post reduction, we should end up with a 2D matrix.
    Tensor<float, 3>::Dimensions prereduce_dims;
    prereduce_dims[0] = postreduce_dims[0];
    prereduce_dims[1] = size * size;
    prereduce_dims[2] = postreduce_dims[1] * postreduce_dims[2] * postreduce_dims[3];

    // The axis along which we will reduce (equal to the one that is size *
    // size).
    Tensor<float, 1>::Dimensions reduction_dims(1);

    // The data arrives in row major order. We need to turn it into column
    // major, but we need to preserve the data layout of NCHW. swap_layout()
    // will convert rowmajor to colmajor, but it will also reverse the
    // dimensionality, so the shuffles will reverse it right back.
    Eigen::array<ptrdiff_t, 4> preshuffle_idx({3,2,1,0});
    Eigen::array<ptrdiff_t, 4> postshuffle_idx({3,2,1,0});

    results_map = inputs_map.shuffle(preshuffle_idx)
                          .swap_layout()
                          .extract_volume_patches(1, size, size, 1, stride,
                                                  stride, PADDING_VALID)
                          .reshape(prereduce_dims)
                          .maximum(reduction_dims)
                          .reshape(postreduce_dims)
                          .shuffle(postshuffle_idx);
#if DEBUG_LEVEL >= 1
    for (int n = 0; n < NUM_TEST_CASES; n++) {
        std::cout << "Final output.\n";
        for (int h = 0; h < hgt; h++) {
            std::cout << "channel " << h << "\n";
            for (int r = 0; r < out_rows; r++) {
                for (int c = 0; c < out_cols + out_pad; c++) {
                    // Since the final data MUST be in row major order, but
                    // this map is colmajor, we have to manually reverse the
                    // indices.
                    printf("%4.6f ", results_map(c, r, h, n));
                }
                std::cout << "\n";
            }
        }
    }
#endif
}

}  // namespace nnet_eigen

#endif
