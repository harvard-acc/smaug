#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"
#include "core/pooling.h"
#include "core/eigen/pooling.h"
#include "utility/eigen/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

// This implementation is based off of the TensorFlow implementation that Eigen
// (they have another one that is more homebrew and supposedly better
// performing).
//
// High level idea: Take the image, and extract patches out of it such that
// each patch is a vector representing all pixels in the region to be pooled.
// Reshape this into a suitable format so that it can be reduced vector-wise
// with maximum(), then reshape it back into the final expected output shape.
//
// Specifics vary based on whether we want to use the row major or col major
// implementation.

void max_pooling_rowmajor(float* activations, float* result, layer_t curr_layer) {
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
    TensorMap<InputTensorType> inputs_map(
            activations, NUM_TEST_CASES, hgt, in_rows, in_cols + in_pad);
    TensorMap<InputTensorType> results_map(
            result, NUM_TEST_CASES, hgt, out_rows, out_cols + out_pad);

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

    // Postreduce dims = final output shape.
    Tensor<float, 4>::Dimensions postreduce_dims;
    static const int idxBatch = 0;
    static const int idxChannel = 3;
    static const int idxRow = 2;
    static const int idxCol = 1;
    postreduce_dims[idxBatch] = NUM_TEST_CASES;
    postreduce_dims[idxChannel] = hgt;
    postreduce_dims[idxRow] = out_rows;
    postreduce_dims[idxCol] = out_cols + out_pad;

    // Shape suitable for reduction. Post reduction, we should end up with a 2D matrix.
    Tensor<float, 3>::Dimensions prereduce_dims;
    prereduce_dims[2] = postreduce_dims[0];
    prereduce_dims[1] = size * size;
    prereduce_dims[0] = postreduce_dims[1] * postreduce_dims[2] * postreduce_dims[3];

    // The axis along which we will reduce (equal to the one that is size *
    // size).
    Tensor<float, 1>::Dimensions reduction_dims(1);

    // The patch extraction process requires a particular order to the
    // dimensions, and they produce patches in an interleaved way.
    Eigen::array<ptrdiff_t, 4> preshuffle_idx({3,2,1,0});
    Eigen::array<ptrdiff_t, 4> postshuffle_idx({0,3,2,1});

    results_map = inputs_map
                          .shuffle(preshuffle_idx)
                          .extract_volume_patches(1, size, size, 1, stride,
                                                  stride, PADDING_VALID)
                          .reshape(prereduce_dims)
                          .maximum(reduction_dims)
                          .reshape(postreduce_dims)
                          .shuffle(postshuffle_idx);
#if DEBUG_LEVEL >= 2
    for (int n = 0; n < NUM_TEST_CASES; n++) {
        std::cout << "Final output.\n";
        for (int h = 0; h < hgt; h++) {
            std::cout << "channel " << h << "\n";
            for (int r = 0; r < out_rows; r++) {
                for (int c = 0; c < out_cols + out_pad; c++) {
                    printf("%4.6f ", results_map(n, h, r, c));
                }
                std::cout << "\n";
            }
        }
    }
#endif
}

void max_pooling_colmajor(float* activations, float* result, layer_t curr_layer) {
    int in_rows = curr_layer.inputs.rows;
    int in_cols = curr_layer.inputs.cols;
    int in_pad = curr_layer.inputs.align_pad;
    int out_rows = curr_layer.outputs.rows;
    int out_cols = curr_layer.outputs.cols;
    int out_pad = curr_layer.outputs.align_pad;
    int hgt = curr_layer.inputs.height;
    int stride = curr_layer.field_stride;
    int size = curr_layer.weights.cols;

    // Input activations are always in NCHW format, ColMajor.
    typedef Tensor<float, 4, ColMajor> InputTensorType;
    TensorMap<InputTensorType, Aligned64> inputs_map(
            activations, NUM_TEST_CASES, hgt, in_rows, in_cols + in_pad);
    TensorMap<InputTensorType, Aligned64> results_map(
            result, NUM_TEST_CASES, hgt, out_rows, out_cols + out_pad);

#if DEBUG_LEVEL >= 1
    print_debug4d(inputs_map);
#endif

    // Mold the input volume patches into a shape suitable for reduction. Post
    // reduction, we should end up with a 2D matrix.
    Tensor<float, 3>::Dimensions prereduce_dims;
    prereduce_dims[0] = NUM_TEST_CASES;
    prereduce_dims[1] = size * size;
    prereduce_dims[2] = hgt * out_rows * out_cols;

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

    // The axis along which we will reduce (equal to the one that is size *
    // size).
    Tensor<float, 1>::Dimensions reduction_dims(1);

#if DEBUG_LEVEL >= 1
    // Print out every step of the reduction.
    Tensor<float, 5> volume_patches = inputs_map.extract_volume_patches(
            1, size, size, 1, stride, stride, PADDING_VALID);
    for (int i : volume_patches.dimensions())
      std::cout << i << ",";
    std::cout << "\nVolume patches:\n" << volume_patches << "\n";

    Tensor<float, 3> prereduced = volume_patches.reshape(prereduce_dims);
    for (int i : prereduced.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped:\n" << prereduced << "\n";

    Tensor<float, 2> reduced = prereduced.maximum(reduction_dims);
    for (int i : reduced.dimensions())
      std::cout << i << ",";
    std::cout << "\nReduced:\n" << reduced << "\n";

    Tensor<float, 4> postreduced = reduced.reshape(postreduce_dims);
    for (int i : postreduced.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped:\n" << postreduced << "\n";

    results_map = postreduced;
#else
    results_map = inputs_map
                          .extract_volume_patches(1, size, size, 1, stride,
                                                  stride, PADDING_VALID)
                          .reshape(prereduce_dims)
                          .maximum(reduction_dims)
                          .reshape(postreduce_dims);

#endif

#if DEBUG_LEVEL >= 1
    print_debug4d(results_map);
#endif
}

void max_pooling(float* activations, float* result, layer_t curr_layer) {
    max_pooling_colmajor(activations, result, curr_layer);
}

}  // namespace nnet_eigen

#endif
