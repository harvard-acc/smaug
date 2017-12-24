#ifdef EIGEN_ARCH_IMPL

#include <cassert>
#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/nnet_fwd_defs.h"
#include "utility/init_data.h"
#include "utility/utility.h"
#include "utility/eigen/init_data.h"
#include "utility/eigen/utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

using Tensor2d = Tensor<float, 2, ColMajor>;
using Tensor3d = Tensor<float, 3, ColMajor>;
using Tensor4d = Tensor<float, 4, ColMajor>;
using Tensor5d = Tensor<float, 5, ColMajor>;

void init_fc_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     data_init_mode mode) {
    TensorMap<Tensor3d, Aligned64> weights_map(
            weights, w_height, w_rows, w_cols);
    for (int h = 0; h < w_height; h++) {
        for (int i = 0; i < w_rows; i++) {
            for (int j = 0; j < w_cols; j++) {
                weights_map(h, i, j) = get_rand_weight(mode, 0);
            }
        }
    }
#if DEBUG_LEVEL >= 1
    Tensor<float, 2>::Dimensions reshape;
    reshape[0] = w_height * w_rows;
    reshape[1] = w_cols;
    std::cout << "Weights:\n" << weights_map.reshape(reshape) << std::endl;
#endif
}

void init_conv_weights(float* weights,
                       int w_depth,
                       int w_height,
                       int w_rows,
                       int w_cols,
                       data_init_mode mode) {
    TensorMap<Tensor4d, Aligned64> weights_map(
            weights, w_depth, w_height, w_rows, w_cols);
    for (int d = 0; d < w_depth; d++) {
        for (int h = 0; h < w_height; h++) {
            for (int i = 0; i < w_rows; i++) {
                for (int j = 0; j < w_cols; j++) {
                    weights_map(d, h, i, j) = get_rand_weight(mode, d);
                }
            }
        }
    }
}

void init_bn_weights(float* weights,
                     int w_height,
                     int w_rows,
                     int w_cols,
                     data_init_mode mode) {
    static const float kEpsilon = 1e-5;
    TensorMap<Tensor3d, Aligned64> weights_map(
            weights, w_height, w_rows, w_cols);

    for (int h = 0; h < w_height; h++) {
        for (int i = 0; i < w_rows; i++) {
            // BN parameters are stored in blocks of w_rows * w_cols.
            // The block order is:
            //   1. mean
            //   2. variance
            //   3. gamma
            //   4. beta
            for (int j = 0; j < w_cols; j++) {
                float val = get_rand_weight(mode, 0);
                bool is_variance_block = (i / (w_rows / 4)) == 1;
                if (is_variance_block) {
                    // Precompute 1/sqrt(var + eps).
                    val = val < 0 ? -val : val;
                    val = 1.0/(sqrt(val + kEpsilon));
                }

                weights_map(h, i, j) = val;
            }
        }
    }
}

void init_weights(float* weights,
                  layer_t* layers,
                  int num_layers,
                  data_init_mode mode) {
    int w_rows, w_cols, w_height, w_depth, w_offset, w_pad;

    assert(mode == RANDOM || mode == FIXED);
    w_offset = 0;
    printf("Initializing weights randomly\n");

    for (int l = 0; l < num_layers; l++) {
        get_weights_dims_layer(
                layers, l, &w_rows, &w_cols, &w_height, &w_depth, &w_pad);
        assert(w_pad == 0 && "Data alignment padding not required for Eigen!");
        float* curr_weight_buf = weights + w_offset;
        switch (layers[l].type) {
            case FC:
                init_fc_weights(
                        curr_weight_buf, w_height, w_rows, w_cols, mode);
                break;
            case CONV:
                init_conv_weights(curr_weight_buf, w_depth, w_height, w_rows,
                                  w_cols, mode);
                break;
            case BATCH_NORM:
                init_bn_weights(
                        curr_weight_buf, w_height, w_rows, w_cols, mode);
                break;
            default:
                continue;
        }
        w_offset += w_rows * w_cols * w_height * w_depth;
    }
    // NOTE: FOR SIGMOID ACTIVATION FUNCTION, WEIGHTS SHOULD BE BIG
    // Otherwise everything just becomes ~0.5 after sigmoid, and results are
    // boring
}

void init_data(float* data,
               network_t* network,
               int num_test_cases,
               data_init_mode mode) {
    int input_rows = network->layers[0].inputs.rows;
    int input_cols = network->layers[0].inputs.cols;
    int input_height = network->layers[0].inputs.height;
    int input_align_pad = network->layers[0].inputs.align_pad;
    int input_dim = input_rows * input_cols * input_height;

    assert(input_align_pad == 0 &&
           "Data alignment padding not required on Eigen!");
    assert(mode == RANDOM || mode == FIXED);

    printf("Initializing data randomly\n");
    TensorMap<Tensor4d> data_map(
            data, num_test_cases, input_height, input_rows, input_cols);

    for (int i = 0; i < num_test_cases; i++) {
        int offset = 0;
        for (int j = 0; j < input_height; j++) {
            for (int k = 0; k < input_rows; k++) {
                for (int l = 0; l < input_cols; l++) {
                    if (mode == RANDOM) {
                        data_map(i, j, k, l) = conv_float2fixed(gen_gaussian());
                    } else {
                        // Make each input image distinguishable.
                        data_map(i, j, k, l) =
                                1.0 * i + (float)offset / input_dim;
                        offset++;
                    }
                }
            }
        }
    }
#if DEBUG_LEVEL >= 1
    print_debug4d(data_map);
#endif
}

void init_labels(int* labels, size_t label_size, data_init_mode mode) {
    unsigned i;
    assert(mode == RANDOM || mode == FIXED);
    printf("Initializing labels randomly\n");
    for (i = 0; i < label_size; i++) {
        labels[i] = 0;  // set all labels to 0
    }
}

}  // namespace nnet_eigen

#endif
