#include <assert.h>
#include <float.h>

#include "nnet_fwd.h"

#include "pooling.h"

// Downsample the input using a max-pooling operation.
//
// @input contains a stack of 3D images.
// The parameters of the pooling operation are given in @curr_layer.
//
// The downsampled result is placed into @result.
void max_pooling(float* input, float* result, layer_t curr_layer) {
    int ni;

maxpool_per_image:
    for (ni = 0; ni < NUM_TEST_CASES; ni++) {
        max_pooling_image3d(input, ni, result, curr_layer);
    }
}

// Downsample a single 3D input image.
void max_pooling_image3d(float* input, int img, float* result, layer_t curr_layer) {
    int h, i, j, k, l, oi, oj;
    float curr_max;

    int rows = curr_layer.input_rows;
    int cols = curr_layer.input_cols;
    int in_pad = curr_layer.input_data_align_pad;
    int hgt = curr_layer.input_height;
    int stride = curr_layer.field_stride;
    int size = curr_layer.field_size;

#if TREE_MAX == 1
    int total_pool_size = size * size;
    float elems[total_pool_size];
    int elem_idx;
#endif

    ARRAY_4D(float, _input, input, hgt, rows, cols + in_pad);
    ARRAY_4D(float, _result, result, hgt, curr_layer.output_rows,
             curr_layer.output_cols + curr_layer.output_data_align_pad);

maxpool_input_height:
    for (h = 0; h < hgt; h++) {
          oi = 0;
          oj = 0;
    maxpool_input_rows:
        for (i = 0; i < rows; i += stride) {
        maxpool_input_cols:
            for (j = 0; j < cols; j += stride) {
#if TREE_MAX == 1
                elem_idx = 0;
            maxpool_tree_outer:
                // Iterate over the pooling field.
                for (k = 0; k < size; k++) {
                maxpool_tree_inner:
                    for (l = 0; l < size; l++) {
                        elems[elem_idx] = _input[img][h][i+k][j+l];
                        elem_idx++;
                    }
                }

                if (total_pool_size == 4)
                    curr_max = max4(elems[0], elems[1], elems[2], elems[3]);
                else if (total_pool_size == 9)
                    curr_max = max9(elems[0], elems[1], elems[2], elems[3],
                                    elems[4], elems[5], elems[6], elems[7],
                                    elems[8]);
                else
                    assert(false && "Unsupported pooling size!");

#else
                curr_max = -FLT_MAX;
            maxpool_iter_outer:
                for (k = 0; k < size; k++) {
                maxpool_iter_inner:
                    for (l = 0; l < size; l++) {
                        float in_val = _input[img][h][i+k][j+l];
                        curr_max = max(in_val, curr_max);
                    }
                }
#endif

                _result[img][h][oi][oj] = curr_max;
                oj++;
            }
            oi++;
            oj = 0;
        }
    }
}
