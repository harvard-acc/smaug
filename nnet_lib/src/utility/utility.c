#include <assert.h>
#include <string.h>
#include "nnet_fwd.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

#include "utility.h"

static float RAND_MAX_RECIPROCAL = (1.0/RAND_MAX);

float randfloat() {
  return rand() * RAND_MAX_RECIPROCAL;
}

#ifdef BITWIDTH_REDUCTION
float conv_float2fixed(float input) {
    // return input;
    int sign = 1;
    if (input < 0) {
        sign = -1;
    }
    long long int long_1 = 1;

    return sign *
           ((float)((long long int)(fabs(input) *
                                    (long_1 << NUM_OF_FRAC_BITS)) &
                    ((long_1 << (NUM_OF_INT_BITS + NUM_OF_FRAC_BITS)) - 1))) /
           (long_1 << NUM_OF_FRAC_BITS);
}
#endif

// Grab matrix n out of the doubly flattened w
// (w is a flattened collection of matrices, each flattened)
float* grab_matrix(float* w, int n, int* n_rows, int* n_columns) {
    int ind = 0;
    int i;
grab_matrix_loop:
    for (i = 0; i < n; i++) {
        ind += n_rows[i] * n_columns[i];
    }
    return w + ind;
}

size_t get_weights_loc_for_layer(layer_t* layers, int layer) {
    size_t offset = 0;
    int i;
    grab_matrix_dma_loop:
    for (i = 0; i < layer; i++) {
        offset += get_num_weights_layer(layers, i);
    }
    return offset;
}

#if defined(DMA_INTERFACE_V2)

#ifdef DMA_MODE

void grab_weights_dma(float* weights, int layer, layer_t* layers) {
    size_t offset = get_weights_loc_for_layer(layers, layer);
    size_t size = get_num_weights_layer(layers, layer) * sizeof(float);
#if DEBUG == 1
    printf("dmaLoad weights, offset: %lu, size: %lu\n", offset*sizeof(float), size);
#endif
    if (size > 0)
        dmaLoad(weights, offset*sizeof(float), 0, size);
}

// Fetch the input activations from DRAM.
// Useful for an accelerator with separate computational blocks.
size_t grab_input_activations_dma(float* activations, int layer, layer_t* layers) {
    size_t activations_size = get_input_activations_size(layers, layer);
    dmaLoad(activations, 0, 0, activations_size * sizeof(float));
    return activations_size;
}

size_t grab_output_activations_dma(float* activations, int layer, layer_t* layers) {
    size_t activations_size = get_output_activations_size(layers, layer);
    dmaLoad(activations, 0, 0, activations_size * sizeof(float));
    return activations_size;
}

size_t store_output_activations_dma(float* activations, int layer, layer_t* layers) {
    size_t activations_size = get_output_activations_size(layers, layer);
    dmaStore(activations, 0, 0, activations_size * sizeof(float));
    return activations_size;
}

#endif

int get_input_activations_size(layer_t* layers, int l) {
    int size = layers[l].inputs.rows * layers[l].inputs.height *
               (layers[l].inputs.cols + layers[l].inputs.align_pad);
    return size * NUM_TEST_CASES;
}

int get_output_activations_size(layer_t* layers, int l) {
    return (layers[l].outputs.rows) *
           (layers[l].outputs.height * NUM_TEST_CASES) *
           (layers[l].outputs.cols + layers[l].outputs.align_pad);
}

#elif defined(DMA_INTERFACE_V3)

#ifdef DMA_MODE

void grab_weights_dma(float* host_weights,
                      float* accel_weights,
                      int layer,
                      layer_t* layers) {
    size_t offset = get_weights_loc_for_layer(layers, layer);
    size_t size = get_num_weights_layer(layers, layer) * sizeof(float);
#if DEBUG == 1
    printf("dmaLoad weights, offset: %lu, size: %lu\n", offset*sizeof(float), size);
#endif
    if (size > 0)
        dmaLoad(accel_weights, &host_weights[offset], size);
}

size_t grab_input_activations_dma(float* host_activations,
                                  float* accel_activations,
                                  layer_t* layer) {
    size_t activations_size = get_input_activations_size(layer);
    dmaLoad(accel_activations, host_activations, activations_size * sizeof(float));
    return activations_size;
}

size_t grab_output_activations_dma(float* host_activations,
                                   float* accel_activations,
                                   layer_t* layer) {
    size_t activations_size = get_output_activations_size(layer);
    dmaLoad(accel_activations, host_activations, activations_size * sizeof(float));
    return activations_size;
}

size_t store_output_activations_dma(float* host_activations,
                                    float* accel_activations,
                                    layer_t* layer) {
    size_t activations_size = get_output_activations_size(layer);
    dmaStore(host_activations, accel_activations, activations_size * sizeof(float));
    return activations_size;
}

void flush_cache_range(void* src, size_t total_bytes) {
#ifdef GEM5_HARNESS
    char* ptr = (char*)src;
    for (size_t i = 0; i < total_bytes; i += CACHELINE_SIZE) {
        clflushopt(&ptr[i]);
    }
#else
#endif
}

#endif

int get_input_activations_size(layer_t* layer) {
    int size = layer->inputs.rows * layer->inputs.height *
               (layer->inputs.cols + layer->inputs.align_pad);
    return size * NUM_TEST_CASES;
}

int get_output_activations_size(layer_t* layer) {
    return (layer->outputs.rows) * (layer->outputs.height * NUM_TEST_CASES) *
           (layer->outputs.cols + layer->outputs.align_pad);
}

#endif

void clear_matrix(float* input, int size) {
    int i;
clear_loop:    for (i = 0; i < size; i++)
        input[i] = 0.0;
}

void copy_matrix(float* input, float* output, int size) {
    int i;
copy_loop:    for (i = 0; i < size; i++)
        output[i] = input[i];
}

int arg_max(float* input, int size, int increment) {
    int i;
    int j = 0;
    int max_ind = 0;
    float max_val = input[0];
arg_max_loop:    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] > max_val) {
            max_ind = i;
            max_val = input[j];
        }
    }
    return max_ind;
}

int arg_min(float* input, int size, int increment) {
    int i;
    int j = 0;
    int min_ind = 0;
    float min_val = input[0];
arg_min_loop:    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] < min_val) {
            min_ind = i;
            min_val = input[j];
        }
    }
    return min_ind;
}

// Return the difference between @value and the next multiple of @alignment.
int calc_padding(int value, unsigned alignment) {
    if (alignment == 0 || value % alignment == 0)
        return 0;
    return (alignment - (value % alignment));
}

// Get the dimensions of this layer's weights.
void get_weights_dims_layer(layer_t* layers,
                            int l,
                            int* num_rows,
                            int* num_cols,
                            int* num_height,
                            int* num_depth,
                            int* num_pad) {
    switch (layers[l].type) {
        case FC:
        case CONV_POINTWISE:
            *num_rows = layers[l].weights.rows;
            *num_cols = layers[l].weights.cols;
            *num_height = layers[l].weights.height;
            *num_depth = 1;
            *num_pad = layers[l].weights.align_pad;
            break;
        case CONV_STANDARD:
        case CONV_DEPTHWISE:
            *num_rows = layers[l].weights.rows;
            *num_cols = layers[l].weights.cols;
            *num_height = layers[l].weights.height;
            // # of this layer's kernels.
            *num_depth = layers[l].outputs.height;
            *num_pad = layers[l].weights.align_pad;
            break;
        case BATCH_NORM:
            *num_rows = layers[l].weights.rows;
            *num_cols = layers[l].weights.cols;
            *num_height = layers[l].weights.height;
            *num_depth = 1;
            *num_pad = layers[l].weights.align_pad;
            break;
        default:
            *num_rows = 0;
            *num_cols = 0;
            *num_height = 0;
            *num_depth = 0;
            *num_pad = 0;
            break;
    }
}

// Get the total number of weights for layer @l in the network.
int get_num_weights_layer(layer_t* layers, int l) {
    int num_weights;
    if (layers[l].type == FC || layers[l].type == CONV_POINTWISE) {
        // Assumes height = 1.
        if (TRANSPOSE_WEIGHTS == 1 && layers[l].type == FC) {
            num_weights = layers[l].weights.cols *
                   (layers[l].weights.rows + layers[l].weights.align_pad);
        } else {
            num_weights = layers[l].weights.rows *
                   (layers[l].weights.cols + layers[l].weights.align_pad);
        }
    } else if (layers[l].type == CONV_STANDARD ||
               layers[l].type == CONV_DEPTHWISE) {
        num_weights = layers[l].weights.rows *
               (layers[l].weights.cols + layers[l].weights.align_pad) *
               layers[l].weights.height * layers[l].outputs.height;
    } else if (layers[l].type == BATCH_NORM) {
        num_weights = layers[l].weights.rows *
               (layers[l].weights.cols + layers[l].weights.align_pad) *
               layers[l].weights.height;
    } else {
        num_weights = 0;
    }
    num_weights += layers[l].biases.rows * layers[l].biases.height *
                   (layers[l].biases.cols + layers[l].biases.align_pad);
    return num_weights;
}

// Get the total number of weights for the entire network.
int get_total_num_weights(layer_t* layers, int num_layers) {
    int l;
    int w_size = 0;
    for (l = 0; l < num_layers; l++) {
        w_size += get_num_weights_layer(layers, l);
    }
    return w_size;
}

// Compute the total size represented by this dims_t object.
//
// This should not be called from within an accelerated region!
size_t get_dims_size(dims_t* dims) {
    return dims->rows * (dims->cols + dims->align_pad) * dims->height;
}

// Compute the total size of this dims_t in NHWC format.
size_t get_nhwc_dims_size(dims_t* dims) {
    return dims->rows * dims->cols * (dims->height + dims->align_pad);
}

// Copy a range of columns from a 2D array data buffer to a new buffer.
//
// The data from section starting at (row, col) = (0, start_col) to (num_rows,
// start_col + num_cols) will be copied.
//
// Args:
//   original_data: Original data buffer
//   original_dims: Dimensions of this buffer. Height is ignored.
//   start_col: The starting column.
//   num_cols: Number of cols in the range to copy.
//   new_buffer: Destination buffer.
void copy_data_col_range(float* original_data,
                         dims_t* original_dims,
                         int start_col,
                         int num_cols,
                         float* new_buffer) {
    int num_rows = original_dims->rows * NUM_TEST_CASES;
    int num_total_cols =
            original_dims->cols + original_dims->align_pad;
    ARRAY_2D(float, _data, original_data, num_total_cols);
    for (int r = 0; r < num_rows; r++) {
        memcpy(new_buffer + r * num_cols,
               &_data[r][start_col],
               num_cols * sizeof(float));
    }
}

size_t next_multiple(size_t request, size_t align) {
  size_t n = request/align;
  if (n == 0)
    return align;  // Return at least this many bytes.
  size_t remainder = request % align;
  if (remainder)
      return (n+1)*align;
  return request;
}

// Return the error rate of the final network output.
float compute_errors(float* network_pred,
                     int* correct_labels,
                     int batch_size,
                     int num_classes) {
    int num_errors = 0;
    for (int i = 0; i < batch_size; i++) {
        if (arg_max(network_pred + i * num_classes, num_classes, 1) !=
            correct_labels[i]) {
            num_errors = num_errors + 1;
        }
    }
    return ((float)num_errors) / batch_size;
}

// Print the output labels and soft outputs.
void write_output_labels(const char* fname,
                         float* network_pred,
                         int batch_size,
                         int num_classes,
                         int pred_pad) {
    FILE* output_labels = fopen(fname, "w");
    ARRAY_2D(float, _network_pred, network_pred, num_classes + pred_pad);
    for (int i = 0; i < batch_size; i++) {
        int pred = arg_max(&_network_pred[i][0], num_classes, 1);
        fprintf(output_labels, "Test %d: %d\n  [", i, pred);
        for (int j = 0; j < num_classes; j++)
            fprintf(output_labels, "%f  ", _network_pred[i][j]);
        fprintf(output_labels, "]\n");
    }
    fclose(output_labels);
}

void print_debug(float* array,
                 int rows_to_print,
                 int cols_to_print,
                 int num_columns) {
    int i, l;
    for (i = 0; i < rows_to_print; i++) {
        for (l = 0; l < cols_to_print; l++) {
            printf("%f, ", array[sub2ind(i, l, num_columns)]);
        }
        printf("\n");
    }
}

void print_debug4d(float* array, int rows, int cols, int height) {
    int img, i, j, h;
    ARRAY_4D(float, _array, array, height, rows, cols);

    for (img = 0; img < NUM_TEST_CASES; img++) {
        printf("Input image: %d\n", img);
        for (h = 0; h < height; h++) {
            printf("Depth %d\n", h);
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                    printf("%f, ", _array[img][h][i][j]);
                }
                printf("\n");
            }
        }
    }
}

// Print data and weights of the first layer.
void print_data_and_weights(float* data, float* weights, layer_t first_layer) {
    int i, j;
    printf("DATA:\n");
    for (i = 0; i < NUM_TEST_CASES; i++) {
        printf("Datum %d:\n", i);
        for (j = 0; j < INPUT_DIM; j++) {
            printf("%e, ", data[sub2ind(i, j, INPUT_DIM)]);
        }
        printf("\n");
    }
    printf("\nWEIGHTS:\n");
    for (i = 0; i < first_layer.inputs.rows; i++) {
        for (j = 0; j < first_layer.inputs.cols; j++) {
            printf("%f\n", weights[sub2ind(i, j, first_layer.inputs.cols)]);
        }
    }
    printf("\nEND WEIGHTS\n");
}

void* malloc_aligned(size_t size) {
    void* ptr = NULL;
    int err = posix_memalign(
            (void**)&ptr, CACHELINE_SIZE, next_multiple(size, CACHELINE_SIZE));
    assert(err == 0 && "Failed to allocate memory!");
    return ptr;
}
