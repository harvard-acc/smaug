#include "nnet_fwd.h"
#include "utility/utility.h"
#include "monolithic.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

// Runs the forward pass of a neural network.
//
// This version loads weights on a per layer basis, and activations are
// ping-ponged between two buffers, hid and hid_temp.
void nnet_fwd_monolithic(float* hid,
                         float* weights,
                         layer_t* layers,
                         int num_layers,
                         float* hid_temp,
                         float* sigmoid_table) {

    int i, j, l;
    layer_t curr_layer;

    // Alternate between reading from/writing to hid and hid_temp so we can
    // avoid copying matrices.
    bool result_in_temp = false;
    bool result_in_input = false;
    bool do_activation_func = true;

    if (PRINT_DATA_AND_WEIGHTS) {
        printf("DATA:\n");
        for (i = 0; i < NUM_TEST_CASES; i++) {
            printf("Datum %d:\n", i);
            for (j = 0; j < INPUT_DIM; j++) {
                printf("%e, ", hid[sub2ind(i, j, INPUT_DIM)]);
            }
            printf("\n");
        }
        printf("\nWEIGHTS:\n");
        for (i = 0; i < layers[1].input_rows; i++) {
            for (j = 0; j < layers[1].input_cols; j++) {
                printf("%f\n", weights[sub2ind(i, j, layers[1].input_cols)]);
            }
        }
        printf("\nEND WEIGHTS\n");
    }

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****

    l = 0;
    dmaLoad(hid, 0, 0, NUM_TEST_CASES * INPUT_DIM * sizeof(float));

    //******************//
    //   PRIMARY LOOP   //
    //******************//

nnet_fwd_outer:
    for (l = 0; l < num_layers; l++) {
        curr_layer = layers[l];
        // Don't run the activation function on the last layer.
        do_activation_func = (l != num_layers - 1);

        grab_matrix_dma(weights, l, layers);

        if (result_in_temp) {
            result_in_input = run_layer(hid_temp, weights, curr_layer, hid,
                                        sigmoid_table, do_activation_func);
        } else {
            result_in_input = run_layer(hid, weights, curr_layer, hid_temp,
                                        sigmoid_table, do_activation_func);
        }

        if (!result_in_input)
           result_in_temp = !result_in_temp;
    }

    layers[num_layers - 1].result_in_temp = (int)result_in_temp;

    if (result_in_temp)
        dmaStore(hid_temp, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    else
        dmaStore(hid, 0, 0, NUM_TEST_CASES * NUM_CLASSES * sizeof(float));
    dmaStore(layers, 0, 0, num_layers*sizeof(layer_t));
}
