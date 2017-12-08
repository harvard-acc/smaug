#include "utility/utility.h"
#include "nnet_fwd.h"

#include "batch_norm.h"

// Perform batch normalization on the data in @input.
void batch_norm_fxp(float* inputs,
                    float* weights,
                    int input_size,
                    int batch_size,
                    float* result) {

    int i, j;
    // The weights are divided into four blocks.
    enum {
        MEAN,
        VARIANCE,
        GAMMA,
        BETA
    };
    ARRAY_2D(float, _weights, weights, input_size);
    ARRAY_2D(float, _inputs, inputs, input_size);
    ARRAY_2D(float, _result, result, input_size);

    PRINT_MSG_V("Batch normalization:\n");

    bn_batch:
    for (i = 0; i < batch_size; i++) {
        bn_input:
        for (j = 0; j < input_size; j++) {
            float mean = _weights[MEAN][j];
            // This is precomputed to avoid having to run a sqrt and division
            // in hardware.
            float recip_sqrt_var = _weights[VARIANCE][j];
            float gamma = _weights[GAMMA][j];
            float beta = _weights[BETA][j];
            _result[i][j] =
                    ((_inputs[i][j] - mean) * recip_sqrt_var) * gamma + beta;
        }
    }
}
