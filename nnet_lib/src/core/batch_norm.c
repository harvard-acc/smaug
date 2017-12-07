#include "utility/utility.h"
#include "nnet_fwd.h"

#include "batch_norm.h"

#define EPS 1e-5

// Perform batch normalization on the data in @input.
void batch_norm_fxp(float* inputs,
                    float* weights,
                    int input_size,
                    int batch_size,
                    float* result) {

    int i, j;
    enum {
        MEAN,
        VARIANCE,
        GAMMA,
        BETA
    };
    ARRAY_2D(float, _weights, weights, input_size);
    ARRAY_2D(float, _inputs, inputs, batch_size);
    ARRAY_2D(float, _result, result, batch_size);

    PRINT_MSG_V("Batch normalization:\n");

    bn0:
    for (i = 0; i < input_size; i++) {
        bn1:
        for (j = 0; j < batch_size; j++) {
            _result[j][i] = (_inputs[j][i] - _weights[MEAN][i]) /
                                    sqrt(_weights[VARIANCE][i] + EPS) *
                                    _weights[GAMMA][i] +
                            _weights[BETA][i];
        }
    }
}

