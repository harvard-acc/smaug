#include <stdio.h>
#include <stdlib.h>

#include "arch/smiv/common.h"
#include "core/nnet_fwd_defs.h"

void init_smiv_work_cfg(smiv_work_cfg_t* cfg, unsigned num_iterations) {
    cfg->num_iterations = num_iterations;
    cfg->iteration = (dims_t*)malloc(sizeof(dims_t) * num_iterations);
}

void free_smiv_work_cfg(smiv_work_cfg_t* cfg) {
    free(cfg->iteration);
}

void print_smiv_work_cfg(smiv_work_cfg_t* cfg) {
    for (unsigned i = 0; i < cfg->num_iterations; i++) {
        INFO_MSG("Iteration %d: height=%d, rows=%d, cols=%d, pad=%d\n",
                 i,
                 cfg->iteration[i].height,
                 cfg->iteration[i].rows,
                 cfg->iteration[i].cols,
                 cfg->iteration[i].align_pad);
    }
}

bool smiv_is_supported_activation_func(layer_type ltype, activation_type func) {
    if (ltype == FC || ltype == CONV_STANDARD || ltype == CONV_POINTWISE ||
        ltype == BATCH_NORM || ltype == CONV_DEPTHWISE) {
        switch (func) {
            case NO_ACTIVATION:
            case RELU:
            case RELU_THRESHOLD:
            case LRELU:
            case ELU:
            case SELU:
            case TANH:
            case SIGMOID:
                return true;
            default:
                return false;
        }
    } else {
        return false;
    }
}
