#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "utility/utility.h"

// Build the sigmoid lookup table.
//
// We can either build a centered table, storing values from -5 to 5, or a
// non-centered table, storing values from 0 to 5 (with -5 to 0 calculated by
// subtracting 1). The non-centered table can store the sigmoid with twice the
// precision and does not require the linear interpolation used by the centered
// table.
void init_sigmoid_table(float** table_ptr) {
    if (SIGMOID_IMPL == ExpUnit) {
        *table_ptr = NULL;
        return;
    }

    PRINT_MSG("Initializing sigmoid lookup table.\n");
    *table_ptr = (float*)malloc_aligned(SIG_TABLE_SIZE * sizeof(float));
    float sig_step = 0, x_sig = 0;
    if (SIGMOID_IMPL == CenteredLUT) {
        sig_step = (float)(SIG_RANGE) / (SIG_TABLE_SIZE - 1.0);
        x_sig = (float)SIG_MIN;
    } else if (SIGMOID_IMPL == NoncenteredLUT) {
        sig_step = (float)(SIG_MAX) / (SIG_TABLE_SIZE - 1.0);
        x_sig = 0;
    }

    for (int i = 0; i < SIG_TABLE_SIZE; i++) {
        (*table_ptr)[i] = conv_float2fixed(sigmoid(x_sig));
        // printf("%f, %f\n", x_sig, (*table_ptr)[i]);
        x_sig += sig_step;
    }
}

// Build an exponential table.
//
// This stores precomputed values for exp(x) from EXP_MIN to EXP_MAX.
void init_exp_table(float** table_ptr) {
    if (SIGMOID_IMPL == ExpUnit) {
        *table_ptr = NULL;
        return;
    }

    PRINT_MSG("Initializing exponential lookup table.\n");
    *table_ptr = (float*)malloc_aligned(EXP_TABLE_SIZE  * sizeof(float));
    float exp_step = (float)(EXP_RANGE) / (EXP_TABLE_SIZE  - 1.0);
    float x_exp = (float)EXP_MIN;
    for (int i = 0; i < EXP_TABLE_SIZE ; i++) {
        (*table_ptr)[i] = conv_float2fixed(exp(x_exp));
        // printf("%f, %f\n", x_exp, (*table_ptr)[i]);
        x_exp += exp_step;
    }
}
