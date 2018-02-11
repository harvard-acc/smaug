#include "core/ref/activation_functions.h"
#include "core/smiv/activation_functions_simd.h"

void smv_activation_fun(float* activations,
                        int batch_size,
                        int input_size,
                        activation_type activation) {
    activation_fun(activations, batch_size, input_size, activation);
}
