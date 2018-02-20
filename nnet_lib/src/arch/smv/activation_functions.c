#include "core/ref/activation_functions.h"
#include "core/smiv/activation_functions_simd.h"

void smv_activation_fun(float* local_activations,
                        float* acp_activations,
                        int batch_size,
                        int input_size,
                        activation_type activation) {
#ifdef ENABLE_SIMD_IMPL
    int num_vectors = input_size / VECTOR_SIZE;
    VEC_ARRAY_1D(v8fp_t, _local_activations, local_activations);
    VEC_ARRAY_1D(v8fp_t, _acp_activations, acp_activations);
    activation_vector:
    for (int i = 0; i < num_vectors; i++) {
        if (acp_activations) {
            _acp_activations[i] =
                    activation_fun_simd(_local_activations[i], activation);
        } else {
            _local_activations[i] =
                    activation_fun_simd(_local_activations[i], activation);
        }
    }
#else
    activation_fun(local_activations, batch_size, input_size, activation);
#endif
}
