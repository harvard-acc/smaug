#include "core/ref/activation_functions.h"
#include "core/smiv/activation_functions_simd.h"

void smv_activation_fun_fxp(float* local_activations,
                            int batch_size,
                            int input_size,
                            int start_offset,
                            activation_type activation) {
#ifdef ENABLE_SIMD_IMPL
    int num_vectors = input_size / VECTOR_SIZE;
    int vec_start_offset = start_offset / VECTOR_SIZE;
    VEC_ARRAY_1D(v8fp_t, _local_activations, local_activations);
    activation_vector:
    for (int i = vec_start_offset; i < vec_start_offset + num_vectors; i++) {
        _local_activations[i] =
                activation_fun_simd_fxp(_local_activations[i], activation);
    }
#else
    activation_fun(local_activations, batch_size, input_size, activation);
#endif
}
