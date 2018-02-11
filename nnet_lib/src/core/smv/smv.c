#include "core/smv/impls.h"

void convolution3d_smv(float* a,
                       float* kernels,
                       layer_t curr_layer,
                       int kern_start,
                       float* result) {
    convolution3d_smv_nhwc_fxp(a, kernels, curr_layer, kern_start, result);
}
