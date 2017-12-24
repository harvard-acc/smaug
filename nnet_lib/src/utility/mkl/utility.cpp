#include "core/nnet_fwd_defs.h"
#include "arch/nnet_mkl.h"

nnet_mkl::MklSession* nnet_mkl::get_session(device_t* device) {
    return reinterpret_cast<nnet_mkl::MklSession*>(device->session);
}
