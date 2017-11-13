#include "confuse.h"

#include "nnet_fwd.h"

int configure_network_from_file(const char* cfg_file,
                                layer_t** layers_ptr,
                                device_t** device_ptr);
