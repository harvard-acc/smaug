#include <string>

#include "confuse.h"

#include "nnet_fwd.h"

#include "core/operator.h"
#include "core/network.h"
#include "core/workspace.h"

namespace smaug {

int configure_network_from_file(const char* cfg_file,
                                layer_t** layers_ptr,
                                device_t** device_ptr,
                                sampling_param_t** sampling_ptr);

Network* readModelConfiguration(const std::string& cfg_file,
                                Workspace* workspace);

}  // namespace smaug
