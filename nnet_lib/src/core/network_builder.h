#ifndef _CORE_NETWORK_BUILDER_H_
#define _CORE_NETWORK_BUILDER_H_

#include <string>

#include "core/workspace.h"
#include "operators/common.h"

namespace smaug {
Network* buildNetwork(const std::string& modelTopoFile,
                      const std::string& modelParamsFile,
                      SamplingInfo& sampling,
                      Workspace* workspace);
}  // namespace smaug

#endif
