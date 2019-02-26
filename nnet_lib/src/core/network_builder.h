#ifndef _CORE_NETWORK_BUILDER_H_
#define _CORE_NETWORK_BUILDER_H_

#include <string>

#include "core/workspace.h"

namespace smaug {
Network* buildNetwork(const std::string& modelFile, Workspace* workspace);
}  // namespace smaug

#endif
