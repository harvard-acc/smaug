#ifndef _MODELCONF_READ_MODEL_CONF_H_
#define _MODELCONF_READ_MODEL_CONF_H_

#include <string>

#include "core/workspace.h"

namespace smaug {
Network* buildNetwork(const std::string& modelFile, Workspace* workspace);
}  // namespace smaug

#endif
