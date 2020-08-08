#ifndef _CORE_NETWORK_BUILDER_H_
#define _CORE_NETWORK_BUILDER_H_

#include <string>

#include "smaug/core/workspace.h"
#include "smaug/core/network.h"
#include "smaug/operators/common.h"

namespace smaug {

/**
 * buildNetwork reads the specified model topology and parameters protobufs and
 * simulation sampling directives and returns a populated Network that can be
 * run.
 *
 * @param modelTopoFile The path to the model topology protobuf.
 * @param modelParamsFile The path to the model parameters protobuf, which
 * contains values for all tensors in the network (weights *and* inputs).
 * @param sampling Level of simulation sampling to apply to applicable kernels.
 * @param workspace Pointer to the global Workspace holding all tensors and
 * operators.
 */
Network* buildNetwork(const std::string& modelTopoFile,
                      const std::string& modelParamsFile,
                      SamplingInfo& sampling,
                      Workspace* workspace);
}  // namespace smaug

#endif
