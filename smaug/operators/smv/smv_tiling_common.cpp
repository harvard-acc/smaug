#include "smaug/operators/smv/smv_tiling_common.h"
#include "smaug/core/tensor_utils.h"

namespace smaug {
namespace smv {

std::ostream& operator<<(std::ostream& os, const TilingDims& dims) {
  switch (dims) {
      case None:
          os << "None";
          break;
      case DimN:
          os << "DimN";
          break;
      case DimNC:
          os << "DimNC";
          break;
      case DimNH:
          os << "DimNH";
          break;
      case DimNW:
          os << "DimNW";
          break;
      case DimNHW:
          os << "DimNHW";
          break;
      case DimNCH:
          os << "DimNCH";
          break;
      case DimNCW:
          os << "DimNCW";
          break;
      case Invalid:
          os << "Invalid";
          break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TilingConfig& config) {
    os << "inputs: " << config.inputs;
    if (config.weights.ndims() != 0)
        os << ", weights: " << config.weights;
    os << ", outputs: " << config.outputs;
    return os;
}

// N means batch for inputs/outputs, whereas this can mean ofmap for convolution
// weights, or neuron for inner product weights.
bool needsNwiseTiling(TilingDims dim) {
    assert(dim != Invalid);
    return (dim != None);
}

// C means channel for convolution and activations for inner product.
bool needsCwiseTiling(TilingDims dim) {
    return (dim == DimNC) || (dim == DimNCH) || (dim == DimNCW);
}

// H means row for convolution.
bool needsHwiseTiling(TilingDims dim) {
    return (dim == DimNH) || (dim == DimNHW) || (dim == DimNCH);
}

// W means column for convolution.
bool needsWwiseTiling(TilingDims dim) {
    return (dim == DimNW) || (dim == DimNHW) || (dim == DimNCW);
}

}  // namespace smv
}  // namespace smaug
