#include "core/tensor.h"
#include "core/tensor_utils.h"

namespace smaug {

Tensor* TiledTensor::getTileWithData(int index) {
    Tile* tile = &tiles[index];
    copyDataToTile(tile);
    return tile->tensor;
}

void TiledTensor::setTile(int index,
                          const std::vector<int>& origin,
                          Tensor* tensor,
                          bool copyData) {
    Tile* tile = &tiles[index];
    tile->tensor = tensor;
    tile->origin = origin;
    tile->hasOrigin = true;
    if (copyData)
        copyDataToTile(tile);
}

void TiledTensor::copyDataToAllTiles() {
    for (auto index = startIndex(); !index.end(); ++index)
        copyDataToTile(&tiles[index]);
}

void TiledTensor::copyDataToTile(Tile* tile) {
    assert(origTensor != nullptr &&
           "TiledTensor must have the original tensor to copy data from!");
    // Don't copy if the tile already has data.
    if (tile->hasData)
        return;

    // Perform the data copy.
    assert(tile->hasOrigin &&
           "Must set the tile's origin in the original tensor!");
    std::vector<int> dstOrigin(tile->tensor->ndims(), 0);
    if (useRawTensor) {
        // Use the raw tensor copy function for the unary tile.
        copyRawTensorData(tile->tensor, origTensor, 0, tile->origin[0],
                          tile->tensor->getShape().storageSize());
    } else {
        copyTensorRegion(tile->tensor, origTensor, dstOrigin, tile->origin,
                         tile->tensor->getShape().dims());
    }
    tile->hasData = true;
}

}  // namespace smaug
