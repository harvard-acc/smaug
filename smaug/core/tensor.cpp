#include "smaug/core/tensor.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/core/globals.h"
#include "smaug/utility/thread_pool.h"

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

void* TiledTensor::tileCopyWorker(void* _args) {
    auto args = reinterpret_cast<CopyTilesArgs*>(_args);
    TiledTensor* tiledTensor = args->tiledTensor;
    int start = args->start;
    int numTiles = args->numTiles;
    TileDataOperation op = args->op;
    for (int i = start; i < start + numTiles; i++) {
        Tile* tile = tiledTensor->getTile(i);
        if (op == Scatter)
            tiledTensor->copyDataToTile(tile);
        else if (op == Gather)
            tiledTensor->gatherDataFromTile(tile);
    }
    delete args;
}

void TiledTensor::parallelCopyTileData(TileDataOperation op) {
    int totalNumTiles = tiles.size();
    int numTilesPerThread = std::ceil(totalNumTiles * 1.0 / threadPool->size());
    int remainingTiles = totalNumTiles;
    while (remainingTiles > 0) {
        int numTiles = std::min(numTilesPerThread, remainingTiles);
        auto args = new CopyTilesArgs(
                this, totalNumTiles - remainingTiles, numTiles, op);
        int cpuid =
                threadPool->dispatchThread(tileCopyWorker, (void*)args);
        assert(cpuid != -1 && "Failed to dispatch thread!");
        remainingTiles -= numTiles;
    }
    threadPool->joinThreadPool();
}

void TiledTensor::copyDataToAllTiles() {
    // Don't copy if all the tiles have data filled.
    if (dataFilled)
        return;

    assert(origTensor != nullptr &&
           "TiledTensor must have the original tensor to copy data from!");
    if (fastForwardMode || !threadPool || tiles.size() == 1) {
        for (auto index = startIndex(); !index.end(); ++index)
            copyDataToTile(&tiles[index]);
    } else {
        parallelCopyTileData(Scatter);
    }
    dataFilled = true;
}

void TiledTensor::copyDataToTile(Tile* tile) {
    // Don't copy if the tile already has data,  or if the tile is the original
    // tensor (we have only one tile).
    if (tile->hasData || tile->tensor == origTensor)
        return;

    // Perform the data copy.
    assert(tile->hasOrigin &&
           "Must set the tile's origin in the original tensor!");
    if (useRawTensor) {
        // Use the raw tensor copy function for the unary tile.
        copyRawTensorData(tile->tensor, origTensor, 0, tile->origin[0],
                          tile->tensor->getShape().storageSize());
    } else {
        std::vector<int> dstOrigin(tile->tensor->ndims(), 0);
        copyTensorRegion(tile->tensor, origTensor, dstOrigin, tile->origin,
                         tile->tensor->getShape().dims());
    }
    tile->hasData = true;
}

void TiledTensor::untile() {
    assert(origTensor != nullptr &&
           "TiledTensor must have the original tensor to copy data to!");
    const TensorShape& tensorShape = origTensor->getShape();
    int ndims = tensorShape.ndims();
    if (tiles.size() == 1) {
        // No need to copy data if the tile is the original tensor.
        return;
    }

    if (fastForwardMode || !threadPool) {
        for (auto index = startIndex(); !index.end(); ++index)
            gatherDataFromTile(&tiles[index]);
    } else {
        parallelCopyTileData(Gather);
    }
}

void TiledTensor::gatherDataFromTile(Tile* tile) {
    // Perform the data copy.
    assert(tile->hasOrigin &&
           "Must set the tile's origin in the original tensor!");
    if (useRawTensor) {
        // Use the raw tensor copy function for the unary tile.
        copyRawTensorData(origTensor, tile->tensor, tile->origin[0], 0,
                          tile->tensor->getShape().storageSize());
    } else {
        std::vector<int> srcOrigin(tile->tensor->ndims(), 0);
        copyTensorRegion(origTensor,
                         tile->tensor,
                         tile->origin,
                         srcOrigin,
                         tile->tensor->getShape().dims());
    }
}

}  // namespace smaug
