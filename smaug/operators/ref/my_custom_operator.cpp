#include "smaug/operators/common.h"
#include "smaug/operators/my_customer_operator.h"

#ifdef __cplusplus
extern "C" {
#endif
// By convention, we prefix all pointers into host memory with "host_".
void device_add(float* host_input0, 
		float* host_input1, 
		float* host_output, 
		float* spad0, 
		float* spad1, 
		float* spad2, 
		int size) {
	// Copy input data from host_inputN to spadN. The first argument to dmaLoad
	// or dmaStore is always the destination.
	dmaLoad(spad0, host_input0, size);
	dmaLoad(spad1, host_input1, size);
	for (int i = 0; i < size; i++) {
		// Accumulate the data from spad0 into spad1.
		// NOTE: This could be optimized more if we had three scratchpads instead
		// of two. This would be a great exercise for the reader :)
		//spad1[i] += spad0[i];
		spad2[i] = spad0[i] + spad1[i];
	}
	// Copy output data from spad1 back to the host.
	dmaStore(host_output, spad1, size);
}

#ifdef __cplusplus
}  // extern "C"
#endif

namespace smaug {

template <>
void MyCustom<ReferenceBackend>::run_tiled() {
    TiledTensor& input0 = tiledTensors[kInput0];
    TiledTensor& input1 = tiledTensors[kInput1];
    TiledTensor& output = tiledTensors[kOutput];

    Tensor* tensorOutput = getOutput(kInput0);

    for (int i = 0; i < input0.size(); i++) {
        Tensor* input0Tile = input0.getTileWithData(i);
        Tensor* input1Tile = input1.getTileWithData(i);
        Tensor* outputTile = output.getTileWithData(i);

        // Get handles to the actual underlying data storage. This performs a
        // dynamic_cast to the specified data type, which we verified is safe inside
        // validate().
        float* input0Data = input0Tile->data<float>();
        float* input1Data = input1Tile->data<float>();
        float* outputData = outputTile->data<float>();
        auto size = outputTile->getShape().size();
        int size_copy = size;

        // Set up the TLB mappings.
        mapArrayToAccelerator(
                ref::kMyCustomOperatorHw,  // The accelerator ID this TLB mapping is for.
                "host_input0",             // The name of the function argument in the kernel function.
                input0Data,                // The pointer to the data.
                size                       // The size of the TLB mapping
                );
        mapArrayToAccelerator(
                ref::kMyCustomOperatorHw, "host_input1",
                input1Data, size);
        mapArrayToAccelerator(
                ref::kMyCustomOperatorHw, "host_output",
                outputData, size);


        // Wrap the call to elementwise_add with invokeKernel.
        invokeKernel(ref::kMyCustomOperatorHw,  // our accelerator ID
                device_add, // if not simulating, the function to call
                            // All of the function call arguments.
                input0Data,
                input1Data,
                outputData,
                ref::spad0,
                ref::spad1,
                ref::spad2,
                (int)size_copy);
        //elementwise_add(input0Data, input1Data, outputData, outputTile->getShape().size());
    }
    // The results of the elementwise_add are stored in the tiled tensor. We need
    // to merge the data from the individual tiles back into a single contiguous
    // Tensor.
    flattenTiledTensor(tiledTensors[kOutput], tensorOutput);
}
}

