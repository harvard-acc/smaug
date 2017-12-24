// Refer to tensorflow/tensorflow/core/kernels/eigen_spatial_convolutions.h

#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/eigen/convolution.h"
#include "utility/eigen/utility.h"
#include "nnet_fwd.h"

namespace nnet_eigen {

using namespace ::Eigen;

// Perform a 3D cuboid convolution over one 3D image and one 3D kernel.
void convolution3d(float* activations,
                   float* kernels,
                   layer_t* curr_layer,
                   float* result) {
    // Ignore alignment padding for now...
    const int a_rows = curr_layer->inputs.rows;
    const int a_cols = curr_layer->inputs.cols;
    const int a_height = curr_layer->inputs.height;
    const int c_pad = curr_layer->c_padding;

    const int o_rows = curr_layer->outputs.rows;
    const int o_cols = curr_layer->outputs.cols;
    const int o_height = curr_layer->outputs.height;

    // Filter is k_width x k_width x k_height.
    const int k_cols = curr_layer->weights.cols;
    const int k_rows = k_cols;
    const int k_height =  curr_layer->inputs.height;
    const int k_stride = curr_layer->field_stride;
    const int num_kerns = curr_layer->outputs.height;

    typedef Tensor<float, 4, ColMajor> InputTensorType;
    typedef Tensor<float, 4, ColMajor> KernelTensorType;

    TensorMap<InputTensorType> input_map(
            activations, NUM_TEST_CASES, a_height, a_rows, a_cols);
    TensorMap<KernelTensorType> kernel_map(
            kernels, num_kerns, k_height, k_rows, k_cols);
    TensorMap<InputTensorType> result_map(
            result, NUM_TEST_CASES, o_height, o_rows, o_cols);

    Tensor<float, 3>::Dimensions input_precontract_dims;
    Tensor<float, 2>::Dimensions kernel_precontract_dims;
    Tensor<float, 4>::Dimensions postcontract_dims;

    input_precontract_dims[0] = NUM_TEST_CASES;
    input_precontract_dims[1] = k_rows * k_cols * k_height;
    input_precontract_dims[2] = o_rows * o_cols;

    kernel_precontract_dims[0] = num_kerns;
    kernel_precontract_dims[1] = k_rows * k_cols * k_height;

    // Postcontract dimensions: output dimensions.
    postcontract_dims[0] = NUM_TEST_CASES;
    postcontract_dims[1] = o_height;
    postcontract_dims[2] = o_rows;
    postcontract_dims[3] = o_cols;

    // The dimensions along which to perform the contraction.
    array<IndexPair<int>, 1> contract_dims = { IndexPair<int>(1, 1) };

#if DEBUG_LEVEL >= 1
    std::cout << "input precontract dims:\n";
    for (int i : input_precontract_dims)
      std::cout << i << ",";
    std::cout << "\nkernel precontract dims:\n";
    for (int i : kernel_precontract_dims)
      std::cout << i << ",";
    std::cout << "\noutput postcontract dims:\n";
    for (int i : postcontract_dims)
      std::cout << i << ",";
    std::cout << "\n";
#endif


#if DEBUG_LEVEL >= 1
    Tensor<float, 5, ColMajor> volume_patches =
            input_map
                    .extract_volume_patches(k_height,
                                            k_rows,
                                            k_cols,
                                            k_height,  // z stride
                                            k_stride,
                                            k_stride,
                                            1,  // inflate strides
                                            1,
                                            1,
                                            0,  // z padding amount.
                                            0,
                                            c_pad,  // xy padding amount
                                            c_pad,
                                            c_pad,
                                            c_pad,
                                            0);  // padding value.

    auto patch_dims = volume_patches.dimensions();
    for (int i : patch_dims)
      std::cout << i << ",";
    std::cout << "\nVolume patches:\n" << volume_patches << "\n\n";

    Tensor<float, 3, ColMajor> precontracted_patches =
            volume_patches.reshape(input_precontract_dims);
    for (int i : precontracted_patches.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped patches:\n" << precontracted_patches << "\n\n";

    Tensor<float, 2, ColMajor> precontracted_kernels =
            kernel_map.reshape(kernel_precontract_dims);
    std::cout << "\nReshaped kernel:\n" << precontracted_kernels << "\n\n";

    Tensor<float, 3, ColMajor> contracted = precontracted_patches.contract(
            precontracted_kernels, contract_dims);
    for (int i : contracted.dimensions())
      std::cout << i << ",";
    std::cout << "\nContraction:\n" << contracted << "\n\n";

    Tensor<float, 4, ColMajor> postcontracted =
            contracted.reshape(postcontract_dims);
    for (int i : postcontracted.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped:\n" << postcontracted << "\n\n";

    result_map = postcontracted;

    std::cout << "Final:\n";
    print_debug4d(result_map);

#else

    // This is the "high-performance version".

    result_map = input_map
                         .extract_volume_patches(k_height,
                                                 k_rows,
                                                 k_cols,
                                                 k_height,  // z stride
                                                 k_stride,
                                                 k_stride,
                                                 1,  // inflate strides
                                                 1,
                                                 1,
                                                 0,  // z padding amount.
                                                 0,
                                                 c_pad,  // xy padding amount
                                                 c_pad,
                                                 c_pad,
                                                 c_pad,
                                                 0)  // padding value.
                         .reshape(input_precontract_dims)
                         .contract(kernel_map.reshape(kernel_precontract_dims),
                                   contract_dims)
                         .reshape(postcontract_dims);

#endif

}

}  // namespace nnet_eigen

#endif
