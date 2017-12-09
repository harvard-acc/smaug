// Refer to tensorflow/tensorflow/core/kernels/eigen_spatial_convolutions.h

#ifdef EIGEN_ARCH_IMPL

#include <iostream>

#include "unsupported/Eigen/CXX11/Tensor"

#include "core/eigen/convolution.h"
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

    typedef Tensor<float, 4, RowMajor> InputTensorType;
    typedef Tensor<float, 4, RowMajor> KernelTensorType;

    TensorMap<InputTensorType> input_map(
            activations, 1, a_height * NUM_TEST_CASES, a_rows, a_cols);
    TensorMap<KernelTensorType> kernel_map(
            kernels, num_kerns, k_height, k_rows, k_cols);
    TensorMap<InputTensorType> result_map(
            result, NUM_TEST_CASES, o_height, o_rows, o_cols);

    // RowMajor storage layout requires a specific reordering of the
    // dimensions.
    // TODO: Now that we support all the kernels in Eigen, use Eigen from the
    // very beginning to initialize the data into the right order and storage
    // layout so that these shuffles can be removed!
    array<ptrdiff_t, 4> preshuffle_idx({ 3, 2, 1, 0 });
    array<ptrdiff_t, 4> postshuffle_idx({ 2, 3, 1, 0 });
    Tensor<float, 2>::Dimensions input_precontract_dims;
    Tensor<float, 2>::Dimensions kernel_precontract_dims;
    Tensor<float, 4>::Dimensions postcontract_dims;

    // Precontract dimensions: the first dimension separates different output
    // pixels, and the second dimension goes over all the input pixels that
    // will be contracted.
    input_precontract_dims[0] = o_rows * o_cols * NUM_TEST_CASES;
    input_precontract_dims[1] = k_rows * k_cols * k_height;

    // Precontract dimensions: first dimension is the contraction dimension.
    kernel_precontract_dims[0] = k_rows * k_cols * k_height;
    kernel_precontract_dims[1] = num_kerns;

    // Postcontract dimensions: output dimensions.
    postcontract_dims[3] = o_height;
    postcontract_dims[2] = NUM_TEST_CASES;
    postcontract_dims[1] = o_rows;
    postcontract_dims[0] = o_cols;

    // The dimensions along which to perform the contraction.
    array<IndexPair<int>, 1> contract_dims = { IndexPair<int>(1, 0) };

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
    Tensor<float, 5, RowMajor> volume_patches =
            input_map.shuffle(preshuffle_idx)
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

    Tensor<float, 2, RowMajor> precontracted_patches =
            volume_patches.reshape(input_precontract_dims);
    for (int i : precontracted_patches.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped patches:\n" << precontracted_patches << "\n\n";

    Tensor<float, 2, RowMajor> precontracted_kernels =
        kernel_map.shuffle(preshuffle_idx).reshape(kernel_precontract_dims);
    std::cout << "\nReshaped kernel:\n" << precontracted_kernels << "\n\n";

    Tensor<float, 2, RowMajor> contracted = precontracted_patches.contract(
            precontracted_kernels, contract_dims);
    std::cout << "\nContraction:\n" << contracted << "\n\n";

    Tensor<float, 4, RowMajor> postcontracted =
            contracted.reshape(postcontract_dims);
    for (int i : postcontracted.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshaped:\n" << postcontracted << "\n\n";

    Tensor<float, 4, RowMajor> reshuffled =
            postcontracted.shuffle(postshuffle_idx);
    for (int i : reshuffled.dimensions())
      std::cout << i << ",";
    std::cout << "\nReshuffled:\n" << reshuffled << "\n\n";

    result_map = reshuffled;

#else

    // This is the "high-performance version".

    result_map = input_map.shuffle(preshuffle_idx)
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
                         .contract(kernel_map.shuffle(preshuffle_idx)
                                           .reshape(kernel_precontract_dims),
                                   contract_dims)
                         .reshape(postcontract_dims)
                         .shuffle(postshuffle_idx);

#endif

}

}  // namespace nnet_eigen

#endif
