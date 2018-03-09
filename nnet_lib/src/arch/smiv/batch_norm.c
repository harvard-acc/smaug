#include "arch/common.h"
#include "arch/smiv/common.h"
#include "core/nnet_fwd_defs.h"
#include "core/ref/activation_functions.h"
#include "core/ref/batch_norm.h"
#include "core/smiv/smiv.h"
#include "utility/profiling.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

static void smiv_batch_norm_layer_hw(float* host_activations,
                                     float* host_weights,
                                     float* host_result,
                                     float* umem,
                                     float* spad0,
                                     float* spad1,
                                     bool input_in_spad0,
                                     layer_t* curr_layer) {
    // DMA in the weights (to UMEM)
    setReadyBits(umem, SMIV_UMEM_SIZE, 0);
    dmaLoad(umem, host_weights, WEIGHT_BYTES(curr_layer, 0));

    // DMA in the inputs (to SPAD0)
    if (curr_layer->input_req == IO_DMA) {
        if (input_in_spad0) {
            setReadyBits(spad0, SMIV_SPAD_SIZE, 0);
            grab_input_activations_dma(host_activations, spad0, curr_layer);
        } else {
            setReadyBits(spad1, SMIV_SPAD_SIZE, 0);
            grab_input_activations_dma(host_activations, spad1, curr_layer);
        }
    }

    // The main kernel
#ifdef ENABLE_SIMD_IMPL
    if (input_in_spad0) {
        batch_norm_simd_fxp(spad0, umem, curr_layer, NUM_TEST_CASES, spad1);
    } else {
        batch_norm_simd_fxp(spad1, umem, curr_layer, NUM_TEST_CASES, spad0);
    }
#else
    int input_size = curr_layer->inputs.rows * curr_layer->inputs.height *
                     (curr_layer->inputs.cols + curr_layer->inputs.align_pad);
    activation_type activation = curr_layer->activation;
    if (input_in_spad0) {
        batch_norm_fxp(spad0, umem, curr_layer, NUM_TEST_CASES, spad1);
        activation_fun(spad1, NUM_TEST_CASES, input_size,
                       curr_layer->outputs.align_pad, activation);
    } else {
        batch_norm_fxp(spad1, umem, curr_layer, NUM_TEST_CASES, spad0);
        activation_fun(spad0, NUM_TEST_CASES, input_size,
                       curr_layer->outputs.align_pad, activation);
    }
#endif

    // DMA out the result (from SPAD1)
    if (curr_layer->output_req == IO_DMA) {
        if (input_in_spad0)
            store_output_activations_dma(host_result, spad1, curr_layer);
        else
            store_output_activations_dma(host_result, spad0, curr_layer);
    }
}

void smiv_batch_norm_layer_impl(float* activations,
                                float* weights,
                                layer_t* layers,
                                int lnum,
                                float* result,
                                smiv_global* g_smiv,
                                device_t* device) {
    layer_t curr_layer = layers[lnum];
    if (device->use_hw_batch_norm) {
        int weights_size = WEIGHT_BYTES(layers, lnum);
        if (weights_size > SMIV_UMEM_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm weights are larger than the "
                            "UMEM - not currently supported!\n");
        }
        assert(weights_size <= SMIV_UMEM_SIZE);
        int inputs_size = INPUT_BYTES(layers, lnum);
        int outputs_size = OUTPUT_BYTES(layers, lnum);
        assert(inputs_size == outputs_size);
        if (inputs_size > SMIV_SPAD_SIZE) {
            fprintf(stderr, "[ERROR]: Batch norm inputs don't fit on the "
                            "scratchpad!\n");
        }
        assert(inputs_size <= SMIV_SPAD_SIZE);
        if (!device->use_hw_activation_func)
            curr_layer.activation = NO_ACTIVATION;

        // Flush cache lines for activations and weights.
        begin_ignored_profiling(lnum);
        flush_cache_range(activations, inputs_size);
        flush_cache_range(weights, weights_size);
        end_profiling();

        MAP_ARRAY_TO_ACCEL(g_smiv->kBatchNormHw, "host_activations",
                           activations, inputs_size);
        MAP_ARRAY_TO_ACCEL(g_smiv->kBatchNormHw, "host_weights",
                           weights, weights_size);
        MAP_ARRAY_TO_ACCEL(
                g_smiv->kBatchNormHw, "host_result", result, outputs_size);
        // TODO: For now, always put the input into spad0.
        INVOKE_KERNEL_PROF(g_smiv->kBatchNormHw, lnum, smiv_batch_norm_layer_hw,
                           activations, weights, result, g_smiv->umem,
                           g_smiv->spad0, g_smiv->spad1, true, &layers[lnum]);
    } else {
        begin_profiling(__func__, lnum);
        // The reference implementation is faster than MKL since we can
        // precompute some of the weights. We have an optimized MKL version,
        // but it just calls this same function, so there's no point going
        // through that overhead.
        batch_norm_fxp(
                activations, weights, &curr_layer, NUM_TEST_CASES, result);
        if (device->use_hw_activation_func) {
            int input_size = get_dims_size(&curr_layer.inputs);
            activation_fun(result, NUM_TEST_CASES, input_size,
                           curr_layer.outputs.align_pad, curr_layer.activation);
        }
        end_profiling();
    }
}
