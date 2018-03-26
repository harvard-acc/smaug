#include <assert.h>
#include "arch/common.h"
#include "arch/smiv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "core/nnet_fwd_defs.h"
#include "core/smiv/params.h"
#include "core/smiv/smiv.h"
#include "utility/compression.h"
#include "utility/utility.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

// Decompress a CSR array in HW.
//
// The compressed data will be sent to one of the scratchpads, and the
// decompressed data will be written to the UMEM.
//
// This one function can be used for any of the available input mechanisms
// (dma/acp/cache), although much of the complexity is a result of needing to
// tile the CSR array to fit in the available sceratchpad space. However, the
// output will always be placed into the UMEM.
//
// Arguments:
//   dma_weights: The compressed data, accessed via DMA.
//   acp_weights: The compressed data, accessed via ACP.
//   cache_weights: The compressed data, accessed via HW cache.
//   cmp_col_offset: The offset (32-bit granularity) into the source compressed
//       data at which the column indices start.
//   cmp_row_offset: The offset (32-bit granularity) into the source compressed
//       data at which the row indices start.
//   dest_offset: The offset (32-bit granularity) into the destination buffer
//       from where the data should start getting written. This is required to
//       support tiled decompression.
//   compressed_size: The size (bytes) of the complete source CSR array.
//   decompressed_size: The size (bytes) that the array will take up once
//       decompressed.
//   input_in_spad0: Send the CSR data to spad0 if true.
//   copy_mechanism: Which mechanism to use for sending the input.
//   spad0: SPAD0 pointer.
//   spad1: SPAD1 pointer.
//   umem: UMEM pointer.
static void smiv_decompress_packed_csr_hw(packed_fp16* dma_weights,
                                          packed_fp16* acp_weights,
                                          packed_fp16* cache_weights,
                                          int cmp_col_offset,
                                          int cmp_row_offset,
                                          int dest_offset,
                                          dims_t* data_dims,
                                          size_t compressed_size,
                                          size_t decompressed_size,
                                          bool input_in_spad0,
                                          io_req_t copy_mechanism,
                                          bool use_pipelined_dma,
                                          float* spad0,
                                          float* spad1,
                                          float* umem) {
    PRINT_MSG("Decompressing CSR data!\n");
    // The umem must be zeroed first.
    int num_rows = decompressed_size / (VECTOR_SIZE * sizeof(float));
    int start_row = dest_offset / VECTOR_SIZE;
    VEC_ARRAY_1D(v8fp_t, _umem, umem);
    decompress_reset:
    for (int i = start_row; i < start_row + num_rows; i++)
        _umem[i] = (v8fp_t){ 0, 0, 0, 0, 0, 0, 0, 0 };

    if (copy_mechanism == IO_DMA) {
        if (input_in_spad0) {
            setReadyBits(spad0, compressed_size, 0);
            dma_load_wrapper(spad0,
                             (float*)dma_weights,
                             compressed_size,
                             use_pipelined_dma);
            decompress_packed_csr_data_smiv_fxp(
                    (packed_fp16*)spad0, cmp_col_offset, cmp_row_offset,
                    dest_offset, data_dims, umem);
        } else {
            setReadyBits(spad1, compressed_size, 0);
            dma_load_wrapper(spad1,
                             (float*)dma_weights,
                             compressed_size,
                             use_pipelined_dma);
            decompress_packed_csr_data_smiv_fxp(
                    (packed_fp16*)spad1, cmp_col_offset, cmp_row_offset,
                    dest_offset, data_dims, umem);
        }
    } else if (copy_mechanism == IO_ACP) {
        decompress_packed_csr_data_smiv_fxp(acp_weights, cmp_col_offset,
                                            cmp_row_offset, dest_offset,
                                            data_dims, umem);
    } else if (copy_mechanism == IO_CACHE) {
        decompress_packed_csr_data_smiv_fxp(cache_weights, cmp_col_offset,
                                            cmp_row_offset, dest_offset,
                                            data_dims, umem);
    }
}

void smiv_decompress_packed_csr_impl(layer_t* layer,
                                     int weights_list_idx,
                                     int start_row,
                                     bool input_in_spad0,
                                     smiv_global* g_smiv,
                                     device_t* device) {
    assert(layer->host_weights->len > weights_list_idx);
    packed_csr_array_t* src_csr =
            layer->host_weights->data[weights_list_idx].packed;
    begin_ignored_profiling(layer->num);
    csr_tile_list* tile_list = tile_packed_csr_array_t(
            src_csr, &layer->weights, start_row, g_smiv->kSpadSize);
    end_profiling();
    assert(tile_list->len > 0 && "CSR tile list cannot be empty!");
    csr_tile* curr_tile = tile_list->head;
    int dest_offset = 0;
    do {
        packed_csr_array_t* array = curr_tile->array;
        dims_t dims = (dims_t){ curr_tile->num_rows, layer->weights.cols,
                                layer->weights.height,
                                layer->weights.align_pad };
        assert(array->total_buf_size <= g_smiv->kSpadSize &&
               "CSR array size exceeds scratchpad capacity!");
        MAP_ARRAY_TO_ACCEL(g_smiv->kInnerProductHw,
                           get_host_weights_var_name(layer->weights_req),
                           array->vals, array->total_buf_size);
        INVOKE_KERNEL_PROF(g_smiv->kInnerProductHw,
                           layer->num,
                           smiv_decompress_packed_csr_hw,
                           array->vals,  // DMA
                           array->vals,  // ACP
                           array->vals,  // Cache
                           array->col_idx - array->vals,
                           array->row_idx - array->vals,
                           dest_offset,
                           &dims,
                           array->total_buf_size,
                           curr_tile->eff_total_bytes,
                           !input_in_spad0,  // Don't overwrite inputs!
                           device->cpu_default_offload,
                           device->use_pipelined_dma,
                           g_smiv->spad0,
                           g_smiv->spad1,
                           g_smiv->umem);
        dest_offset += (curr_tile->eff_total_bytes / sizeof(uint32_t));
        curr_tile = curr_tile->next_tile;
    } while (curr_tile);
    free_csr_tile_list(tile_list);
}
