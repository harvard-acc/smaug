#include <stdbool.h>

#include "core/nnet_fwd_defs.h"
#include "arch/common.h"
#include "arch/smv/common.h"
#include "arch/smiv/dispatch_utils.h"
#include "arch/smv/load_and_unpack_fp16_data.h"

#ifdef DMA_MODE
#include "gem5_harness.h"
#endif

void smv_dma_load_hw(float* local_dest,
                     int length,
                     float* umem,
                     float* spad0,
                     float* spad1,
                     float* host_src,
                     int host_offset,
                     int local_offset,
                     bool use_pipelined_dma,
                     bool fp16_input) {
    if (fp16_input) {
        length /= sizeof(float16);
        if (local_dest == umem) {
            dma_load_and_unpack_fp16(umem, (packed_fp16*)host_src, length,
                                local_offset, host_offset);
        } else if (local_dest == spad0) {
            dma_load_and_unpack_fp16(spad0, (packed_fp16*)host_src, length,
                                local_offset, host_offset);
        } else if (local_dest == spad1) {
            dma_load_and_unpack_fp16(spad1, (packed_fp16*)host_src, length,
                                local_offset, host_offset);
        }
    } else {
        if (local_dest == umem) {
            dmaLoad(&umem[local_offset], &host_src[host_offset], length);
        } else if (local_dest == spad0) {
            dmaLoad(&spad0[local_offset], &host_src[host_offset], length);
        } else if (local_dest == spad1) {
            dmaLoad(&spad1[local_offset], &host_src[host_offset], length);
        }
    }
}

void smv_dma_store_hw(float* host_dest,
                      int length,
                      float* umem,
                      float* spad0,
                      float* spad1,
                      float* local_src,
                      int host_offset,
                      int local_offset,
                      bool use_pipelined_dma,
                      bool fp16_input) {
    if (fp16_input) {
        length /= sizeof(float16);
        if (local_src == umem) {
            dma_pack_and_store_fp16((packed_fp16*)host_dest, umem, length,
                                    local_offset, host_offset);
        } else if (local_src == spad0) {
            dma_pack_and_store_fp16((packed_fp16*)host_dest, spad0, length,
                                    local_offset, host_offset);
        } else if (local_src == spad1) {
            dma_pack_and_store_fp16((packed_fp16*)host_dest,  spad1,length,
                                     local_offset, host_offset);
        }
    } else {
        if (local_src == umem) {
            dmaStore(&host_dest[host_offset], &umem[local_offset], length);
        } else if (local_src == spad0) {
            dmaStore(&host_dest[host_offset], &spad0[local_offset], length);
        } else if (local_src == spad1) {
            dmaStore(&host_dest[host_offset], &spad1[local_offset], length);
        }
    }
}

// Copy data over DMA.
//
// This invokes the DMA engine on gem5-aladdin so that we can do DMA operations
// without making it part of the main accelerator kernel. The accelerator IDs
// used in a call to this function is the same as the accelerator that needs to
// use the copied data.
//
// The source and destination pointers can be either accelerator or host; this
// is determined by whether this DMA operation is a DMA load or store.
//
// TODO: Currently we don't support pipelined DMA with this block because those
// function wrappers don't support an arbitrary source/dest offset.
void dma_copy_impl(float* dst_base_loc,
                   float* src_base_loc,
                   unsigned accel_id,
                   int layer_num,
                   smv_global* g_smv,
                   dma_options* options) {
    if (options->is_load) {
        MAP_ARRAY_TO_ACCEL(accel_id,
                           "host_src",
                           &src_base_loc[options->src_offset],
                           options->length);
        INVOKE_KERNEL_PROF(accel_id, layer_num, smv_dma_load_hw, dst_base_loc,
                           options->length, g_smv->umem, g_smv->spad0,
                           g_smv->spad1, src_base_loc, options->src_offset,
                           options->dst_offset, options->use_pipelined_dma,
                           options->fp16_input);
    } else {
        MAP_ARRAY_TO_ACCEL(accel_id,
                           "host_dest",
                           &dst_base_loc[options->dst_offset],
                           options->length);
        INVOKE_KERNEL_PROF(accel_id,
                           layer_num,
                           smv_dma_store_hw,
                           dst_base_loc,
                           options->length,
                           g_smv->umem,
                           g_smv->spad0,
                           g_smv->spad1,
                           src_base_loc,
                           options->src_offset,
                           options->dst_offset,
                           options->use_pipelined_dma,
                           options->fp16_input);
    }
}
