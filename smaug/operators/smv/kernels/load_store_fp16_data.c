#include "smaug/operators/smv/kernels/load_store_fp16_data.h"

#ifdef __cplusplus
extern "C" {
#endif

void host_load_fp16(float* local_data,
                    float16* remote_data,
                    int num_elems,
                    int local_offset,
                    int remote_offset) {
    VEC_ARRAY_1D(v8ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes =
            next_multiple(num_elems * sizeof(float16), CACHELINE_SIZE);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    host_fp16_to_fp32:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        int curr_offset = (i * page_size * 2) / sizeof(float);
        hostLoad(local_data + local_offset + curr_offset,
                 remote_data + remote_offset + curr_offset,
                 transfer_size);

        // This loads N bytes of FP16 data into local_data. We now expand
        // N bytes of half precision to 2*N bytes of single precision, in
        // place, 32 bytes at a time. In order to do this without overwriting
        // the data we're trying to unpack, we need to start from the back.
        int num_vectors =
                FRAC_CEIL(transfer_size * 2, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_offset) / VECTOR_SIZE;
        vector_fp16_to_fp32:
        for (int v = num_vectors - 1; v >= 0; v--) {
            v8ph_t fp16_data = _local_data_hp[page_offset_vec * 2 + v];
            v8fp_t fp32_data = _CVT_PH_PS_256(fp16_data);
            _local_data_sp[page_offset_vec + v] = fp32_data;
        }
        num_bytes_remaining -= transfer_size;
    }
}

void host_store_fp16(float* local_data,
                     float16* remote_data,
                     int num_elems,
                     int local_offset,
                     int remote_offset) {
    VEC_ARRAY_1D(v8ph_t, _local_data_hp, local_data);
    VEC_ARRAY_1D(v8fp_t, _local_data_sp, local_data);
    const int page_size = (1 << LOG_PAGE_SIZE);
    const int max_transfer_size = page_size;
    const int total_bytes =
            next_multiple(num_elems * sizeof(float16), CACHELINE_SIZE);
    int num_xfers = FRAC_CEIL(total_bytes, max_transfer_size);
    int num_bytes_remaining = total_bytes;
    host_fp32_to_fp16:
    for (int i = 0; i < num_xfers; i++) {
        int transfer_size = min2(num_bytes_remaining, max_transfer_size);
        // The effective transfer size is the size in terms of FP32.
        int eff_transfer_size = transfer_size * 2;
        int curr_offset = (i * 2 * page_size) / sizeof(float);

        int num_vectors =
                FRAC_CEIL(eff_transfer_size, VECTOR_SIZE * sizeof(float));
        int page_offset_vec = (local_offset + curr_offset) / VECTOR_SIZE;
        vector_fp32_to_fp16:
        for (int v = 0; v < num_vectors; v++){
            v8fp_t fp32_data = _local_data_sp[page_offset_vec + v];
            v8ph_t fp16_data = _CVT_PS_PH_256(fp32_data, 0);
            _local_data_hp[page_offset_vec * 2 + v] = fp16_data;
        }

        hostStore(remote_data + remote_offset + curr_offset,
                  local_data + local_offset + curr_offset,
                  transfer_size);

        num_bytes_remaining -= transfer_size;
    }
}

#ifdef __cplusplus
}  // extern "C"
#endif
