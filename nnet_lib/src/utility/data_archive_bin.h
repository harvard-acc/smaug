#ifndef _DATA_ARCHIVE_BIN_H_
#define _DATA_ARCHIVE_BIN_H_

#include "utility/data_archive_common.h"

mmapped_file open_bin_data_file(const char* filename);
void close_bin_data_file(mmapped_file* file);

void save_global_parameters_to_bin_file(FILE* fp, network_t* network);
void save_labels_to_bin_file(FILE* fp, iarray_t* labels, size_t num_labels);
void save_inputs_to_bin_file(FILE* fp, farray_t* inputs, size_t num_values);
void save_weights_to_bin_file(FILE* fp, farray_t* weights, size_t num_weights);
global_sec_header read_global_header_from_bin_file(mmapped_file* file);
void read_weights_from_bin_file(mmapped_file* file, farray_t* weights);
void read_inputs_from_bin_file(mmapped_file* file, farray_t* data);
void read_labels_from_bin_file(mmapped_file* file, iarray_t* labels);

#endif
