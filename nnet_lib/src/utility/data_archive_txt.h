#ifndef _DATA_ARCHIVE_TXT_H_
#define _DATA_ARCHIVE_TXT_H_

#include "utility/data_archive_common.h"

void save_global_parameters_to_txt_file(FILE* fp, network_t* network);
void save_labels_to_txt_file(FILE* fp, iarray_t* labels, size_t num_labels);
void save_data_to_txt_file(FILE* fp, farray_t* data, size_t num_values);
void save_weights_to_txt_file(FILE* fp, farray_t* weights, size_t num_weights);
void save_compress_type_to_txt_file(FILE* fp,
                                    iarray_t* compress_type,
                                    size_t num_layers);

global_sec_header read_global_header_from_txt_file(const char* filename);
void read_weights_from_txt_file(const char* filename, farray_t** weights);
void read_data_from_txt_file(const char* filename, farray_t** data);
void read_labels_from_txt_file(const char* filename, iarray_t* labels);
void read_compress_type_from_txt_file(const char* filename,
                                      iarray_t* compress_type);

#endif
