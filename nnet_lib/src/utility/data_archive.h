#ifndef _DATA_ARCHIVE_H_
#define _DATA_ARCHIVE_H_

#include "utility/data_archive_impl.h"

void verify_global_parameters(global_sec_header* header, network_t* network);
void save_global_parameters(FILE* fp, network_t* network);

void save_weights(FILE* fp, farray_t* weights, size_t num_weights);
void save_data(FILE* fp, farray_t* data, size_t num_values);
void save_labels(FILE* fp, iarray_t* labels, size_t num_labels);

void read_weights_from_file(const char* filename, farray_t* weights);
void read_data_from_file(const char* filename, farray_t* data);
void read_labels_from_file(const char* filename, iarray_t* labels);

void save_all_to_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels);

void read_all_from_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels);
#endif
