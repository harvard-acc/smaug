#ifndef _DATA_ARCHIVE_H_
#define _DATA_ARCHIVE_H_

void verify_global_parameters(const char* filename, network_t* network);
void save_global_parameters(FILE* fp, network_t* network);

void save_weights(FILE* fp, farray_t* weights, size_t num_weights);
void save_data(FILE* fp, farray_t* data, size_t num_values);
void save_labels(FILE* fp, iarray_t* labels, size_t num_labels);

void read_weights_from_file(const char* filename, farray_t* weights);
void read_data_from_file(const char* filename, farray_t* data);
void read_labels_from_file(const char* filename, iarray_t* labels);

#endif
