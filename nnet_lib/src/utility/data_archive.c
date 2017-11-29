// Save all weights, data, and labels to a file in a simple format.
//
// Usage:
//
// To archive data, use the save_weights(), save_data(), and save_labels()
// functions based on the type of data.  Pass a FILE pointer, the data buffer,
// and the number of elements to archive. These functions must be passed the
// number of elements to archive because the arrays that store the data may be
// larger than the data we need to archive.  For example, the arrays that store
// the inputs are also used to store input/output activations of hidden layers,
// which could be larger than the input size.
//
// To read a data archive, use the read_*_from_file functions based on the type
// of data, passing the filename and a preallocated buffer. If the buffer size
// is smaller than the number of elements in the section, this function will
// fail with an error message.
//
// Weights and input data are floating point, while labels are integers.
//
// This archival functionality saves and restores raw buffer contents, without
// accounting for data alignment and zeropadding. If these are required, then
// the data must already be saved with all the appropriate alignment and
// padding in the archive! To help ensure that the data is appropriately
// formatted, there is a global section that contains metadata about the
// archive. Use the verify_global_parameters() and save_global_parameters()
// functions to manipulate this section.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"
#include "utility/data_archive_impl.h"

void verify_global_parameters(global_sec_header* header, network_t* network) {
    if (header->arch != ARCHITECTURE)
        FATAL_MSG("The architecture used to generate this archive is not the "
                  "same as the current architecture! Got %s, expected %s.\n",
                  header->arch_str, ARCH_STR)

    if (header->num_layers != network->depth)
        FATAL_MSG("Number of layers in this archive does not match the current "
                  "network's topology! Found %d layers, expected %d instead.\n",
                  header->num_layers, network->depth);

    if (header->data_alignment != DATA_ALIGNMENT)
        FATAL_MSG("The data alignment of this archive does not match the "
                  "current architecture's data alignment requirements! Found "
                  "%d, expected %d instead.\n",
                  header->data_alignment, DATA_ALIGNMENT);
}

void save_global_parameters(FILE* fp, network_t* network) {
    save_global_parameters_to_txt_file(fp, network);
}

void save_weights(FILE* fp, farray_t* weights, size_t num_weights) {
    save_weights_to_txt_file(fp, weights, num_weights);
}

void save_data(FILE* fp, farray_t* data, size_t num_values) {
    save_data_to_txt_file(fp, data, num_values);
}

void save_labels(FILE* fp, iarray_t* labels, size_t num_labels) {
    save_labels_to_txt_file(fp, labels, num_labels);
}

void read_weights_from_file(const char* filename, farray_t* weights) {
    read_weights_from_txt_file(filename, weights);
}

void read_data_from_file(const char* filename, farray_t* data) {
    read_data_from_txt_file(filename, data);
}

void read_labels_from_file(const char* filename, iarray_t* labels) {
    read_labels_from_txt_file(filename, labels);
}

bool is_txt_file(const char* filename) {
    unsigned size = strlen(filename);
    if (strncmp(&filename[size - 4], "txt", 3) == 0)
      return true;
    return false;
}

void save_all_to_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels) {
    if (is_txt_file(filename)) {
        FILE* network_dump = fopen(filename, "w");
        save_global_parameters(network_dump, network);
        save_weights(network_dump, weights, weights->size);
        save_data(network_dump, data, INPUT_DIM * NUM_TEST_CASES);
        save_labels(network_dump, labels, labels->size);
        fclose(network_dump);
    }
}

void read_all_from_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels) {
    if (is_txt_file(filename)) {
        global_sec_header header = read_global_header_from_txt_file(filename);
        verify_global_parameters(&header, network);
        read_weights_from_file(filename, weights);
        read_data_from_file(filename, data);
        read_labels_from_file(filename, labels);
        free(header.arch_str);
    }
}
