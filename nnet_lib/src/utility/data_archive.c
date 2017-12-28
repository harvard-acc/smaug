// Save all weights, data, and labels to a file.
//
// We support both a text-based format and a binary file format. Text file
// names must end in .txt, and binary file names must end in .bin.
//
// Usage:
//
// To archive data, use the save_all_to_file() function. To read data, use the
// read_all_from_file() function. For both, pass a filename, all data buffer,
// and the network configuration. Based on the filename, the appropriate type
// of data format will be used.
//
// Weights and input data are floating point, while labels are integers.
//
// This archival functionality saves and restores raw buffer contents, without
// accounting for data alignment and zeropadding. If these are required, then
// the data must already be saved with all the appropriate alignment and
// padding in the archive! To help ensure that the data is appropriately
// formatted, there is a global section that contains metadata about the
// archive. Use the verify_global_parameters() function to ensure that all
// requirements match.

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"
#include "utility/data_archive_bin.h"
#include "utility/data_archive_txt.h"
#include "utility/data_archive_common.h"

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

bool is_txt_file(const char* filename) {
    unsigned size = strlen(filename);
    if (strncmp(&filename[size - 3], "txt", 3) == 0)
      return true;
    return false;
}

void save_all_to_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* inputs,
                      iarray_t* labels) {
    int input_rows = network->layers[0].inputs.rows;
    int input_cols = network->layers[0].inputs.cols;
    int input_height = network->layers[0].inputs.height;
    int input_align_pad = network->layers[0].inputs.align_pad;
    int input_dim = input_rows * (input_cols + input_align_pad) * input_height;

    if (is_txt_file(filename)) {
        printf("Saving data to text file %s...\n", filename);
        FILE* network_dump = fopen(filename, "w");
        save_global_parameters_to_txt_file(network_dump, network);
        save_weights_to_txt_file(network_dump, weights, weights->size);
        save_data_to_txt_file(network_dump, inputs, input_dim * NUM_TEST_CASES);
        save_labels_to_txt_file(network_dump, labels, labels->size);
        fclose(network_dump);
    } else {
        printf("Saving data to binary file %s...\n", filename);
        FILE* network_dump = fopen(filename, "w");
        save_global_parameters_to_bin_file(network_dump, network);
        save_weights_to_bin_file(network_dump, weights, weights->size);
        save_inputs_to_bin_file(
                network_dump, inputs, input_dim * NUM_TEST_CASES);
        save_labels_to_bin_file(network_dump, labels, labels->size);
        fclose(network_dump);
    }
}

void read_all_from_file(const char* filename,
                        network_t* network,
                        farray_t* weights,
                        farray_t* inputs,
                        iarray_t* labels) {
    if (is_txt_file(filename)) {
        printf("Reading data from text file %s...\n", filename);
        global_sec_header header = read_global_header_from_txt_file(filename);
        verify_global_parameters(&header, network);
        read_weights_from_txt_file(filename, weights);
        read_data_from_txt_file(filename, inputs);
        read_labels_from_txt_file(filename, labels);
        free_global_sec_header(&header);
    } else {
        printf("Reading data from binary file %s...\n", filename);
        mmapped_file file = open_bin_data_file(filename);
        global_sec_header global_header =
                read_global_header_from_bin_file(&file);
        verify_global_parameters(&global_header, network);

        read_weights_from_bin_file(&file, weights);
        read_inputs_from_bin_file(&file, inputs);
        read_labels_from_bin_file(&file, labels);
        free_global_sec_header(&global_header);

        close_bin_data_file(&file);
    }
#if DEBUG_LEVEL > 2
    int input_rows = network->layers[0].inputs.rows;
    int input_cols = network->layers[0].inputs.cols;
    int input_height = network->layers[0].inputs.height;
    int input_align_pad = network->layers[0].inputs.align_pad;
#endif
    PRINT_MSG("Input activations:\n");
    PRINT_DEBUG4D(
            inputs->d, input_rows, input_cols + input_align_pad, input_height);
}
