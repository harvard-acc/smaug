#ifndef _DATA_ARCHIVE_H_
#define _DATA_ARCHIVE_H_

#include "utility/data_archive_common.h"

void verify_global_parameters(global_sec_header* header, network_t* network);

void save_all_to_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels,
                      iarray_t* compress_type);

void read_all_from_file(const char* filename,
                      network_t* network,
                      farray_t* weights,
                      farray_t* data,
                      iarray_t* labels,
                      iarray_t* compress_type);
#endif
