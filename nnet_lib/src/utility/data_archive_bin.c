// Binary file data archive format.
//
// Global section format:
// GLOBAL
// [header-size]
// [header-payload]
//
// Section format:
// [SECTION_NAME]
// [header-size]
// [section-header-payload]
// [data-payload]
//
// The header will indicate the size of the data payload as well as other
// metadata about the data payload.
//
// When reading the file, it is mapped into memory and then manipulated
// directly with pointers to avoid expensive file I/O system calls. The
// mmapped_file struct is used to store the file's file descriptor, the current
// position (as a pointer into the mmapped region), and the total size of the
// file. All functions that read from the mmapped region update the current
// position, just like usual fread()/fwrite()/fseek() functions.

#include <assert.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"
#include "utility/data_archive_bin.h"
#include "utility/data_archive_common.h"

static void* find_section_header(mmapped_file* file, const char* section_name) {
    unsigned section_name_len = strlen(section_name);
    unsigned match_index = 0;
    unsigned pos = 0;
    const char* data = (const char*) file->addr;
    while (match_index < section_name_len && pos < file->file_size) {
        if (data[pos] == section_name[match_index]) {
            match_index++;
        } else {
            match_index = 0;
        }
        pos++;
    }
    if (match_index != section_name_len || pos >= file->file_size)
        FATAL_MSG("Unable to find section %s!\n", section_name);

    // Otherwise, return the file position pointer pointed to the address right
    // after the section name.
    return (void*)(data + pos);
}

static void read_section_header(void* header,
                                void** section_start,
                                unsigned expected_header_size) {
    unsigned header_size = 0;
    memcpy(&header_size, *section_start, sizeof(unsigned));
    // Sanity check on the header size.
    if (header_size != expected_header_size)
        FATAL_MSG("Header size %d bytes, expected header size of %d bytes.\n",
                  header_size, expected_header_size);
    // Move to the start of the header section.
    *section_start = (char*)(*section_start) + sizeof(unsigned);
    memcpy(header, *section_start, header_size);

    // Move to the start of the data payload.
    *section_start = (char*)(*section_start) + header_size;
}

static void save_global_sec_header_to_bin_file(FILE* fp,
                                               global_sec_header* header) {
    static const char* section_name = "GLOBAL";
    unsigned section_name_len = strlen(section_name);
    const unsigned header_size = sizeof(global_sec_header);
    assert(header_size > 0);
    assert(section_name_len > 0);

    fwrite(section_name, sizeof(char), strlen(section_name), fp);
    fwrite(&header_size, sizeof(unsigned), 1, fp);
    fwrite(header, header_size, 1, fp);
}

static void save_data_to_bin_file(FILE* fp,
                                  data_sec_header* header,
                                  void* data,
                                  unsigned header_size,
                                  unsigned data_size,
                                  const char* section_name) {
    unsigned section_name_len = strlen(section_name);
    assert(header_size > 0);
    assert(section_name_len > 0);

    fwrite(section_name, sizeof(char), strlen(section_name), fp);
    fwrite(&header_size, sizeof(unsigned), 1, fp);
    fwrite(header, sizeof(data_sec_header), 1, fp);
    if (data_size > 0 && data)
        fwrite(data, sizeof(float), data_size, fp);
}

static void read_array_from_bin_file(mmapped_file* file,
                                     void* data_buf,
                                     unsigned max_size,
                                     unsigned elem_size,
                                     const char* section_name) {
    void* section_start = find_section_header(file, section_name);
    data_sec_header header;
    unsigned header_size = sizeof(data_sec_header);
    read_section_header((void*)&header, &section_start, header_size);
    if (header.num_elems > max_size) {
        FATAL_MSG("The amount of data found in section %s exceeds the size of "
                  "the array allocated to store it!\n", section_name);
    } else if (header.num_elems > 0) {
        memcpy(data_buf, section_start, header.num_elems * elem_size);
    }
}

static void read_farray_from_bin_file(mmapped_file* file,
                                      farray_t* data,
                                      const char* section_name) {
    read_array_from_bin_file(
            file, (void*)data->d, data->size, sizeof(float), section_name);
}

static void read_iarray_from_bin_file(mmapped_file* file,
                                      iarray_t* data,
                                      const char* section_name) {
    read_array_from_bin_file(
            file, (void*)data->d, data->size, sizeof(int), section_name);
}

mmapped_file open_bin_data_file(const char* filename) {
    struct stat st;
    stat(filename, &st);
    size_t size = st.st_size;
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Unable to open file");
        exit(1);
    }

    void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (!addr) {
        close(fd);
        perror("Unable to mmap the file into memory");
        exit(1);
    }

    mmapped_file file = { addr, fd, size };
    return file;
}

void close_bin_data_file(mmapped_file* file) {
    munmap(file->addr, file->file_size);
    close(file->fd);
}

void save_global_parameters_to_bin_file(FILE* fp, network_t* network) {
    // We can't store the architecture string with ARCH_STR directly since
    // that's a pointer to memory that would not be valid across executions.
    // Instead, fix this up when deserializing the binary file.
    global_sec_header header = { (Architecture) ARCHITECTURE, NULL,
                                 network->depth, DATA_ALIGNMENT };
    save_global_sec_header_to_bin_file(fp, &header);
}

void save_weights_to_bin_file(FILE* fp, farray_t* weights, size_t num_weights) {
    data_sec_header header = { num_weights, SAVE_DATA_FLOAT };
    save_data_to_bin_file(fp,
                          &header,
                          (void*)weights->d,
                          sizeof(data_sec_header),
                          num_weights,
                          "WEIGHTS");
}

void save_inputs_to_bin_file(FILE* fp, farray_t* inputs, size_t num_values) {
    data_sec_header header = { num_values, SAVE_DATA_FLOAT };
    save_data_to_bin_file(fp,
                          &header,
                          (void*)inputs->d,
                          sizeof(data_sec_header),
                          num_values,
                          "INPUTS");
}

void save_labels_to_bin_file(FILE* fp, iarray_t* labels, size_t num_labels) {
    data_sec_header header = { num_labels, SAVE_DATA_INT };
    save_data_to_bin_file(fp,
                          &header,
                          (void*)labels->d,
                          sizeof(data_sec_header),
                          num_labels,
                          "LABELS");
}

void save_compress_type_to_bin_file(FILE* fp,
                                    iarray_t* compress_types,
                                    size_t num_layers) {
    data_sec_header header = { num_layers, SAVE_DATA_INT };
    save_data_to_bin_file(fp,
                          &header,
                          (void*)compress_types->d,
                          sizeof(data_sec_header),
                          num_layers,
                          "COMPRESSTYPE");
}

global_sec_header read_global_header_from_bin_file(mmapped_file* file) {
    global_sec_header global_header;
    void* section = find_section_header(file, "GLOBAL");
    read_section_header(
            (void*)&global_header, &section, sizeof(global_sec_header));
    global_header.arch_str = arch2str(global_header.arch);
    return global_header;
}

void read_weights_from_bin_file(mmapped_file* file, farray_t* weights) {
    read_farray_from_bin_file(file, weights, "WEIGHTS");
}

void read_inputs_from_bin_file(mmapped_file* file, farray_t* inputs) {
    read_farray_from_bin_file(file, inputs, "INPUTS");
}

void read_labels_from_bin_file(mmapped_file* file, iarray_t* labels) {
    read_iarray_from_bin_file(file, labels, "LABELS");
}

void read_compress_type_from_bin_file(mmapped_file* file,
                                      iarray_t* compress_type) {
    read_iarray_from_bin_file(file, compress_type, "COMPRESSTYPE");
}
