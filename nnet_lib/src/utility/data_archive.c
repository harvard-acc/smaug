// Save all weights, data, and labels to a file in a simple format.
//
// The file is comprised of sections, with a section header indicating the size
// of the data payload and a section footer denoting the end of the section.
//
// Section format.
// ===[SECTION_NAME] BEGIN===
// # NUM_ELEMS n
// # TYPE float/int
// 1,2,3,4,.... (total of n comma separated values)
// ===[SECTION_NAME] END===
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
// NOTE: This archival functionality saves and restores raw buffer contents,
// without accounting for data alignment and zeropadding. If these are
// required, then the data must already be saved with all the appropriate
// alignment and padding in the archive!

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/nnet_fwd_defs.h"
#include "utility/utility.h"
#include "utility/data_archive.h"

#define FATAL_MSG(args...)                                                     \
    do {                                                                       \
        fprintf(stderr, "[DATA FILE ERROR]: " args);                           \
        exit(1);                                                               \
    } while (0);

typedef enum _datatype {
  SAVE_DATA_INT,
  SAVE_DATA_FLOAT,
  NUM_DATATYPES,
} datatype;

typedef struct _file_header {
    unsigned num_elems;
    datatype type;
} file_header;

static const char* kWeightsHeader = "===WEIGHTS BEGIN===";
static const char* kWeightsFooter = "===WEIGHTS END===";
static const char* kDataHeader = "===DATA BEGIN===";
static const char* kDataFooter = "===DATA END===";
static const char* kLabelsHeader = "===LABELS BEGIN===";
static const char* kLabelsFooter = "===LABELS END===";

//=-------------- INTERNAL FUNCTIONS ---------------=//

static void save_farray_to_file(FILE* fp, farray_t data, unsigned size) {
    fprintf(fp, "# NUM_ELEMS %d\n# TYPE float\n", size);
    for (unsigned i = 0; i < size; i++) {
        fprintf(fp, "%2.8f,", data.d[i]);
    }
    fprintf(fp, "\n");
}

static void save_iarray_to_file(FILE* fp, iarray_t data, unsigned size) {
    fprintf(fp, "# NUM_ELEMS %d\n# TYPE int\n", size);
    for (unsigned i = 0; i < size; i++) {
        fprintf(fp, "%d,", data.d[i]);
    }
    fprintf(fp, "\n");
}

static bool find_section_start(FILE* fp, const char* section_header) {
    char* line = NULL;
    size_t line_len = 0;
    bool found_section = false;
    const size_t header_len = strlen(section_header);
    while (getline(&line, &line_len, fp) != -1) {
        if (strncmp(line, section_header, header_len) == 0) {
            found_section = true;
            break;
        }
    }
    if (line)
        free(line);

    return found_section;
}

static file_header read_file_header(FILE* fp, const char* section_header) {
    if (fp == NULL)
        FATAL_MSG("Can't open data file!\n");

    if (!find_section_start(fp, section_header)) {
        fclose(fp);
        FATAL_MSG("Section header was not found in the file!\n");
    }

    file_header header;
    int num_elems;
    char data_type[6];
    int ret = fscanf(fp, "# NUM_ELEMS %d\n", &num_elems);
    if (ret != 1)
        FATAL_MSG("Corrupted header! NUM_ELEMS not found.\n");
    if (num_elems < 0)
        FATAL_MSG("NUM_ELEMS cannot be negative!\n");
    header.num_elems = num_elems;

    ret = fscanf(fp, "# TYPE %6s\n", &data_type[0]);
    if (ret != 1)
        FATAL_MSG("Corrupted header! Datatype not found.\n");

    if (strncmp(data_type, "float", 6) == 0) {
      header.type = SAVE_DATA_FLOAT;
    } else if (strncmp(data_type, "int", 4) == 0) {
      header.type = SAVE_DATA_INT;
    } else {
        FATAL_MSG("Invalid datatype %s found.\n", data_type);
    }
    return header;
}

static void read_fp_data_from_file(const char* filename,
                                   farray_t* data,
                                   const char* section_header) {
    FILE* fp = fopen(filename, "r");
    file_header header = read_file_header(fp, section_header);
    if (header.num_elems > data->size) {
        FATAL_MSG("This file section contains more data than can be "
                  "stored in the provided array!\n");
    } else if (header.type != SAVE_DATA_FLOAT) {
        FATAL_MSG("Expected datatype float, got datatype int!");
    }

    float read_float;
    for (unsigned i = 0; i < header.num_elems; i++) {
        int ret_f_scanf = fscanf(fp, "%f,", &read_float);
        if (ret_f_scanf != 1) {
            fclose(fp);
            FATAL_MSG("The number of values expected to be read "
                      "exceeded the number of values found!\n");
        }
        data->d[i] = conv_float2fixed(read_float);
    }

    fclose(fp);
}

static void read_int_data_from_file(const char* filename,
                                    iarray_t* data,
                                    const char* section_header) {
    FILE* fp = fopen(filename, "r");
    file_header header = read_file_header(fp, section_header);
    if (header.num_elems > data->size) {
        FATAL_MSG("This file section contains more data than can be "
                  "stored in the provided array!\n");
    } else if (header.type != SAVE_DATA_INT) {
        FATAL_MSG("Expected datatype int, got datatype float!");
    }

    int read_int;
    for (unsigned i = 0; i < header.num_elems; i++) {
        int ret_f_scanf = fscanf(fp, "%d,", &read_int);
        if (ret_f_scanf != 1) {
            fclose(fp);
            FATAL_MSG("The number of data expected to be read "
                      "exceeded the number of data found!\n");
        }
        data->d[i] = read_int;
    }

    fclose(fp);
}

//=-------------- EXTERNAL API ---------------=//

void save_weights(FILE* fp, farray_t weights, size_t num_weights) {
    fprintf(fp, "%s\n", kWeightsHeader);
    save_farray_to_file(fp, weights, num_weights);
    fprintf(fp, "%s\n", kWeightsFooter);
}

void save_data(FILE* fp, farray_t data, size_t num_values) {
    fprintf(fp, "%s\n", kDataHeader);
    save_farray_to_file(fp, data, num_values);
    fprintf(fp, "%s\n", kDataFooter);
}

void save_labels(FILE* fp, iarray_t labels, size_t num_labels) {
    fprintf(fp, "%s\n", kLabelsHeader);
    save_iarray_to_file(fp, labels, num_labels);
    fprintf(fp, "%s\n", kLabelsFooter);
}

void read_weights_from_file(const char* filename, farray_t* weights) {
    printf("Reading weights from %s...\n", filename);
    read_fp_data_from_file(filename, weights, kWeightsHeader);
}

void read_data_from_file(const char* filename, farray_t* data) {
    printf("Reading input data from %s...\n", filename);
    read_fp_data_from_file(filename, data, kDataHeader);
}

void read_labels_from_file(const char* filename, iarray_t* labels) {
    printf("Reading output labels from %s...\n", filename);
    read_int_data_from_file(filename, labels, kLabelsHeader);
}
