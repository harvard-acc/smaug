#ifndef _DATA_ARCHIVE_IMPL_H_
#define _DATA_ARCHIVE_IMPL_H_

#define FATAL_MSG(args...)                                                     \
    do {                                                                       \
        fprintf(stderr, "[DATA FILE ERROR]: " args);                           \
        exit(1);                                                               \
    } while (0);

#endif

typedef enum _datatype {
  SAVE_DATA_INT,
  SAVE_DATA_FLOAT,
  NUM_DATATYPES,
} datatype;

typedef enum _Architecture {
    Arch_Monolithic = MONOLITHIC,
    Arch_Composable = COMPOSABLE,
    Arch_SMIV = SMIV,
    Arch_Eigen = EIGEN,
    Arch_END
} Architecture;

typedef struct _data_sec_header {
    unsigned num_elems;
    datatype type;
} data_sec_header;

typedef struct _global_sec_header {
    Architecture arch;
    char* arch_str;
    int num_layers;
    int data_alignment;
} global_sec_header;

//=------------- TEXT FILE FORMAT -----------------=//

/*
extern const char* kTxtGlobalHeader;
extern const char* kTxtGlobalFooter;
extern const char* kTxtWeightsHeader;
extern const char* kTxtWeightsFooter;
extern const char* kTxtDataHeader;
extern const char* kTxtDataFooter;
extern const char* kTxtLabelsHeader;
extern const char* kTxtLabelsFooter;
*/

/*
void read_int_data_from_txt_file(const char* filename,
                                 iarray_t* data,
                                 const char* section_header);

void read_fp_data_from_txt_file(const char* filename,
                                farray_t* data,
                                const char* section_header);
void save_farray_to_txt_file(FILE* fp, farray_t data, unsigned size);
void save_iarray_to_txt_file(FILE* fp, iarray_t data, unsigned size);
*/
void save_labels_to_txt_file(FILE* fp, iarray_t* labels, size_t num_labels);
void save_data_to_txt_file(FILE* fp, farray_t* data, size_t num_values);
void save_weights_to_txt_file(FILE* fp, farray_t* weights, size_t num_weights);
void read_weights_from_txt_file(const char* filename, farray_t* weights);
void read_data_from_txt_file(const char* filename, farray_t* data);
void read_labels_from_txt_file(const char* filename, iarray_t* labels);
void save_global_parameters_to_txt_file(FILE* fp, network_t* network);
global_sec_header read_global_header_from_txt_file(FILE* fp);

//=------------- BINARY FILE FORMAT ---------------=//

/*
void read_int_data_from_bin_file(const char* filename, iarray_t* data);
void read_fp_data_from_bin_file(const char* filename, farray_t* data);
void save_farray_to_bin_file(FILE* fp, farray_t data, unsigned size);
void save_iarray_to_bin_file(FILE* fp, iarray_t data, unsigned size);
*/
