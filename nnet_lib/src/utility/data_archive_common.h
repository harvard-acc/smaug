#ifndef _DATA_ARCHIVE_COMMON_H_
#define _DATA_ARCHIVE_COMMON_H_

#include "core/nnet_fwd_defs.h"

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

typedef enum _Architecture {
    Arch_Monolithic = MONOLITHIC,
    Arch_Composable = COMPOSABLE,
    Arch_SMIV = SMIV,
    Arch_Eigen = EIGEN,
    Arch_END
} Architecture;

typedef struct _data_sec_header {
    size_t num_elems;
    datatype type;
} data_sec_header;

typedef struct _global_sec_header {
    Architecture arch;
    char* arch_str;
    int num_layers;
    int data_alignment;
} global_sec_header;

typedef struct _mmapped_file {
    void* addr;
    int fd;
    size_t file_size;
} mmapped_file;

Architecture str2arch(const char* arch_str, size_t len);
char* arch2str(Architecture arch);

void free_global_sec_header(global_sec_header* header);

#endif
