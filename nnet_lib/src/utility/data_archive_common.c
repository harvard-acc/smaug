#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utility/data_archive_common.h"

// Converts a string describing an architecture into the appropriate
// Architecture enum.
Architecture str2arch(const char* arch_str, size_t len) {
    if (strncmp(arch_str, "MONOLITHIC", len) == 0)
        return Arch_Monolithic;
    if (strncmp(arch_str, "COMPOSABLE", len) == 0)
        return Arch_Composable;
    if (strncmp(arch_str, "SMIV", len) == 0)
        return Arch_SMIV;
    if (strncmp(arch_str, "EIGEN", len) == 0)
        return Arch_Eigen;
    return Arch_END;
}

// Converts an Architecture enum into a string.
//
// This string's storage is heap allocated. The caller is responsible for
// freeing this, typically by calling free_global_sec_header().
char* arch2str(Architecture arch) {
    char* arch_str = (char*)malloc(11);
    switch (arch) {
      case Arch_Monolithic:
        snprintf(arch_str, 11, "%s", "MONOLITHIC");
        break;
      case Arch_Composable:
        snprintf(arch_str, 11, "%s", "COMPOSABLE");
        break;
      case Arch_SMIV:
        snprintf(arch_str, 5, "%s", "SMIV");
        break;
      case Arch_Eigen:
        snprintf(arch_str, 6, "%s", "EIGEN");
        break;
      default:
        snprintf(arch_str, 8, "%s", "UNKNOWN");
        break;
    }
    return arch_str;
}

// Free dynamic memory used by the global section header.
void free_global_sec_header(global_sec_header* header) {
    free(header->arch_str);
}
