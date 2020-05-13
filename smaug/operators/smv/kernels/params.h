#ifndef _OPERATORS_SMV_KERNELS_PARAMS_H_
#define _OPERATORS_SMV_KERNELS_PARAMS_H_

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 8
#elif VECTOR_SIZE != 8
#error "Existing VECTOR_SIZE is incompatible with SMV!"
#endif

#define NUM_MACC_INSTS 4
#define NUM_PE_INSTS 8

#define DATA_PE_ALIGNMENT (NUM_MACC_INSTS)*(VECTOR_SIZE)

#endif
