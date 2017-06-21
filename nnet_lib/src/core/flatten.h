#ifndef _FLATTEN_H_
#define _FLATTEN_H_

result_buf flatten_input(float* input,
                         layer_t* layers,
                         int lnum,
                         float* result);

void im2row(float* input,
            int input_rows,
            int input_cols,
            int input_height,
            int input_pad,
            float* result);

#endif
