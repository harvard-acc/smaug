#include "nnet_fwd.h"

// The rectified linear activation function
void RELU(int* hid, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (hid[i] < 0.0) {
            hid[i] = 0.0;
        }
    }
}
/*
   int inner_prod(int* weights, int* hid, int hid_size){
   int j;
   int tmp;

   tmp=0;
   for(j = 0; j < hid_size; j++)
   tmp += (int)j;//weights[ j ] * hid[ j ];
   return tmp;
   }

 */

void matrix_vector_product_with_bias1(int* weights01, int*hid, int hid_size,
        int w_height, int *hid_temp) {
    int i, j, tmp_prod;
    for (i = 0; i < w_height; i++) {
        tmp_prod = 0.0;
        for (j = 0; j < hid_size; j++){
            tmp_prod += weights01[j] * hid[j];
        }
        *(hid_temp + i) = tmp_prod;
    }
}
void matrix_vector_product_with_bias2(int* weights12, int*hid, int hid_size,
        int w_height, int *hid_temp) {
    int i, j, tmp_prods;
    //int tmp_prods;
    for (i = 0; i < w_height; i++) {
        tmp_prods = 0.0;
        for (j = 0; j < hid_size; j++){
            tmp_prods += weights12[j] * hid[j];
        }
        *(hid_temp + i) = tmp_prods;
    }
}
void matrix_vector_product_with_bias3(int* weights23, int*hid, int hid_size,
        int w_height, int *hid_temp) {
    int i, j, tmp_prods;
    //int tmp_prods;
    for (i = 0; i < w_height; i++) {
        tmp_prods = 0.0;
        for (j = 0; j < hid_size; j++){
            tmp_prods += weights23[j] * hid[j];
        }
        *(hid_temp + i) = tmp_prods;
    }
}
void matrix_vector_product_with_bias4(int* weights34, int*hid, int hid_size,
        int w_height, int *hid_temp) {
    int i, j, tmp_prods;
    //int tmp_prods;
    for (i = 0; i < w_height; i++) {
        tmp_prods = 0.0;
        for (j = 0; j < hid_size; j++){
            tmp_prods += weights34[j] * hid[j];
        }
        *(hid_temp + i) = tmp_prods;
    }
}

//void nnet_fwd_relu(int* weights, int* num_units, int* hid, int* hid_temp) {
void nnet_fwd_relu(int* weights01, int* weights12, int* weights23, int* weights34,
        int* num_units, int* hid, int* hid_temp) {
    int i;
    int w_ind;

    matrix_vector_product_with_bias1(weights01, hid, num_units[0], num_units[1], hid_temp);
    RELU(hid, num_units[1]);
    matrix_vector_product_with_bias2(weights12, hid, num_units[1], num_units[2], hid_temp);
    RELU(hid, num_units[2]);
    matrix_vector_product_with_bias3(weights23, hid, num_units[2], num_units[3], hid_temp);
    RELU(hid, num_units[3]);
    matrix_vector_product_with_bias4(weights34, hid, num_units[3], num_units[4], hid_temp);
    //RELU(hid, num_units[4]);
    //
    //RELU(hid, num_units[1]);
    //w_ind = (num_units[0]+1)*num_units[1];
    //for (i = 1; i < NUM_LAYERS; i++)
    //{
    //   matrix_vector_product_with_bias(weights+w_ind, hid, num_units[i],
    //                                    num_units[i+1], hid_temp);
    //RELU(hid, num_units[i+1]);
    //    w_ind += (num_units[i]+1)*num_units[i+1];
    //}
    //matrix_vector_product_with_bias(weights+w_ind, hid, num_units[NUM_LAYERS],
    //                                num_units[NUM_LAYERS+1], hid_temp);
}

int randint() {
    return rand() / ((int)(RAND_MAX));
}

int main( int argc, const char* argv[] )
{
    srand(time(NULL));

    int* result;
    int i, j;
    int num_units[NUM_LAYERS+2];

    num_units[0] = INPUT_DIM;  // input dimensionality
    for (i = 1; i <= NUM_LAYERS; i++) {
        num_units[i] = NUM_HIDDEN_UNITS[i-1];
    }

    num_units[NUM_LAYERS+1] = NUM_CLASSES;    // number of classes
    int biggest_row = num_units[0]; // including input layer
    for (i = 1; i < NUM_LAYERS+2; i++) {
        if (num_units[i] > biggest_row) {
            biggest_row = num_units[i];
        }
    }

    int hid[biggest_row];
    int hid_temp[biggest_row];
    for (i = 0; i < INPUT_DIM; i++) {
        hid[i] = randint() - 0.5;
    }
    // Compute number of weights in total
    int w_size = 0;
    for (i = 0; i < NUM_LAYERS+1; i++)
        w_size += (num_units[i]+1)*num_units[i+1];

    // Randomly initialize weights
    int size1, size2, size3, size4;
    size1 = num_units[0] * num_units[1];
    size2 = num_units[1] * num_units[2];
    size3 = num_units[2] * num_units[3];
    size4 = num_units[3] * num_units[4];

    int weights01[size1];
    int weights12[size2];
    int weights23[size3];
    int weights34[size4];

    for (i = 0; i < size1; i++) {
        weights01[i] = (randint()-0.5);
    }
    for (i = 0; i < size2; i++) {
        weights12[i] = (randint()-0.5);
    }
    for (i = 0; i < size3; i++) {
        weights23[i] = (randint()-0.5);
    }
    for (i = 0; i < size4; i++) {
        weights34[i] = (randint()-0.5);
    }


    // Actual HW sims..
    nnet_fwd_relu(weights01, weights12, weights23, weights34, num_units, hid, hid_temp);
    //nnet_fwd_relu(weights, num_units, hid, hid_temp);
}
