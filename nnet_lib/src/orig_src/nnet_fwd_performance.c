#include "nnet_fwd_performance.h"
// All the memory used in nnet:
// name,type,size/value,
// data,float *,NUM_TEST_CASES*INPUT_DIM
// weights,float *,INPUT_DIM * NUM_UNITS_1 + NUM_UNITS_1 * NUM_UNITS_2 + NUM_UNITS_2 * NUM_CLASSES
// num_test_cases,int,NUM_TEST_CASES
// num_layers,int,NUM_LAYERS
// num_units,int *,NUM_LAYERS + 2
// activation_fun,int,ACTIVATION_FUN
// num_rows,int *,NUM_LAYERS + 1
// num_colums,int *,NUM_LAYERS + 1
// hid,float *,NUM_TEST_CASES * BIGGEST_ROW
// hid_temp,float *,NUM_TEST_CASES * BIGGEST_ROW


/*float convert_float2fixed ( unsigned num_of_int_bits, unsigned num_of_frac_bits, float input)*/
#ifdef BITWIDTH_QUANTIZATION
float conv_float2fixed ( float input)
{
    // return input;
    int sign = 1;
    if (input < 0) {
        sign = -1;
    }
    long long int long_1 = 1;

    return sign * ((float) ((long long int) (fabs(input) * ( long_1 << NUM_OF_FRAC_BITS)) & ((long_1 << (NUM_OF_INT_BITS+ NUM_OF_FRAC_BITS) )  - 1) )) / ( long_1 << NUM_OF_FRAC_BITS);

    // printf("Original: %f\n", x);
    // x = fabs(x);
    // printf("Abs: %f\n", x);
    // float y = (x * ( 1 << NUM_OF_FRAC_BITS));
    // printf("Shited left: %f\n", y);
    // unsigned long long int z = (unsigned long long int) y;
    // printf("Fractional bits truncated to proper length: %llu\n", z);
    // unsigned long long int bitmask = (((unsigned long long int) 1 << (NUM_OF_INT_BITS+ NUM_OF_FRAC_BITS) )  - 1);
    // z = z & (bitmask);
    // printf("Int bits truncated from left: %llu\n", z);
    // y = (float) z;
    // printf("Cast to float: %f\n", y);
    // y = y / ( 1 << NUM_OF_FRAC_BITS);
    // printf("Final unsigned value: %f\n", y);
    // y = y * sign;
    // printf("Final value: %f\n", y);
}
#endif



// The rectified linear activation function
// ** this function is in-place (modifies a) **
void RELU(float* a, int num_units) {
    int i;
    for (i = 0; i < num_units; i++) {
        if (a[i] < 0.0) {
            a[i] = 0.0;
        }
    }
}

// The logistic activation function
// ** this function is in-place (modifies a) **
void sigmoid(float* a, int num_units) {
    int i;
    float value;
    for (i = 0; i < num_units; i++) {
        value = 1.0 / (1.0 + exp(-a[i]));
        a[i] = conv_float2fixed(value);
    }
}

// The logistic activation function, implemented with a lookup table
// and linear interpolation
// ** this function is in-place (modifies a) **
void sigmoid_lookup(float* a, int num_units, float* sigmoid_table) {
    int i, ind;
    float temp, delta_x;
    float SIG_RANGE = SIG_MAX - SIG_MIN;
    for (i = 0; i < num_units; i++) {
        if (a[i] < SIG_MIN) {
            a[i] = 0.0; // do I need to convert these?? I guess not?
        }
        else if (a[i] >= SIG_MAX) {
            a[i] = 1.0;
        }
        else {
        	temp = conv_float2fixed(((a[i] - SIG_MIN) / SIG_RANGE) * ((1<<LG_SIGMOID_COARSENESS)-1.0));
            ind = (int) temp;
            delta_x = conv_float2fixed(temp-ind); // in [0,1]
            // printf("%f   %f\n", delta_x, sigmoid_table[ind]);
            a[i] = conv_float2fixed(sigmoid_table[ind] * (1.0-delta_x) + sigmoid_table[ind+1] * delta_x);
        }
    }
}


// Convert from (r,c) subscript to linear index (row major)
//int sub2ind(int r, int c, int n_columns) {
//    return r*n_columns + c;
//}

#define sub2ind(r, c, n_columns) r*n_columns+c

// Convert matrix n, element (r,c) into linear index
// Used for indexing a doubly flattened collection of matrices
// int sub2ind2(int r, int c, int n, int* n_rows, int* n_columns) {
//     int ind = 0;
//     int i;
//     for (i = 0; i < n; i++) {
//         ind += n_rows[i]*n_columns[i];
//     }
//     return ind + sub2ind(r, c, n_columns[n]);
// }

// Grab matrix n out of the doubly flattened w
// (w is a flattened collection of matrices, each flattened)
float* grab_matrix(float* w, int n, int* n_rows, int*n_columns) {
    int ind = 0;
    int i;
    for (i = 0; i < n; i++) {
        ind += n_rows[i]*n_columns[i];
    }
    return w + ind;
}

void clear_matrix (float *input, int size)
{
  int i;
  for ( i = 0; i < size; i++)
    input[i] = 0.0;
}
void copy_matrix (float *input, float *output, int size)
{
  int i;
  for( i = 0; i < size; i++)
    output[i] = input[i];
}

int arg_max (float *input, int size, int increment)
{
    int i;
    int j = 0;
    int max_ind = 0;
    float max_val = input[0];
    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] > max_val) {
            max_ind = i;
            max_val = input[j];
        }
    }
    return max_ind;
}
int arg_min (float *input, int size, int increment)
{
    int i;
    int j = 0;
    int min_ind = 0;
    float min_val = input[0];
    for (i = 1; i < size; i++) {
        j += increment;
        if (input[j] < min_val) {
            min_ind = i;
            min_val = input[j];
        }
    }
    return min_ind;
}



// Multiply two matrices a and b with given sizes
// Place the answer in "result_goes_here"
// We could do something tricky by switching the role of result and temp, to avoid copying
// but let's leave that for now
void matrix_multiply(float* a, float*b, int a_height, int a_width_b_height, int b_width, float* result_goes_here, float *result_temp) {

    int i, j, k;
    float value;

    // Need to allocate a temp array because one of the inputs (a or b) might be the same as result_goes_here
    // and if we start overwriting result_goes_here before the end we might mess up the calculation
    /*float result_temp[a_height*b_width];*/

    // Initialize to zero
    int size = a_height*b_width;
    clear_matrix (result_temp, size );
    /*for (i = 0; i < a_height*b_width; i++) {*/
    /*result_temp[i] = 0.0;*/
    /*}*/

    for (i = 0; i < a_height; i++) {
        for (j = 0; j < b_width; j++) {
            for(k = 0; k < a_width_b_height; k++) {
                value = conv_float2fixed(a[sub2ind(i,k,a_width_b_height)]) * conv_float2fixed(b[sub2ind(k,j,b_width)]);
                result_temp[sub2ind(i,j,b_width)] = conv_float2fixed(result_temp[sub2ind(i,j,b_width)] +  conv_float2fixed(value));
            }
        }
    }
    copy_matrix(result_temp, result_goes_here, size);
}




// Multiply the weight matrix a by the units b
// Treat the last row of the weights (b) as the biases
// So we expect a_width = b_height - 1
void matrix_multiply_with_bias(float* a, float*b, int a_height, int b_height, int b_width, float* result_goes_here, float *result_temp) {

    int i, j, k;
    float value;

    // Initialize to zero
    int size = a_height*b_width;
    clear_matrix (result_temp, size );
    /*for (i = 0; i < a_height*b_width; i++) {*/
    /*result_temp[i] = 0.0;*/
    /*}*/

    // a is hid, b is weights

    for (i = 0; i < a_height; i++) {
        for (j = 0; j < b_width; j++) {
            for (k = 0; k < b_height; k++) {
                value = conv_float2fixed(a[sub2ind(i,k,b_height)]) * conv_float2fixed(b[sub2ind(k,j,b_width)]);
                result_temp[sub2ind(i,j,b_width)] = conv_float2fixed(result_temp[sub2ind(i,j,b_width)] + conv_float2fixed(value));
            }
            result_temp[sub2ind(i,j,b_width)] = conv_float2fixed(result_temp[sub2ind(i,j,b_width)] + b[sub2ind(b_height,j,b_width)]);
        }
    }
    copy_matrix(result_temp, result_goes_here, size);
}


// Softmax function on matrix a
// a is num_test_cases by num_classes
// the softmax function exponentiates each element and then normalizes each row to sum to 1
// ** this function is in-place (modifies a) **
/*
float* softmax(float* a, int num_test_cases, int num_classes) {

    int i, j;
    float numerator, normaliz;
    for (i = 0; i < num_test_cases; i++) {
        // compute the normalization factor
        normaliz = 0.0;
        for (j = 0; j < num_classes; j++) {
            numerator = sigmoid(a[sub2ind(i,j,num_classes)]);
            // replace a[i,j] with exp(a[i,j])
            a[sub2ind(i,j,num_classes)] = numerator;
            normaliz += numerator;
        }
        for (j = 0; j < num_classes; j++) {
            a[sub2ind(i,j,num_classes)] /= normaliz;
        }
    }
    return a;
}
*/

void activation_fun(float* hid, int size, float* sigmoid_table) {
    if (ACTIVATION_FUN == 0) {
        RELU(hid, size);
    }
    else if (ACTIVATION_FUN == 1) {
        sigmoid_lookup(hid, size, sigmoid_table);
    }
    else {
        sigmoid(hid, size);
    }
}

void print_debug(float* hid, int rows_to_print, int cols_to_print, int num_columns) {
    int i, l;
    printf("\nHidden units:\n");
        for (i = 0; i < rows_to_print; i++) {
            for (l = 0; l < cols_to_print; l++) {
                printf("%f, ", hid[sub2ind(i, l, num_columns)]);
            }
            printf("\n");
        }
}

// Does the forward predictive pass of a neural net.
// Returns a float array of class predictions in row major format of size
// num_test_cases*num_labels
void nnet_fwd(float* data, float* weights, int* num_units, int* num_rows, int* num_columns, float* hid, float* hid_temp, float* sigmoid_table) {

    int i, j, l;

    // printf("\nDATA:\n");
    // for (i = 0; i < NUM_TEST_CASES; i++) {
    //     printf("Datum %d:\n", i);
    //     for (l = 0; l < INPUT_DIM; l++) {
    //         printf("%e, ", data[sub2ind(i, l, NUM_TEST_CASES)]);
    //     }
    //     printf("\n");
    // }

    // printf("\nWEIGHTS:\n\n");
    // for (l = 0; l < num_rows[0] * num_columns[0]; l++) {
    //     printf("%f\n", weights[l]);
    // }
    // printf("\nEND WEIGHTS:\n\n");

    // FORMAT HERE IS H TIMES W, NOT W TIMES H!!!!!
    // SO EACH DATA POINT IS A ***ROW****


    // Get hidden activations (hid should be num_test_cases x num_units[1])
    matrix_multiply_with_bias(data, weights, NUM_TEST_CASES, num_units[0], num_units[1], hid, hid_temp); // Don't need to grab 0th matrix

    if (PRINT_DEBUG == 1) {
        // Rows to print, cols to print, number of cols
        print_debug(hid, NUM_TEST_CASES, num_units[1], num_units[1]);
    }

    // Pass through activation function
    activation_fun(hid, NUM_TEST_CASES*num_units[1], sigmoid_table);

    if (PRINT_DEBUG == 1) {
        print_debug(hid, NUM_TEST_CASES, num_units[1], num_units[1]);
    }

    for (l = 1; l < NUM_LAYERS; l++) {

        // Get hidden activations
        matrix_multiply_with_bias(hid, grab_matrix(weights, l, num_rows, num_columns), NUM_TEST_CASES, num_units[l], num_units[l+1], hid, hid_temp);

        if (PRINT_DEBUG == 1) {
            print_debug(hid, NUM_TEST_CASES, num_units[l+1], num_units[l+1]);
        }

        // Pass through activation function
        activation_fun(hid, NUM_TEST_CASES*num_units[l+1], sigmoid_table);

        if (PRINT_DEBUG == 1) {
            print_debug(hid, NUM_TEST_CASES, num_units[l+1], num_units[l+1]);
        }

    }

    matrix_multiply_with_bias(hid, grab_matrix(weights, NUM_LAYERS, num_rows, num_columns), NUM_TEST_CASES, num_units[NUM_LAYERS], NUM_CLASSES, hid, hid_temp);
    // hid now contains the output

    if (PRINT_DEBUG == 1) {
        print_debug(hid, NUM_TEST_CASES, NUM_CLASSES, NUM_CLASSES);
    }

    // we now apply the softmax to turn the outputs into class probabilities
    // softmax(hid, NUM_TEST_CASES, num_units[NUM_LAYERS+1]);

    // if (PRINT_DEBUG == 1) {
    //     print_debug(hid, 10, NUM_CLASSES, NUM_CLASSES);
    // }

}


float randfloat() {
    return rand() / ((float)(RAND_MAX));
}


// This is the thing that we want to be good at in hardware
int main( int argc, const char* argv[] )
{
    int ret_f_scanf;
    // set random seed (need to #include <time.h>)
    srand(1);

    float* result;
    int i, j;

    int num_units[NUM_LAYERS+2];

    num_units[0] = INPUT_DIM;  // input dimensionality
    for (i = 1; i <= NUM_LAYERS; i++) {
        num_units[i] = NUM_HIDDEN_UNITS[i-1];
    }
    num_units[NUM_LAYERS+1] = NUM_CLASSES;    // number of classes

    int RANDOM_WEIGHTS = 1;
    int RANDOM_DATA = 1;

    // We have NUM_LAYERS weight matrices, sizes are given in num_units
    // NOTE: we do not necessarily need full precision here in the weights ...............
    int w_size = 0;                 // number of weights total
    int num_rows[NUM_LAYERS+1];     // the sizes of each weight matrix
    int num_columns[NUM_LAYERS+1];  // ""
    for (i = 0; i < NUM_LAYERS+1; i++) {
        printf("Weight matrix %d has size (%d, %d)\n", i, num_units[i]+1, num_units[i+1]);
        num_columns[i] = num_units[i] + 1; // For the bias
        num_rows[i] = num_units[i+1];
        w_size += num_columns[i]*num_rows[i];
    }
    printf("Network has %d weights in total.\n", w_size);
    float weights[w_size];

    if (RANDOM_WEIGHTS) {
        // Randomly initialize weights
        printf("Initializing weights randomly\n");

        for (i = 0; i < w_size; i++) {
          weights[i] = conv_float2fixed( (randfloat()-0.5) * 10); // Question: does nan output take longer in simulation?
        }
        // NOTE: FOR SIGMOID ACTIVATION FUNCTION, WEIGHTS SHOULD BE BIG
        // Otherwise everything just becomes ~0.5 after sigmoid, and results are boring
    }
    else {
        // Read in the weights
        printf("Reading in weights from %s\n", WEIGHTS_FILENAME);

        FILE* weights_file;
        weights_file=fopen(WEIGHTS_FILENAME, "r");
        if (weights_file == NULL) {
            fprintf(stderr, "Can't open input file %s!\n", WEIGHTS_FILENAME);
            exit(1);
        }

        float read_float;
        for (i = 0; i < w_size; i++) {
            ret_f_scanf = fscanf(weights_file,"%f,",&read_float);
            // printf("%f,", read_float);
            weights[i] = conv_float2fixed( read_float );
        }
        fclose(weights_file);
    }

    // for (i = 0; i < w_size; i++) {
    //     printf("%f,", weights[i]);
    // }


    float *data = (float *) malloc(sizeof(float) * NUM_TEST_CASES * INPUT_DIM);
    int *labels = (int *) malloc(sizeof(int) * NUM_TEST_CASES);

    if (RANDOM_DATA) {
        printf( "Initializing data randomly\n" );
    // Generate random input data, size num_test_cases by num_units[0] (input dimensionality)
        for (i = 0; i < NUM_TEST_CASES*INPUT_DIM; i++) {
            data[i] = conv_float2fixed( randfloat() - 0.5) ;
        }
        for (i = 0; i < NUM_TEST_CASES; i++) {
            labels[i] = 0; // set all labels to 0
        }
    }
    else {
        printf("Reading in %d data of dimensionality %d from %s\n", NUM_TEST_CASES, INPUT_DIM, INPUTS_FILENAME);

        FILE* data_file;
        float read_float;
        data_file = fopen(INPUTS_FILENAME, "r");
        if (data_file == NULL) {
            fprintf(stderr, "Can't open inputs text file!\n");
            exit(1);
        }

        for (i = 0; i < NUM_TEST_CASES; i++) {
            for (j = 0; j < INPUT_DIM; j++) {
                ret_f_scanf = fscanf(data_file, "%f,", &read_float);
                // printf("%f,", read_float);
                // fflush(stdout);
                // data[sub2ind(j, i, NUM_TEST_CASES)] = conv_float2fixed( read_float );
                data[sub2ind(i, j, INPUT_DIM)] = conv_float2fixed( read_float );
                // each data point is a *ROW* !!!!!!!!!!!!!
                // this is our convention!!!!!!!!!!!!!!!!
            }
        }
        fclose(data_file);

        printf("Reading in %d labels from %s\n", NUM_TEST_CASES, LABELS_FILENAME);
        FILE* labels_file;
        labels_file = fopen(LABELS_FILENAME, "r");
        if (labels_file == NULL) {
            fprintf(stderr, "Can't open labels text file.txt!\n");
            exit(1);
        }
        int read_int;
        for (i = 0; i < NUM_TEST_CASES; i++) {
            ret_f_scanf = fscanf(labels_file,"%d,",&read_int);
//            printf("(%d):%d  ", i, read_int);
            // fflush(stdout);
            labels[i] = read_int;
        }
        fclose(labels_file);
    }

    // for (i = 0; i < NUM_TEST_CASES*INPUT_DIM; i++) {
    //     printf("%f,", data[i]);
    // }
    // for (i = 0; i < NUM_TEST_CASES; i++) {
    //     printf("%d,", labels[i]);
    // }

    printf("Setting up arrays\n");
    // Get the dimensions of the biggest matrix that will ever come out of matrix_multiply
    // All of them will have NUM_TEST_CASES columns
    // So I just find the biggest number of rows
    int biggest_rows = num_units[1];
    for (i = 2; i < NUM_LAYERS+2; i++) {
        if (num_units[i] > biggest_rows) {
            biggest_rows = num_units[i];
        }
    }
    printf("Largest hidden/output layer: %d\n", biggest_rows);
    fflush(stdout);

    // Then, allocate memory for it. We will always place the result of our matrix multiplications in here.
    //float hid[NUM_TEST_CASES*biggest_rows];
    //float hid_temp[NUM_TEST_CASES*biggest_rows];
    float* hid = (float*) malloc(NUM_TEST_CASES*biggest_rows*sizeof(float));
    float* hid_temp = (float*) malloc(NUM_TEST_CASES*biggest_rows*sizeof(float));
    // This file is not looked at by aladdin so malloc is fine.
    // If I do the old version then I get a memory overflow, because the
    // max stack size is not big enough for TIMIT stuff.

    //testing
    /*printf("%16.16f, %16.16f\n", 0.0125, conv_float2fixed(0.0125));*/
    /*printf("%16.16f, %16.16f\n", 0.000000000125, conv_float2fixed(0.000000000125));*/
    /*printf("%16.16f, %16.16f\n", 1250000000000.0, conv_float2fixed(12500000000000.0));*/
    /*printf("%16.16f, %16.16f\n", 125.0, conv_float2fixed(125.0));*/
    /*printf("%16.16f, %16.16f\n", 25.0, conv_float2fixed(25.0));*/
    // for (i = 0; i < 3; i++) {
    //     printf("%d, %d\n", num_rows[i], num_columns[i]);
    // }

    printf("Setting up sigmoid lookup table...\n");
    // Build the sigmoid lookup table
    // May want to change this to be "non-centered"
    // to avoid (sigmoid_coarseness - 1.0)
    // so we can use bit shift in lookup function with fixed point precisions
    int sigmoid_coarseness = 1 << LG_SIGMOID_COARSENESS;
    float sigmoid_table[sigmoid_coarseness];
    float sig_step = (float) (SIG_MAX - SIG_MIN) / (sigmoid_coarseness - 1.0);
    float x_sig = (float) SIG_MIN;
    for (i = 0; i < sigmoid_coarseness; i++) {
        sigmoid_table[i] = conv_float2fixed(1.0 / (1.0 + exp(-x_sig)));
        // printf("%f, %f\n", x_sig, sigmoid_table[i]);
        x_sig += sig_step;
    }


    // Test the sigmoid lookup table
    // int n_test = 4;
    // float test[n_test];
    // test[0] = -20.0;
    // test[1] = -0.25;
    // test[2] = 0.25;
    // test[3] = 0.00;
    // float atest[n_test];
    // atest[0] = -20.0;
    // atest[1] = -0.25;
    // atest[2] = 0.25;
    // atest[3] = 0.00;
    // sigmoid(test, n_test);
    // sigmoid_lookup(atest, n_test, sigmoid_table);
    // printf("Sigmoid test (left: true, right: lookup): \n");
    // for (i = 0; i < n_test; i ++) {
    //     printf("%f, %f\n", test[i], atest[i]);
    // }


    // -------------------------------------------------------- //
    //     THIS IS THE FUNCTION BEING SIMULATED IN  HARDWARE    //
    // -------------------------------------------------------- //
    // Run a forward pass through the neural net
    printf( "Running forward pass\n" );
    nnet_fwd(data, weights, num_units, num_rows, num_columns, hid, hid_temp, sigmoid_table); // The function being synthesized

    // "hid" now contains the outputs

    // Print the result, maybe not all the test_cases
    int num_to_print = 1;
    // don't try to print more test cases than there are
    num_to_print = num_to_print < NUM_TEST_CASES ? num_to_print : NUM_TEST_CASES;
/*    printf("Un-normalized output for first %d test_cases:\n", num_to_print);
    for (i = 0; i < num_to_print; i++) {
        for (j = 0; j < NUM_CLASSES; j++) {
            printf("%.4f ", (float)hid[sub2ind(i, j, NUM_TEST_CASES)]);
        }
        printf("<-- %d\n", arg_max(hid+i*NUM_CLASSES, NUM_CLASSES, 1));
    } */

    // Compute the classification error rate
    int num_errors = 0;
    for (i = 0; i < NUM_TEST_CASES; i++) {
        if (arg_max(hid+i*NUM_CLASSES, NUM_CLASSES, 1) != labels[i]) {
            num_errors = num_errors + 1;
        }
    }
    float error_fraction = ((float) num_errors) / ( (float) NUM_TEST_CASES);
    printf("Fraction incorrect (over %d cases) = %f\n", NUM_TEST_CASES, error_fraction);
    FILE* output_labels = fopen("output_labels.out", "w");
    for (i = 0; i < NUM_TEST_CASES; i++) {
        fprintf(output_labels, "Test %d label: %d\n", i, arg_max(hid+i*NUM_CLASSES, NUM_CLASSES, 1));
    }
    fclose(output_labels);

    // Write this number to a file
    FILE* accuracy_file;
    accuracy_file = fopen("accuracy.txt", "w");
    fprintf(accuracy_file, "%f", error_fraction);
    fclose(accuracy_file);

    free(hid);
    free(hid_temp);
    free(data);
    free(labels);
}
