#include <cstdio>

#include "unsupported/Eigen/CXX11/Tensor"

#include "utility/eigen_utility.h"

namespace nnet_eigen {

using namespace ::Eigen;

// Return the error rate of the final network output.
float compute_errors(float* network_soft_pred,
                     int* correct_labels,
                     int batch_size,
                     int num_classes) {
    TensorMap<Tensor<float, 2>, Aligned64> soft_pred_map(
            network_soft_pred, batch_size, num_classes);
    Tensor<float, 1> hard_predictions = compute_hard_targets(soft_pred_map);

    int num_errors = 0;
    for (int i = 0; i < batch_size; i++) {
        if (hard_predictions(i) != correct_labels[i]) {
            num_errors = num_errors + 1;
        }
    }
    return ((float)num_errors) / batch_size; }

// Print the output labels and soft outputs.
void write_output_labels(const char* fname,
                         float* network_soft_pred,
                         int batch_size,
                         int num_classes) {
    TensorMap<Tensor<float, 2>, Aligned64> soft_pred_map(
            network_soft_pred, batch_size, num_classes);
    Tensor<float, 1> hard_predictions = compute_hard_targets(soft_pred_map);

    FILE* output_labels = fopen(fname, "w");
    for (int i = 0; i < batch_size; i++) {
        int pred = hard_predictions(i);
        fprintf(output_labels, "Test %d: %d\n  [", i, pred);
        for (int j = 0; j < num_classes; j++)
            fprintf(output_labels, "%f  ", soft_pred_map(i, j));
        fprintf(output_labels, "]\n");
    }
    fclose(output_labels);
}

}  // namespace nnet_eigen
