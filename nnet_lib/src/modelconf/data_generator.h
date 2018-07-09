#ifndef _MODELCONF_DATA_GENERATOR_H_
#define _MODELCONF_DATA_GENERATOR_H_

#include <cmath>
#include <vector>

#include "core/network.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "operators/batch_norm_op.h"

namespace smaug {

class Network;

float randfloat();

template <typename DType>
class DataGenerator {
   public:
    DataGenerator() {}
    virtual ~DataGenerator() {}
    virtual DType next() = 0;
    virtual void reset() {}

    // Return a uniformly distributed value on [-RAND_MAX, RAND_MAX).
    //
    // Oftentimes, generating a value from some distribution will require first
    // generating a value from the uniform distribution.
    float genUniform() const { return rand() * RAND_MAX_RECIPROCAL; }

   protected:
    const float RAND_MAX_RECIPROCAL = (1.0 / RAND_MAX);
};

// Generates a sequence of increasing strided numbers.
//
// The stride is fixed and set to 1 by default. The first value generated is
// always zero.
template <typename DType>
class FixedDataGenerator : public DataGenerator<DType> {
   public:
    FixedDataGenerator(DType _stride = 1)
            : DataGenerator<DType>(), state(0), stride(_stride) {}
    using DataGenerator<DType>::DataGenerator;
    virtual DType next() {
        DType value = state;
        state += stride;
        return value;
    }
    virtual void reset() { state = 0; }

   protected:
    DType state;
    DType stride;
};

// Generates a uniformly distributed random value from [-RAND_MAX, RAND_MAX).
template <typename DType>
class UniformRandomDataGenerator : public DataGenerator<DType> {
   public:
    using DataGenerator<DType>::DataGenerator;
    virtual DType next() { return this->genUniform(); }
};

// Generates a normally distributed random value.
//
// This uses the Box-Muller method.
template <typename DType>
class GaussianDataGenerator : public DataGenerator<DType> {
   public:
    GaussianDataGenerator(DType _mu = 0, DType _sigma = 0.1)
            : DataGenerator<DType>(), mu(_mu), sigma(_sigma), useSaved(false),
              savedValue(0) {}

    virtual DType next() {
        if (useSaved) {
            useSaved = false;
            return savedValue * sigma + mu;
        } else {
            float u = this->genUniform();
            float v = this->genUniform();
            float scale = sqrt(-2 * log(u));
            float x = scale * cos(2 * 3.1415926535 * v);
            float y = scale * sin(2 * 3.1415926535 * v);
            savedValue = y;
            useSaved = true;
            return x * sigma + mu;
        }
    }

    virtual void reset() { useSaved = false; }

   protected:
    DType mu;
    DType sigma;
    bool useSaved;
    DType savedValue;
};

/* Generate a random tensor invariant to data layout.
 *
 * Invariance means that the data logically is the same across different
 * physical layouts. For example, if the tensor argument is in NHWC, then the
 * random tensor generated would be equivalent to calling this function with an
 * NCHW input tensor followed by a reorder operation to NHWC.
 */
template <typename DType, typename Backend>
void generateConsistentRandomTensor(Tensor<Backend>* tensor,
                                    DataGenerator<DType>* generator) {
    const TensorShape& shape = tensor->getShape();
    TensorIndexIterator index = tensor->startIndex();
    DType* ptr = tensor->template allocateStorage<DType>();
    if (shape.getLayout() == DataLayout::NCHW) {
        int N = shape[0];
        int C = shape[1];
        int H = shape[2];
        int W = shape[3];
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        ptr[index(n, c, h, w)] = generator->next();
    } else if (shape.getLayout() == DataLayout::NHWC) {
        int N = shape[0];
        int H = shape[1];
        int W = shape[2];
        int C = shape[3];
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                        ptr[index(n, h, w, c)] = generator->next();
    } else if (shape.getLayout() == DataLayout::NC) {
        int N = shape[0];
        int C = shape[1];
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                ptr[index(n, c)] = generator->next();
    } else if (shape.getLayout() == DataLayout::X) {
        // If it's X, then it can be any format, which means we don't know what
        // layout to target.
        assert(false && "Cannot generate random data if data layout is X!");
    }
}

template <typename DType, typename Backend>
void generateRandomTensor(Tensor<Backend>* tensor,
                          DataGenerator<DType>* generator) {
    generateConsistentRandomTensor<DType, Backend>(tensor, generator);
}

template <typename DType, typename Backend>
void generateWeights(Network* network, DataGenerator<DType>* generator) {
    for (auto it = network->begin(); it != network->end(); ++it) {
        generator->reset();
        Operator* op = it->second;
        std::vector<TensorBase*> weights = op->getParameterizableInputs();
        for (auto t : weights) {
            Tensor<Backend>* tensor = dynamic_cast<Tensor<Backend>*>(t);
            assert(tensor && "Weight tensor was of incorrect type!");
            generateRandomTensor<DType, Backend>(tensor, generator);
        }

        if (Backend::PrecomputeBNVariance &&
            op->getOpType() == OpType::BatchNorm) {
            auto variance =
                    op->getInput<Backend>(BatchNormOp<Backend>::Variance);
            DType* ptr = variance->template data<DType>();
            for (auto idx = variance->startIndex(); !idx.end(); ++idx) {
                DType value = std::abs(ptr[idx]);
                value = 1.0 / sqrt(value + BatchNormOp<Backend>::kEpsilon);
                ptr[idx] = value;
            }
        }
    }
}

}  // namespace smaug

#endif
