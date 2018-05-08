#ifndef _UTILITY_UTILS_H_
#define _UTILITY_UTILS_H_

#include <string>

#include "core/datatypes.h"

namespace smaug {

// TODO: Allow these to take rvalue references.
template <typename T>
int product(std::vector<T> array) {
    int prod = 1;
    for (auto val : array)
        prod *= val;
    return prod;
}

template <typename T>
std::vector<T> sum(std::vector<T> array0, std::vector<T> array1) {
    assert(array0.size() == array1.size());
    std::vector<T> sum;
    for (int i = 0; i < array0.size(); i++)
      sum[i] = array0[i] + array1[i];
    return sum;
}

// Return the difference between @value and the next multiple of @alignment.
int calc_padding(int value, unsigned alignment);

std::string dataLayoutToStr(DataLayout layout);

}  // namespace smaug

#endif
