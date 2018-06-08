#ifndef _CORE_BACKEND_H_
#define _CORE_BACKEND_H_

#include <string>

// These are compile-time switches that selectively build a copy of SMAUG with
// a particular backend.
#define REFERENCE 0
#define SMVBACKEND 1

namespace smaug {

enum BackendName {
    Reference = REFERENCE,
    Smv = SMVBACKEND,
    UnknownBackend,
};

class ReferenceBackend {
   public:
    static const int Alignment = 0;
    static const bool PrecomputeBNVariance = true;
    static const bool ColumnMajorFCWeights = true;
    static const std::string Name;
};

namespace smv {
extern int kSpadSize;
}  // namespace smv

class SmvBackend {
   public:
    static const int Alignment = 8;
    static const bool PrecomputeBNVariance = true;
    static const bool ColumnMajorFCWeights = true;
    static const std::string Name;

    static int SpadSize() { return smv::kSpadSize; }
};

}  // namespace smaug

#endif
