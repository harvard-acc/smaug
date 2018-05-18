#ifndef _CORE_GLOBALS_H_
#define _CORE_GLOBALS_H_

#include "backend.h"

namespace smaug {

template <int Backend>
class BackendSelector {};

template <>
class BackendSelector<Reference> {
   public:
    typedef ReferenceBackend Backend;
};

using GlobalBackend = BackendSelector<CONFIG_BACKEND>::Backend;

}  // namespace smaug

#endif
