#ifndef _UTILITY_DEBUG_STREAM_H_
#define _UTILITY_DEBUG_STREAM_H_

#include <iostream>

namespace smaug {

class DebugStream {
   public:
    DebugStream(bool _enabled) : enabled(_enabled) {}

#ifndef FAST_BUILD
    template <typename T>
    const DebugStream& operator<<(T message) const {
        if (enabled)
            std::cout << message;
        return *this;
    }
#else
    template <typename T>
    const DebugStream& operator<<(T message) const {
        return *this;
    }
#endif

   protected:
    bool enabled;
};

void initDebugStream(int debugLevel);
const DebugStream& dout(int debugLevel);

}  // namespace smaug

#endif
