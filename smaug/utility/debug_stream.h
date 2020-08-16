#ifndef _UTILITY_DEBUG_STREAM_H_
#define _UTILITY_DEBUG_STREAM_H_

#include <iostream>

namespace smaug {

/**
 * An stream class to consume debug logs. Depending on the globalDebugLevel,
 * logs are either printed to std::cout or swallowed.
 */
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

/** Initializes the global debug stream for the given debug level. */
void initDebugStream(int debugLevel);

/** Returns a DebugStream instance for the given debug level. */
const DebugStream& dout(int debugLevel);

}  // namespace smaug

#endif
