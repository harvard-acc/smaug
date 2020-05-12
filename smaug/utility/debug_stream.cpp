#include <cassert>
#include "smaug/utility/debug_stream.h"

namespace smaug {

static int globalDebugLevel = -1;
static DebugStream nullStream(false);
static DebugStream debugStream(true);

void initDebugStream(int debugLevel) {
    assert(globalDebugLevel == -1 &&
           "Debug stream cannot initialized more than once!");
    globalDebugLevel = debugLevel;
}

const DebugStream& dout(int debugLevel) {
    if (debugLevel >= 0 && debugLevel <= globalDebugLevel)
        return debugStream;
    return nullStream;
}

}  // namespace smaug
