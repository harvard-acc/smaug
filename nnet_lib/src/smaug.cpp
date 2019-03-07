#include <string>

#include <boost/program_options.hpp>

#include "core/backend.h"
#include "core/scheduler.h"
#include "core/network_builder.h"
#include "utility/debug_stream.h"

namespace po = boost::program_options;

using namespace smaug;

int main(int argc, char* argv[]) {
    std::string modelpb;
    int debugLevel = -1;
    std::string lastOutputFile;
    bool dumpGraph = false;
    po::options_description options("SMAUG Usage:  ./smaug model.pb [options]");
    options.add_options()
        ("help,h", "Display this help message")
        ("debug-level", po::value(&debugLevel)->implicit_value(0),
            "Set the debugging output level. If omitted, all debugging output "
            "is ignored. If specified without a value, the debug level is set to "
            "zero.")
        ("dump-graph", po::value(&dumpGraph)->implicit_value(true),
            "Dump the network in GraphViz format.")
        ("print-last-output,p",
            po::value(&lastOutputFile)->implicit_value("stdout"),
            "Dump the output of the last layer to this file. If specified with "
            "no argument, it is printed to stdout.");

    po::options_description hidden;
    hidden.add_options()(
            "model-pb-file", po::value(&modelpb), "Model protobuf file");
    po::options_description all, visible;
    all.add(options).add(hidden);
    visible.add(options);

    po::positional_options_description p;
    p.add("model-pb-file", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                      .options(all)
                      .positional(p)
                      .run(),
              vm);
    try {
        po::notify(vm);
    } catch (po::error& e) {
        std::cout << "ERROR: " << e.what() << "\n";
        exit(1);
    }

    if (vm.count("help")) {
        std::cout << visible << "\n";
        return 1;
    }
    if (modelpb.empty()) {
        std::cout << "The model protobuf file must be specified!\n";
        exit(1);
    }
    initDebugStream(debugLevel);

    std::cout << "Model protobuf file: " << modelpb << "\n";

    Workspace* workspace = new Workspace();
    Network* network = buildNetwork(modelpb, workspace);
    network->dumpDataflowGraph();
    if (dumpGraph)
        network->dumpDataflowGraph();

    if (!network->validate())
        return -1;

    delete network;
    delete workspace;

    return 0;
}
