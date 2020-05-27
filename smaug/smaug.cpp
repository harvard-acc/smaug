#include <fstream>
#include <string>

#include <boost/program_options.hpp>

#include "core/backend.h"
#include "core/globals.h"
#include "core/scheduler.h"
#include "core/network_builder.h"
#include "operators/common.h"
#include "utility/debug_stream.h"
#include "utility/utils.h"
#include "utility/thread_pool.h"

namespace po = boost::program_options;

using namespace smaug;

int main(int argc, char* argv[]) {
    std::string modelTopo;
    std::string modelParams;
    int debugLevel = -1;
    std::string lastOutputFile;
    bool dumpGraph = false;
    runningInSimulation = false;
    SamplingInfo sampling;
    std::string samplingLevel = "no";
    sampling.num_sample_iterations = 1;
    numAcceleratorsAvailable = 1;
    int numThreads = -1;
    useSystolicArrayWhenAvailable = false;
    po::options_description options(
            "SMAUG Usage:  ./smaug model_topo.pbtxt model_params.pb [options]");
    // clang-format off
    options.add_options()
        ("help,h", "Display this help message")
        ("debug-level", po::value(&debugLevel)->implicit_value(0),
         "Set the debugging output level. If omitted, all debugging output "
         "is ignored. If specified without a value, the debug level is set "
         "to zero.")
        ("dump-graph", po::value(&dumpGraph)->implicit_value(true),
         "Dump the network in GraphViz format.")
        ("gem5", po::value(&runningInSimulation)->implicit_value(true),
         "Run the network in gem5 simulation.")
        ("print-last-output,p",
         po::value(&lastOutputFile)->implicit_value("stdout"),
         "Dump the output of the last layer to this file. If specified with "
         "'proto', the output tensor is serialized to a output.pb file. By "
         "default, it is printed to stdout.")
        ("sample-level",
          po::value(&samplingLevel)->implicit_value("no"),
         "Set the sampling level. By default, SMAUG doesn't do any sampling. "
         "There are five options of sampling: no, low, medium, high and "
         "very_high. With more sampling, the simulation speed can be greatly "
         "improved at the expense of accuracy loss.")
        ("sample-num",
          po::value(&(sampling.num_sample_iterations))->implicit_value(1),
         "Set the number of sample iterations used by every sampling enabled "
         "entity. By default, the global sample number is set to 1. Larger "
         "sample number means less sampling.")
        ("num-accels",
          po::value(&numAcceleratorsAvailable)->implicit_value(1),
          "The number of accelerators that the backend has. As far as "
          "simulation goes, if there are multiple accelerators available, "
          "SMAUG requires the accelerator IDs (configured in the gem5 "
          "configuration file) to be monotonically incremented by 1.")
        ("num-threads",
         po::value(&numThreads)->implicit_value(1),
         "Number of threads in the thread pool.")
        ("use-systolic-array",
         po::value(&useSystolicArrayWhenAvailable)->implicit_value(true),
         "If the backend contains a systolic array, use it whenever possible.");
    // clang-format on

    po::options_description hidden;
    hidden.add_options()("model-topo-file", po::value(&modelTopo),
                         "Model topology protobuf file");
    hidden.add_options()("model-params-file", po::value(&modelParams),
                         "Model parameters protobuf file");
    po::options_description all, visible;
    all.add(options).add(hidden);
    visible.add(options);

    po::positional_options_description p;
    p.add("model-topo-file", 1);
    p.add("model-params-file", 1);
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
    if (modelTopo.empty() || modelParams.empty()) {
        std::cout << "The model protobuf files must be specified!\n";
        exit(1);
    }
    initDebugStream(debugLevel);

    std::cout << "Model topology file: " << modelTopo << "\n";
    std::cout << "Model parameters file: " << modelParams << "\n";

    if (samplingLevel == "no") {
        sampling.level = NoSampling;
    } else if (samplingLevel == "low") {
        sampling.level = Low;
    } else if (samplingLevel == "medium") {
        sampling.level = Medium;
    } else if (samplingLevel == "high") {
        sampling.level = High;
    } else if (samplingLevel == "very_high") {
        sampling.level = VeryHigh;
    } else {
        std::cout << "Doesn't support the specified sampling option: "
                  << samplingLevel << "\n";
        exit(1);
    }
    if (sampling.level > NoSampling) {
        std::cout << "Sampling level: " << samplingLevel
                  << ", number of sample iterations: "
                  << sampling.num_sample_iterations << "\n";
    }

    if (numAcceleratorsAvailable > maxNumAccelerators) {
        std::cout << "The number of accelerators exceeds the max number!\n";
        exit(1);
    }
    std::cout << "Number of accelerators: " << numAcceleratorsAvailable << "\n";
    if (numAcceleratorsAvailable > 1 && runningInSimulation) {
        std::cout << "SMAUG requires the accelerator IDs (configured in the "
                     "gem5 configuration file) to be monotonically incremented "
                     "by 1.\n";
    }

    if (numThreads != -1) {
        std::cout << "Using a thread pool, size: " << numThreads << ".\n";
        threadPool = new ThreadPool(numThreads);
    }

    Workspace* workspace = new Workspace();
    Network* network =
            buildNetwork(modelTopo, modelParams, sampling, workspace);
    SmvBackend::initGlobals();

    if (dumpGraph)
        network->dumpDataflowGraph();

    if (!network->validate())
        return -1;

    Scheduler scheduler(network, workspace);
    Tensor* output = scheduler.runNetwork();

    if (!lastOutputFile.empty()) {
        if (lastOutputFile == "stdout") {
            std::cout << "Final network output:\n" << *output << "\n";
        } else if (lastOutputFile == "proto") {
            // Serialize the output tensor into a proto buffer.
            std::fstream outfile("output.pb", std::ios::out | std::ios::trunc |
                                                      std::ios::binary);
            TensorProto* tensorProto = output->asTensorProto();
            if (!tensorProto->SerializeToOstream(&outfile)) {
                std::cerr << "Failed to serialize the output tensor and write "
                             "it to the given C++ ostream! Did you run out of "
                             "disk space?\n";
                return 1;
            }
            delete tensorProto;
        } else {
            std::ofstream outfile(lastOutputFile);
            outfile << "Final network output:\n" << *output << "\n";
        }
    }

    if (threadPool)
        delete threadPool;

    delete network;
    delete workspace;
    SmvBackend::freeGlobals();

    return 0;
}
