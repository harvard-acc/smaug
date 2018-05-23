#include <string>

#include <boost/program_options.hpp>

#include "core/backend.h"
#include "core/globals.h"
#include "core/scheduler.h"
#include "modelconf/data_generator.h"
#include "modelconf/read_model_conf.h"
#include "utility/debug_stream.h"

namespace po = boost::program_options;

using namespace smaug;

int main(int argc, char* argv[]) {
    std::string modelconf;
    std::string datamode = "RANDOM";
    int debugLevel = -1;
    po::options_description options(
            "SMAUG Usage:  ./smaug model.conf [options]");
    options.add_options()
        ("help,h", "Display this help message")
        ("data-init-mode,d", po::value(&datamode),
            "Random data generation mode (FIXED, RANDOM)")
        ("debug-level", po::value(&debugLevel)->implicit_value(0),
            "Set the debugging output level. If omitted, all debugging output "
            "is ignored. If specified without a value, the debug level is set to "
            "zero.");

    po::options_description hidden;
    hidden.add_options()(
            "model-config", po::value(&modelconf), "Model configuration file");
    po::options_description all, visible;
    all.add(options).add(hidden);
    visible.add(options);

    po::positional_options_description p;
    p.add("model-config", -1);
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
    if (modelconf.empty()) {
        std::cout << "[ERROR] Need to specify the model configuration file!\n"
                  << visible << "\n";
        return 1;
    }

    if (datamode != "FIXED" && datamode != "RANDOM") {
        std::cerr << "[ERROR] Invalid value for --data-init-mode: \""
                  << datamode << "\"\n";
        return 1;
    }
    initDebugStream(debugLevel);

    std::cout << "Model configuration: " << modelconf << "\n";

    Workspace* workspace = new Workspace();
    Network* network = readModelConfiguration(modelconf, workspace);
    network->dumpDataflowGraph();
    DataGenerator<float>* generator;
    if (datamode == "FIXED") {
        generator = new FixedDataGenerator<float>(0.1);
    } else if (datamode == "RANDOM") {
        generator = new GaussianDataGenerator<float>();
    } else {
        assert(false && "Invalid data init mode!");
    }

    generateWeights<float, GlobalBackend>(network, generator);
    Tensor<GlobalBackend>* inputTensor =
            workspace->getTensor<GlobalBackend>("input");
    generator->reset();
    generateRandomTensor<float, GlobalBackend>(inputTensor, generator);

    if (!network->validate())
        return -1;
    runNetwork<GlobalBackend>(network, workspace);

    delete generator;
    delete network;
    delete workspace;

    return 0;
}
