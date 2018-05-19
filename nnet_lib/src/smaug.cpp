#include <string>

#include <boost/program_options.hpp>

#include "core/backend.h"
#include "core/globals.h"
#include "core/scheduler.h"
#include "modelconf/data_generator.h"
#include "modelconf/read_model_conf.h"

namespace po = boost::program_options;

using namespace smaug;

int main(int argc, char* argv[]) {
    std::string modelconf;
    po::options_description options("SMAUG options");
    options.add_options()("help", "Display this help message");

    po::options_description hidden;
    hidden.add_options()("model-config", po::value(&modelconf)->required(),
                         "Model configuration file");
    po::options_description all;
    all.add(options);
    all.add(hidden);

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
        std::cout << all << "\n";
        return 1;
    }
    std::cout << "Model configuration: " << modelconf << "\n";

    Workspace* workspace = new Workspace();
    Network* network = readModelConfiguration(modelconf, workspace);
    network->dumpDataflowGraph();
    DataGenerator<float>* generator = new GaussianDataGenerator<float>();
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
