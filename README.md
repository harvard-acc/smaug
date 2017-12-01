Neural Network Library for gem5-Aladdin
=======================================

[![build status](https://travis-ci.com/xyzsam/composability.svg?token=rpmJkoccAjPMAABDKPM9&branch=master)](https://travis-ci.com/xyzsam/composability)

This is a library for building and simulationg neural networks, designed to
work with gem5-Aladdin. It supports the basic layer types and basic activation
functions for a linearly stack network architectures.

In this library, several reference implementations are provided, along with a
model of an actual SoC containing multiple DNN accelerators.

# Overview #

## Getting started ##

To build and run a network that resembles LeNet5, clone the repository and make
sure satisfy all dependencies (see the Dependencies section) are satisfied.

  ```bash
  git clone https://github.com/xyzsam/composability
  cd composability/nnet_lib
  make native
  ./build/nnet-monolithic-native ../models/mnist/lenet5-ish.conf
  ```

To understand what is happening, please continue to read.

## Why does this exist? ##

The need for hardware acceleration of deep learning workloads is obvious, but
understanding the impact of accelerator design choices on performance, power,
and area often requires manual implementation of the accelerator, and writing
RTL is slow. High level synthesis (HLS) can reduce this development time by
allowing the developer to describe hardware in a higher level language, but
ultimately, evaluation of the design requires the HLS to generate RTL, which
can be a very time consuming process. Aladdin is a pre-RTL accelerator
simulator which allows developers to explore the hardware design space by
describing their accelerator designs in C, without actually needing to write
RTL or wait for RTL to be generated.

However, like HLS tools, Aladdin supports a subset of the C language, and the
coding style required is rather unique and idiosyncratic. Writing C to describe
hardware is radically different from writing C for high performance software.
As a result, we cannot simply run code from existing optimized linear algebra
libraries, like BLAS, MKL, or Eigen, through Aladdin.

This library provides a set of implementation of core deep learning kernels
that comply with Aladdin's coding requirements. Some of the implementations are
merely reference implementations, while others represent actual accelerator
designs.  It also integrates these kernels with gem5-Aladdin, so that the
combined SoC simulator can be used to measure overall end-to-end performance.
The user can decide which kernels to offload to specialized hardware and which
kernels to run on the CPU.

## Features ##
Layer types supported:
* Fully connected
  - Arbitrary number of hidden nodes *
* Convolutional
  - Arbitrary kernel size *
* Pooling
  - Max pooling

Activation functions:
* ReLU, ReLU thresholded
* Sigmoid

Network configuration:
* Caffe style configuration files
* Linear stack of layers
* Arbitrarily deep networks

Backends:
* Reference - manual naive implementations of each kernel.
* Eigen - high performance on CPU.
* Aladdin - simulation of hardware acceleration.

\* This depends on the SoC architecture and accelerator support.

### Unsupported features ###
Plan to support:
* Softmax
* tanh activation function
* Recurrent layers
* Average pooling

No plans to support:
* Skip connections
* Inception modules

# Using the library #

## Dependencies ##

* [gem5-Aladdin](https://github.com/harvard-acc/gem5-aladdin)

  The `ALADDIN_HOME` environment variable must be set to where the Aladdin
  submodule is located.
* [LLVM-Tracer](https://github.com/ysshao/LLVM-Tracer)

  The `LLVM_HOME` environment variable must be set to where LLVM 3.4 is
  installed, and `TRACER_HOME` must be set to where LLVM-Tracer is installed.
* [libconfuse 3.2](https://github.com/martinh/libconfuse)

  If libconfuse cannot be installed to the system default directory
  (/usr/local), then set the `CONFUSE_ROOT` environment variable to the
  installation location.
* gcc 5.4.0 or later
* Eigen 3.3.4 (this is bundled so you don't need to install it separately).

## Architectures and Execution Targets ##

This library can be built for a particular "SoC architecture" and "execution
target". These terms are described below:

**SoC architecture**: this describes the collection of hardware accelerators
that the SoC contains. Based on what functionality can be offloaded, an
architecture implementation will offload specific computations to the
dedicated hardware blocks.

**Execution target**: This determines where the actual binary executable will
be run: on a native host or in simulation under gem5.

Currently, we have four architectures and three execution targets.

**Architectures**

1. Monolithic: This represents a hypothetical SoC with a single hardware
   block that can run an entire network. The CPU simply hands the accelerator
   a description of the entire network, offloads the input data and weights,
   and the accelerator will run until it produces its final output predictions.
   This implementation is considered a reference implementation, as it
   implements GEMM and convolution in the simplest way possible.

2. Composable: This represents a hypothetical SoC with a collection of hardware
   blocks that each handle a particular kernel, like GEMM or convolution. The
   CPU is responsible for moving data around between accelerators and invoking
   each accelerator as needed based on the layer type. The idea here is that
   this architecture affords much more flexibility than the monolithic
   architecture. This is also a reference implementation.

3. SMIV: This represents the SMIV SoC that was taped out by the Harvard
   Architecture, Circuits, and Compilers research group. The implementations
   are designed to model the actual dataflow and logic of the taped out
   accelerators. This is NOT a reference implementation, and certain types of
   GEMMs and convolutions are not supported, based on the hardware spec.
   Functionality that cannot be accelerated in hardware are offloaded to the
   CPU, using the Eigen backend whenever possible.

4. Eigen: This represents a CPU-only scenario in which we use Eigen as a highly
   optimized backend for all computation. No accelerators are involved here.

**Execution targets**

1. Native: this builds the executable to run everything on the host CPU.
   Naturally, Aladdin is not involved, since Aladdin is a simulator.

   Debug: this is a sub-target of native. It enables debug print messages.
   Currently, intermediate layer output and data are printed to stdout.

2. gem5: this builds the executable to be run under gem5. There are two
   executables that are produced:
   * gem5-cpu: This executable is designed to be simulated on a gem5 cpu, but
     since it does not invoke Aladdin, it *could* also be executed on the host
     machine.
   * gem5-accel: This executable will invoke Aladdin when the functionality can
     be offloaded, based on the architecture. As a result, this can only be run
     in simulation, not on the host machine.

   Note that gem5 only supports the SSE vector extensions on x86, so the Eigen
   architecture cannot take advantage of AVX instructions in simulation.
   However, AVX instructions are available to the native target, if the host
   machine's CPU supports them.
3. Trace: this instruments the binary using LLVM-Tracer, so that a dynamic
   trace can be generated and used by Aladdin.

## Build instructions ##

**All commands given in this section assume that your current working
directory is** `composability/nnet_lib`.

To build the binary, specify the execution target as the Make target and the
Soa architecture through the ARCH variable. The available execution targets
are displayed by running `make help`.

For example, to build the SMIV architecture for native and gem5 execution:

   ```bash
   make native ARCH=SMIV
   make gem5 ARCH=SMIV
   ```

To build the instrumented tracing executable, for the monolithic architecture:

   ```bash
   make dma-trace-binary ARCH=MONOLITHIC
   ```

Eigen is an exception to this build instruction: instead of specifying
ARCH=EIGEN, specify `eigen` as the Make target for the native target, and
`eigen-gem5` for the gem5 execution target. Leave the ARCH field blank.

   ```bash
   make eigen  # Builds Eigen for native execution.
   make eigen-gem5  # Builds Eigen for gem5 simulation.
   ```

All build products will be produced in the `build/`subdirectory.

## Running a network ##

In the most basic form, pass the path to the model configuration file as the
only argument. A set of models for commonly used networks can be found in the
`models` directory. For example:

  ```bash
  ./build/nnet-monolithic-native ../models/mnist/lenet5-ish.conf
  ```

You can also specify how to initialize the data - either pseudorandomly, with a
fixed incrementing pattern (useful for debugging), or from a file, using the -d
option. If you use to read from a file, you must also provide the path to the
file.

  ```bash
  ./build/nnet-monolithic-native path/to/model/file \
     -d [RANDOM | FIXED | READ_FILE] -f path/to/file
  ```

The file contains data either stored in text format (.txt suffix) or binary
format (.bin suffix). The binary format is far far far faster to load and
store, both natively and in simulation, so we highly recommend it over the text
format.

To generate this archive file, pass the `-s` option to the binary as well as
the `-f` option to specify the filename.

  ```bash
  ./build/nnet-monolithic-native path/to/model/file \
     -d [RANDOM | FIXED] -s -f path/to/output/file
  ```

This file contains weights, inputs, and output labels. Why bundle inputs and
weights together? Well, this library is only designed to evaluate performance
on the feed-forward pass, so the actual data values (generally) do not matter.

All available options can be shown wth the `--help` flag.

When execution is completed, the final layer's output soft targets and final
predicted label are written to `output_labels.out`.

## Running a network in simulation ##

The quickest way to get started with simulation is to run an already-existing
setup. These are located under the `sim` directory. For this example, let's use
the monolithic architecture and run LeNet5.

First, we need to build gem5-aladdin. Please refer to that repo for detailed
instructions.

Second, build the gem5 simulation binaries. Assuming your current working
directory is `nnet_lib`:

   ```bash
   make gem5 ARCH=MONOLITHIC
   ```

Finally, go to the appropriate sim directory.

   ```bash
   cd ../sim/monolithic/LeNet5-ish
   sh run.sh
   ```

If everything has been set up correctly, gem5 will now run the network in
simulation, invoking Aladdin to handle any accelerated regions as defined by
the monolithic architecture. The simulation trace can be found in `stdout.gz`,
and additional output files are in the `outputs` directory.

# Questions #

Please direct them to Sam Xi (samxi [at] seas dot harvard dot edu).
