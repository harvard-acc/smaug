FROM ubuntu:18.04
LABEL maintainer="Sam Xi, Yuan Yao"

###################################################
####      Install tools for development.       ####
###################################################

# By default, Ubuntun 18.04 installs Python 2.7.
RUN apt-get update && apt-get install -y \
      python \
      python-pip \
      git \
      tmux \
      vim \
      cmake \
      wget \
      curl
RUN pip install --upgrade pip

###########################################
####      Environment variables        ####
###########################################

RUN mkdir -p /workspace
ENV ALADDIN_HOME /workspace/gem5-aladdin/src/aladdin
ENV TRACER_HOME /workspace/LLVM-Tracer
ENV LLVM_HOME /usr/local
ENV BOOST_ROOT /usr/include
ENV SMAUG_HOME /workspace/smaug
ENV PROTOC /usr/local/bin/protoc
ENV PYTHONPATH "${SMAUG_HOME}/build:${PYTHONPATH}"
ENV SHELL /bin/bash

#######################################################
####      Install gem5-Aladdin dependencies.       ####
#######################################################

# Install gem5 dependencies.
RUN apt-get update -qq
RUN apt-get install -y m4 \
      libsqlite3-dev \
      libtcmalloc-minimal4 \
      scons \
      zlib1g \
      zlib1g-dev
RUN pip install six numpy

# Install Aladdin dependencies.
RUN apt-get install -y \
      libboost-graph-dev \
      libpthread-stubs0-dev \
      libreadline-dev \
      libncurses5-dev

# Install a supported version of pyparsing for Xenon.
RUN pip install pyparsing==2.3.0

# Build LLVM and Clang 6.0 from source.
WORKDIR /tmp
RUN wget -q http://releases.llvm.org/6.0.0/llvm-6.0.0.src.tar.xz && \
    tar -xf llvm-6.0.0.src.tar.xz && \
    wget -q http://releases.llvm.org/6.0.0/cfe-6.0.0.src.tar.xz && \
    tar -xf cfe-6.0.0.src.tar.xz && \
    mkdir -p llvm-6.0.0.src/tools/clang && \
    mv cfe-6.0.0.src/* llvm-6.0.0.src/tools/clang && \
    cd llvm-6.0.0.src && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release .. && \
    make -j24 && \
    make install && \
    rm -rf /tmp/llvm-6.0.0* && \
    rm -rf /tmp/cfe-6.0.0*

# Install a newer version of protobuf (3.11.4).
RUN wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-all-3.11.4.tar.gz && \
    tar -xzf protobuf-all-3.11.4.tar.gz && \
    cd protobuf-3.11.4 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/protobuf-3.11.4

WORKDIR /workspace

# Build and install LLVM-Tracer
RUN git clone https://github.com/ysshao/LLVM-Tracer && \
    cd LLVM-Tracer && \
    git fetch --all && \
    mkdir bin && \
    mkdir lib && \
    mkdir build && cd build && \
    cmake .. && make && make install

# Clone gem5-aladdin, but don't build. We'll assume the developer will build.
RUN git clone https://github.com/harvard-acc/gem5-aladdin && \
    cd gem5-aladdin && \
    git submodule init && \
    git submodule update

################################################
####      Install SMAUG dependencies.       ####
################################################

# Install SMAUG dependencies. We need Python 3.5+ to run the TensorFlow tests.
RUN apt-get install -y \
      python3 \
      python3-pip \
      libboost-program-options-dev

# Change the default Python version to 3.6.
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip2 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 2
RUN pip3 install --user --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

# Install TensorFlow and addons.
RUN pip install tensorflow tensorflow-addons

# Clone SMAUG, but don't build. We'll assume the developer will build.
WORKDIR /workspace
RUN git clone https://github.com/harvard-acc/smaug.git && \
    cd smaug && \
    git submodule init && \
    git submodule update

