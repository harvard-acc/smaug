FROM ubuntu:16.04
LABEL maintainer="Sam Xi"

# Install tools for development.
RUN apt-get update && apt-get install -y \
    python \
    python-pip \
    git \
    tmux \
    vim \
    cmake \
    wget \
    curl

# Install dependencies.
RUN apt-get install -y libconfuse-dev

RUN pip install --upgrade pip

WORKDIR /workspace

RUN git clone https://github.com/harvard-acc/gem5-aladdin && \
    cd gem5-aladdin && \
    git submodule update --init --recursive && \
    cd ../..

RUN git clone https://github.com/01org/mkl-dnn && \
    cd mkl-dnn/scripts && \
    ./prepare_mkl.sh && \
    cd .. &&  mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make install

ENV ALADDIN_HOME /workspace/gem5-aladdin/src/aladdin
ENV MKLDNNROOT /usr/local
ENV SHELL /bin/bash
ENV LD_LIBRARY_PATH $MKLDNNROOT/lib
