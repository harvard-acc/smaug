#!/bin/bash

git submodule update --init --recursive
cd nnet_lib/tests
make all
