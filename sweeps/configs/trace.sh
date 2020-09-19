#!/usr/bin/env bash

. ./model_files

${SMAUG_HOME}/build/bin/smaug-instrumented \
  ${topo_file} ${params_file} --sample-level=high --debug-level=0 --num-accels=%(num-accels)s
