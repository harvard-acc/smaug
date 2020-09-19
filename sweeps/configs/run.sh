#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..

%(gem5-binary)s \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=1GB \
  --mem-type=%(mem-type)s  \
  --sys-clock=%(sys-clock)s \
  --cpu-clock=%(cpu-clock)s \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --l1d_size=%(l1d_size)s \
  --l1d_assoc=%(l1d_assoc)s \
  --l1d_hit_latency=%(l1d_hit_latency)s \
  --l2_size=%(l2_size)s \
  --l2_assoc=%(l2_assoc)s \
  --l2_hit_latency=%(l2_hit_latency)s \
  --cacheline_size=32 \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${SMAUG_HOME}/build/bin/smaug \
  -o "${topo_file} ${params_file}
      --sample-level=high
      --debug-level=0
      --gem5
      --num-accels=%(num-accels)s"
