#!/bin/bash

python collect_scaling_data.py \
  data/1gpu/baseline_final:1 \
  data/1node/baseline_final:4 \
  data/2nodes/baseline:8 \
  data/10nodes/baseline:40 \
  data/25nodes/baseline:100 \
  data/50nodes/baseline:200 \
  data/100nodes/baseline:400


python collect_simple_profiler.py \
  data/1gpu/baseline_final:1 \
  data/1node/baseline_final:4 \
  data/2nodes/baseline:8 \
  data/10nodes/baseline:40 \
  data/25nodes/baseline:100 \
  data/50nodes/baseline:200 \
  data/100nodes/baseline:400 2>/dev/null