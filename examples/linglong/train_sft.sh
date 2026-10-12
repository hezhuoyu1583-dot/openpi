#!/usr/bin/env bash
set -euo pipefail
cd /iag_ad_01/ad/dushuyue/openpi
export OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/cache/openpi
export HF_DATASETS_CACHE=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/linglong_sft15
export HF_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.hf_cache/linglong_hf
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Do not compete with an existing policy service. No process is stopped here.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU has active compute processes; defer Linglong training.' >&2
  exit 1
fi
gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if awk '$1>1024 {busy=1} END {exit !busy}' <<< "$gpu_memory"; then
  echo 'GPU memory is occupied (including processes outside this container); defer training.' >&2
  exit 1
fi
[[ -f data/linglong_0714_sft_15hz/verification.json ]] || { echo 'Run prepare_sft and verify_sft first.' >&2; exit 1; }
.venv/bin/python -c 'import json; r=json.load(open("data/linglong_0714_sft_15hz/verification.json")); assert r["status"]=="passed"; assert r["full_training_loader_batch"]["actions"][-2:]==[25,32]'
experiment_name=${1:-linglong_0714_absolute_v1}
if [[ $# -gt 0 ]]; then shift; fi
exec .venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  examples/linglong/train.py pi05_linglong_sft --exp-name="$experiment_name" "$@"
