#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /iag_ad_01/ad/dushuyue/openpi
export ROBOARENA_TOKEN_FILE=/iag_ad_01/ad/dushuyue/data/roboarena_security/token
export ROBOARENA_TLS_CERT=/iag_ad_01/ad/dushuyue/data/roboarena_security/server.crt
export ROBOARENA_TLS_KEY=/iag_ad_01/ad/dushuyue/data/roboarena_security/server.key
# Set evaluator CIDRs once verified; don't invent or trust forwarded source IPs.
export ROBOARENA_ALLOWED_CIDRS="${ROBOARENA_ALLOWED_CIDRS:-}"
export OPENPI_DATA_HOME=/iag_ad_01/ad/dushuyue/openpi/data/.openpi_cache
export PYTHONPATH=/iag_ad_01/ad/dushuyue/roboarena:/iag_ad_01/ad/dushuyue/openpi/src
export PYTHONUNBUFFERED=1
exec .venv/bin/python examples/droid/serve_pi05_roboarena.py \
  --checkpoint-dir=/iag_ad_01/ad/dushuyue/openpi/data/checkpoints/pi05_droid \
  --host=0.0.0.0 --port=8000 --execution-horizon=8
