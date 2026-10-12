#!/usr/bin/env bash
# Offline wrapper: reuse the shared Oh My Zsh tree already mounted at
# /iag_ad_01/ad/dushuyue instead of downloading from GitHub.
set -euo pipefail

SHARED_ZSH="${TIANYI_SHARED_ZSH:-/iag_ad_01/ad/dushuyue/oh-my-zsh/.oh-my-zsh}"
[[ -d "$SHARED_ZSH" ]] || {
  echo "[tianyi-env][error] shared Oh My Zsh not found: $SHARED_ZSH" >&2
  echo "Check that /iag_ad_01/ad/dushuyue is mounted and readable on the 4090 host." >&2
  exit 1
}
export ZSH="$SHARED_ZSH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup_tianyi_4090_env_full.sh" "$@"
