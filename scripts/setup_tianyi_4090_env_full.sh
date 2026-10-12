#!/usr/bin/env bash
# Reproduce current TianYi development shell/Codex configuration on a new host.
# Credentials are intentionally excluded; authenticate Codex separately.
set -euo pipefail

TARGET_HOME="${HOME:?HOME must be set}"
ZSH_DIR="${ZSH:-$TARGET_HOME/.oh-my-zsh}"
ZSH_CUSTOM_DIR="${ZSH_CUSTOM:-$ZSH_DIR/custom}"
CODEX_DIR="${CODEX_HOME:-$TARGET_HOME/.codex}"
FORCE_ZSHRC="${TIANYI_FORCE_ZSHRC:-0}"
log() { printf '[tianyi-env] %s\n' "$*"; }
die() { printf '[tianyi-env][error] %s\n' "$*" >&2; exit 1; }
as_root() { if [[ "$(id -u)" == 0 ]]; then "$@"; elif command -v sudo >/dev/null 2>&1; then sudo "$@"; else die "need root or sudo for: $*"; fi; }

install_system_deps() {
  local missing=(); command -v zsh >/dev/null 2>&1 || missing+=(zsh); command -v git >/dev/null 2>&1 || missing+=(git); command -v curl >/dev/null 2>&1 || missing+=(curl)
  ((${#missing[@]} == 0)) && return
  command -v apt-get >/dev/null 2>&1 || die "missing ${missing[*]}; install with target OS package manager"
  log "Installing: ${missing[*]}"; as_root apt-get update; as_root apt-get install -y "${missing[@]}"
}

install_oh_my_zsh() {
  if [[ ! -d "$ZSH_DIR" ]]; then
    log "Installing Oh My Zsh into $ZSH_DIR"
    RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
  fi
  [[ -d "$ZSH_DIR" ]] || die "Oh My Zsh installation failed"
  mkdir -p "$ZSH_CUSTOM_DIR/plugins"
  for plugin in zsh-autosuggestions zsh-syntax-highlighting; do
    local dst="$ZSH_CUSTOM_DIR/plugins/$plugin"
    [[ -d "$dst" ]] || { log "Installing zsh plugin: $plugin"; git clone --depth=1 "https://github.com/zsh-users/$plugin.git" "$dst"; }
  done
}

write_zshrc() {
  local zshrc="$TARGET_HOME/.zshrc"
  if [[ -e "$zshrc" && "$FORCE_ZSHRC" != 1 ]]; then cp -a "$zshrc" "$zshrc.tianyi-backup.$(date +%Y%m%d_%H%M%S)"; zshrc="$TARGET_HOME/.zshrc.tianyi"; fi
  cat > "$zshrc" <<EOF
export ZSH="$ZSH_DIR"
ZSH_THEME="ys"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source "\$ZSH/oh-my-zsh.sh"
export OPENPI_ROOT="/iag_ad_01/ad/dushuyue/openpi"
export OPENPI_DATA_ROOT="\$OPENPI_ROOT/data"
export HF_HOME="\$OPENPI_DATA_ROOT/.hf_cache"
export HF_DATASETS_CACHE="\$HF_HOME"
export OPENPI_DATA_HOME="\$OPENPI_DATA_ROOT/.openpi_cache"
if [[ -f "$TARGET_HOME/miniconda3/etc/profile.d/conda.sh" ]]; then source "$TARGET_HOME/miniconda3/etc/profile.d/conda.sh"; fi
EOF
  chmod 600 "$zshrc"; log "zsh config written to $zshrc"
}

write_codex_config() {
  mkdir -p "$CODEX_DIR/rules"; chmod 700 "$CODEX_DIR"
  if [[ ! -e "$CODEX_DIR/config.toml" || "$FORCE_ZSHRC" == 1 ]]; then
    cat > "$CODEX_DIR/config.toml" <<'EOF'
model_provider = "crs"
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"
disable_response_storage = true
preferred_auth_method = "apikey"

[model_providers.crs]
name = "crs"
base_url = "https://crs.senseauto.com/openai"
wire_api = "responses"
requires_openai_auth = true
http_headers = { "User-Agent" = "codex_cli_rs/0.2333" }

[projects."/iag_ad_01/ad/dushuyue"]
trust_level = "trusted"
EOF
    chmod 600 "$CODEX_DIR/config.toml"
  fi
  if [[ -n "${TIANYI_CODEX_RULES:-}" && -f "$TIANYI_CODEX_RULES" ]]; then cp -a "$TIANYI_CODEX_RULES" "$CODEX_DIR/rules/default.rules"; chmod 600 "$CODEX_DIR/rules/default.rules"; fi
  log "Codex credentials are not copied; run codex login on target host"
}

main() {
  install_system_deps; install_oh_my_zsh; write_zshrc; write_codex_config
  if command -v codex >/dev/null 2>&1; then codex --version || true; else log "Codex CLI not found; install the OpenAI Codex VS Code extension or approved CLI"; fi
  log "Done. Start a new zsh (or exec zsh)."
}
main "$@"
