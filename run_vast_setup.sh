#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Vast.ai instance setup script
###############################################################################

############################
# Tokens
############################
HF_TOKEN=""

############################
# Helpers
############################
log() { printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

############################
# 1) Clone repo
############################
log "Cloning REPAIR repo"
git clone https://github.com/ha11ucin8/REPAIR.git
cd REPAIR

############################
# 2) Download models into ./hugging_cache
############################
log "Logging in to Hugging Face"
pip install -q huggingface_hub
hf auth login --token "$HF_TOKEN"

CACHE_DIR="./hugging_cache"
mkdir -p "$CACHE_DIR"

log "Downloading meta-llama/Meta-Llama-3-8B-Instruct"
hf download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir "$CACHE_DIR/llama3-8b-instruct"

log "Downloading Qwen/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$CACHE_DIR/qwen2.5-7b-instruct"

log "Downloading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir "$CACHE_DIR/deepseek-r1-distill-qwen-1.5b"

############################
# 3) Install requirements
############################
log "Installing requirements.txt"
pip install -r requirements.txt

log "Setup complete"
