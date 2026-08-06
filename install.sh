#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it first from https://brew.sh, then re-run this script." >&2
  exit 1
fi

echo "==> Installing Homebrew packages (whisper-cpp, ollama, portaudio, ffmpeg)"
brew install whisper-cpp ollama portaudio ffmpeg

echo "==> Starting the Ollama background service"
brew services start ollama

echo "==> Pulling local cleanup model (qwen2.5:3b)"
ollama pull qwen2.5:3b

echo "==> Downloading the Whisper medium multilingual model"
MODEL_DIR="$HOME/.config/dictify/models"
mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/ggml-medium.bin"
if [ ! -f "$MODEL_PATH" ]; then
  curl -fL -o "$MODEL_PATH.part" https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin
  mv "$MODEL_PATH.part" "$MODEL_PATH"
else
  echo "Model already present at $MODEL_PATH"
fi

echo "==> Setting up the Python virtual environment"
cd "$(dirname "$0")"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "==> Building Dictify.app"
./.venv/bin/python setup.py py2app -A

echo "==> Installing Dictify.app to /Applications"
rm -rf /Applications/Dictify.app
cp -R dist/Dictify.app /Applications/Dictify.app

REPO_DIR="$(pwd)"
LAUNCH_AGENT_PATH="$HOME/Library/LaunchAgents/local.dictify.plist"

install_launch_agent() {
  echo "==> Installing launch-at-login"
  sed "s|__WORKING_DIRECTORY__|$REPO_DIR|" local.dictify.plist.template > "$LAUNCH_AGENT_PATH"
  launchctl bootout "gui/$(id -u)/local.dictify" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$LAUNCH_AGENT_PATH"
}

if [ -f "$LAUNCH_AGENT_PATH" ]; then
  # Already installed from a previous run of this script - refresh it (new
  # code, and the repo may have moved) rather than asking again.
  install_launch_agent
else
  read -rp "Launch Dictify automatically at login? [y/N] " REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    install_launch_agent
  else
    echo "==> Launching Dictify once so you can try it (re-run install.sh later if you want launch-at-login)"
    open /Applications/Dictify.app
  fi
fi

echo "==> Done. Dictify is installed - press ⌃⌥⌘D to start dictating once macOS finishes prompting for permissions."
