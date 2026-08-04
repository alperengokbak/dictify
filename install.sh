#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing Homebrew packages (whisper-cpp, ollama, portaudio, ffmpeg)"
brew install whisper-cpp ollama portaudio ffmpeg

echo "==> Starting the Ollama background service"
brew services start ollama

echo "==> Pulling local cleanup model (qwen2.5:3b)"
ollama pull qwen2.5:3b

echo "==> Downloading the Whisper medium multilingual model"
MODEL_DIR="$HOME/.config/dictate-mac/models"
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

echo "==> Done. Run the app with: dictate-mac/.venv/bin/python dictate-mac/dictate.py"
