#!/bin/bash

# Variable
REPO_DIR=$(pwd)

# Commons For All GPUs

echo "WhisperS2T - CTranslate2"
python3 scripts/benchmark_whisper_s2t.py --repo_path="$REPO_DIR" --backend="CTranslate2" --batch_size=56 --eval_mp3="yes"

echo "WhisperS2T - OpenAI"
python3 scripts/benchmark_whisper_s2t.py --repo_path="$REPO_DIR" --backend="OpenAI" --batch_size=16 --eval_mp3="no"

echo "WhisperS2T - HuggingFace"
python3 scripts/benchmark_whisper_s2t.py --repo_path="$REPO_DIR" --backend="HuggingFace" --batch_size=48 --eval_mp3="no"

echo "WhisperS2T - HuggingFace - BT"
python3 scripts/benchmark_whisper_s2t.py --repo_path="$REPO_DIR" --backend="HuggingFace" --batch_size=48 --better_transformer="yes" --eval_mp3="no"

echo "WhisperX"
python3 scripts/benchmark_whisperx.py --repo_path="$REPO_DIR" --batch_size=56

echo "HuggingFace"
python3 scripts/benchmark_huggingface.py --repo_path="$REPO_DIR" --batch_size=48 --eval_mp3="no"

echo "HuggingFace - BT"
python3 scripts/benchmark_huggingface.py --repo_path="$REPO_DIR" --batch_size=48 --better_transformer="yes" --eval_mp3="no"

# Flash Attention 2 Supported Arch

echo "WhisperS2T - HuggingFace - FA"
python3 scripts/benchmark_whisper_s2t.py --repo_path="$REPO_DIR" --backend="HuggingFace" --batch_size=48 --flash_attention="yes" --eval_mp3="no"

echo "HuggingFace - FA"
python3 scripts/benchmark_huggingface.py --repo_path="$REPO_DIR" --batch_size=48 --flash_attention="yes" --eval_mp3="yes"

echo "OpenAI"
python3 scripts/benchmark_openai.py --repo_path="$REPO_DIR"
