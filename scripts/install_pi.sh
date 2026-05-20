#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  python3 \
  python3-numpy \
  python3-opencv \
  python3-pip \
  python3-venv

python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deep,agent]"

if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
fi

pisight --config config.yaml doctor
