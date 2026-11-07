#!/usr/bin/env bash
# usage: bash sync_llamacpp.sh <upstream-commit>            (dry run, changes nothing)
#        bash sync_llamacpp.sh <upstream-commit> --apply
set -euo pipefail
C="${1:?give an upstream commit}"; MODE="-n"; [[ "${2:-}" == "--apply" ]] && MODE=""
SRC=~/up-new; DST=~/edge-llm-server
rm -rf "$SRC" && mkdir "$SRC"
git -C ~/llama.cpp-upstream archive "$C" cmake CMakeLists.txt common ggml include scripts src tools vendor LICENSE | tar -x -C "$SRC"
sync_dir() {
  local d=$1; shift; local ex=()
  for e in "$@"; do ex+=(--exclude="/$e"); done
  echo "=== $d"
  rsync -a $MODE --delete -i "${ex[@]}" "$SRC/$d/" "$DST/$d/"
}
sync_dir cmake arm64-apple-clang.cmake arm64-windows-llvm.cmake riscv64-spacemit-linux-gnu-gcc.cmake x64-windows-llvm.cmake
sync_dir common
sync_dir include
sync_dir src
sync_dir vendor
sync_dir ggml .gitignore src/ggml-blas src/ggml-cann src/ggml-cuda src/ggml-hexagon src/ggml-hip src/ggml-metal src/ggml-musa src/ggml-openvino src/ggml-rpc src/ggml-sycl src/ggml-virtgpu src/ggml-webgpu src/ggml-zdnn src/ggml-zendnn src/ggml-cpu/spacemit
sync_dir tools cli completion cvector-generator export-lora fit-params gguf-split imatrix parser perplexity results rpc tokenize tts
sync_dir scripts apple gen-authors.sh gen-unicode-data.py hip install-oneapi.bat jinja pr2wt.sh serve-static.js snapdragon sync-ggml-am.sh sync-ggml.last sync-ggml.sh sync_vendor.py wc2wt.sh
echo "=== top-level files"
rsync -a $MODE -i "$SRC/CMakeLists.txt" "$SRC/LICENSE" "$DST/"
