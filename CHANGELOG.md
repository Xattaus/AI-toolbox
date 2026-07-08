# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-07-08

First public release.

### Features
- **Model Hub** — download from HuggingFace, browse/sort/search the local library, cleanup, import.
- **GGUF Tools** — convert HuggingFace → GGUF, quantize (Q8…Q2 and IQ types), VRAM calculator, iMatrix.
- **Training Center** — LoRA training (Unsloth/PEFT, QLoRA), dataset preparation, adapter merge, abliteration.
- **Mergekit Wizard** — SLERP, DARE-TIES, DARE-LINEAR, TIES and DELLA merges with ready-made presets and 10 GB-VRAM optimizations.
- **Abliteration** — remove the refusal direction from models, with a hardware-aware wizard.
- **Ollama Manager** — create, manage and run Ollama models from GGUF files.
- **Benchmark Suite** — speed, memory, latency and quality comparison.
- Finnish-first CLI with consistent Rich/Questionary theming.

### Security
- Zip Slip protection when extracting downloaded llama.cpp binaries.
- The config file (which may hold an HF token) is restricted to owner-only (`0600`) on POSIX.

### Reliability
- Pinned external model tools: llama.cpp tag `b9902`, mergekit `0.1.4`. A `constraints.txt` pins the core Python dependencies for repeatable installs (Python deps are otherwise version ranges; there is no full lockfile yet, and the llama.cpp download is not checksum-verified).
- GGUF conversion no longer reports a stale/leftover output file as a success.
- Atomic writes with `.bak` recovery for both the library index and the config file.

### Tooling & Quality
- GitHub Actions CI running the test suite on Python 3.10 and 3.12.
- 107-test suite covering config, paths, quantization, conversion, library, merging and dataset modules.
- Clear optional-dependency extras: `chat`, `training`, `unsloth`, `merge`, `mcp`, `all`.

### Requirements
- Python 3.10+ (the project previously mislabeled itself as 3.9-compatible).
- Prebuilt llama.cpp binaries target Windows x64; on Linux/macOS the `llama-quantize` binary is built from source.

[3.0.0]: https://github.com/Xattaus/AI-toolbox/releases/tag/v3.0.0
