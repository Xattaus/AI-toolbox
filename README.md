<div align="center">

<img src="assets/banner.svg" alt="AI TOOLBOX — The local LLM workshop" width="820">

**The local LLM workshop** — download, convert, train, merge, and uncensor local LLMs.

![Python](https://img.shields.io/badge/Python-3.9%2B-ff9d00?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-ff9d00?style=flat-square)
![Version](https://img.shields.io/badge/Version-3.0.0-ff9d00?style=flat-square)
![Platform](https://img.shields.io/badge/Windows%20%7C%20Linux%20%7C%20macOS-ff9d00?style=flat-square)

</div>

A powerful collection of tools for working with local AI models. Download, convert, train, merge, abliterate, and chat with your LLM models.

---

## What's New in 3.0

- **Mergekit Wizard** - Professional model merging with SLERP, DARE-TIES, DARE-LINEAR, TIES, DELLA methods
- **Abliteration** - Remove refusal behavior from models
- **Model Hub** - Enhanced library with sorting, cleanup, and smart naming
- **Ollama Manager** - Create and manage Ollama models
- **Training Center** - Unified LoRA training, datasets, merging, and abliteration
- **Beautiful CLI** - Consistent styling with technical terms in English, descriptions in Finnish

---

## Features Overview

### Main Menu (7 Options)

| Category | Tool | Description |
|----------|------|-------------|
| **Chat** | Tool Master | Interactive chat with local AI models |
| | Claude Assistant | Claude CLI for development |
| **Models** | Model Hub | Download, browse & manage models |
| | GGUF Tools | Convert, quantize & VRAM calculator |
| | Ollama Manager | Create and manage Ollama models |
| **Advanced** | Training Center | LoRA, datasets, merging & abliteration |
| | Benchmark Suite | Performance testing & comparison |

---

## Quick Start

### Windows

```batch
# First time setup
setup.bat

# Launch AI Toolbox
toolbox.bat
```

### Linux/macOS

```bash
chmod +x toolbox.sh
./toolbox.sh   # creates venv and installs dependencies on first run
```

---

## Model Hub

Central hub for model management:

| Feature | Description |
|---------|-------------|
| **Download** | Search and download from HuggingFace |
| **Library** | Browse all models with smart sorting |
| **Cleanup** | Remove duplicates and missing files |
| **Import** | Add local models to library |
| **Search** | Filter by name, format, or tags |

### Sorting Options

- By date (newest first)
- By name (alphabetical)
- By size (largest first)
- By quantization (Q8 → Q2)
- By format (GGUF, SafeTensors)

---

## GGUF Tools

Convert and optimize models for local inference:

| Tool | Description |
|------|-------------|
| **Convert** | HuggingFace to GGUF format |
| **Quantize** | Reduce model size (Q8, Q4, Q2...) |
| **VRAM Calculator** | Estimate memory requirements |
| **iMatrix** | Importance matrix for better Q2-Q4 |

---

## Training Center

Unified hub for all training and model modification tools:

### LoRA Training

| Feature | Description |
|---------|-------------|
| **Quick Train** | One-click training with defaults |
| **Advanced Train** | Full parameter control |
| **Test Adapter** | Test trained adapter before merging |
| **Merge Adapter** | Merge adapter into base model |

- Automatic Unsloth acceleration (2-5x faster, 50-70% less VRAM)
- QLoRA support (4-bit/8-bit training)
- Supports Alpaca, Chat, and ShareGPT formats

### Dataset Tools

| Feature | Description |
|---------|-------------|
| **Inspect** | Analyze dataset structure |
| **Convert** | Convert between formats |
| **Clean** | Remove duplicates, filter by length |
| **Split** | Create train/test/val splits |
| **Merge** | Combine multiple datasets |
| **Token Count** | Count tokens in dataset |

### Mergekit Wizard

Professional model merging with optimized VRAM usage:

| Method | Models | Description |
|--------|--------|-------------|
| **SLERP** | 2 | Spherical interpolation (most popular) |
| **DARE-TIES** | 2+ | Advanced pruning + TIES (recommended) |
| **DARE-LINEAR** | 2+ | Linear DARE variant |
| **TIES** | 2+ | Task-specific interpolation |
| **DELLA** | 2+ | Efficient pruning method |

**Features:**
- Works with 10GB VRAM (lazy loading, sharding)
- Automatic architecture compatibility check
- YAML config save/load
- Ready-to-use presets
- Automatic vocab_size handling

**Presets:**

| Preset | Method | Use Case |
|--------|--------|----------|
| `slerp_balanced` | SLERP t=0.5 | Balanced 2-model merge |
| `slerp_light_finetune` | SLERP t=0.3 | Light finetune weight |
| `dare_ties_language` | DARE-TIES | Language model merging |
| `reasoning_r1_style` | SLERP t=0.35 | Reasoning optimized |
| `della_efficient` | DELLA | Pruning + merge |

### Abliteration

Remove refusal behavior from models:

| Feature | Description |
|---------|-------------|
| **Remove Censorship** | Abliterate refusal direction |
| **Test Model** | Verify abliteration results |

---

## Ollama Manager

Create and manage Ollama models from GGUF files:

| Feature | Description |
|---------|-------------|
| **Create Model** | Create Ollama model from GGUF |
| **List Models** | Show all Ollama models |
| **Delete Model** | Remove Ollama model |
| **Run Model** | Start chat with model |

---

## Benchmark Suite

Compare model performance:

| Metric | Description |
|--------|-------------|
| **Speed** | Tokens per second |
| **Memory** | RAM/VRAM usage |
| **Quality** | Response comparison |
| **Latency** | Time to first token |

---

## Installation

### Requirements

- Python 3.9 or higher
- 8GB+ RAM (for model operations)
- NVIDIA GPU recommended (for training/merging)
- Internet connection (for downloading models)

> **Platform support:** The app runs on Windows, Linux, and macOS, and the
> Python dependencies install automatically everywhere. However, **prebuilt
> `llama.cpp` binaries are downloaded for Windows x64 only.** On Linux/macOS
> the `llama.cpp` source is cloned but the `llama-quantize` binary needed for
> quantization must be built manually (`cd ~/.ai-toolbox/llama.cpp && make`).
> GGUF conversion itself works on all platforms.

### Automatic Setup

1. Download or clone this repository
2. **Windows:** run `toolbox.bat` (or `setup.bat`) — creates the venv and installs on first run
   **Linux/macOS:** run `./toolbox.sh` — same, sets up automatically on first run
3. That's it: dependencies install automatically, no manual steps

### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install
pip install -e .
```

### Optional Dependencies

```bash
# For LoRA training
pip install peft transformers datasets accelerate

# For Mergekit
pip install mergekit

# For Abliteration
pip install transformers torch
```

---

## Directory Structure

```
AI Toolbox/
├── src/
│   └── ai_toolbox/
│       ├── cli/                 # CLI commands
│       │   ├── app.py           # Main application
│       │   ├── model_hub_cmd.py # Model Hub commands
│       │   ├── gguf_tools_cmd.py
│       │   ├── training_center_cmd.py
│       │   ├── merger_cmd.py    # Mergekit Wizard
│       │   ├── training_cmd.py
│       │   ├── dataset_cmd.py
│       │   ├── abliteration_cmd.py
│       │   ├── benchmark_cmd.py
│       │   ├── ollama_cmd.py
│       │   └── settings_cmd.py
│       ├── core/                # Core utilities
│       │   ├── ui.py            # UI components & themes
│       │   ├── paths.py         # Path management
│       │   └── config.py        # Configuration
│       ├── models/              # Model management
│       │   ├── library.py       # Model library
│       │   ├── downloader.py
│       │   └── types.py
│       ├── conversion/          # GGUF conversion
│       │   ├── converter.py
│       │   ├── quantization.py
│       │   └── llama_cpp.py
│       ├── training/            # Training tools
│       │   ├── lora.py          # LoRA trainer
│       │   └── dataset.py       # Dataset preparation
│       ├── inference/           # Inference tools
│       │   ├── chat.py          # AI chat
│       │   ├── benchmark.py
│       │   └── assistant.py
│       ├── merging/             # Model merging
│       │   ├── merger.py        # Legacy merger
│       │   ├── mergekit_wrapper.py  # Mergekit integration
│       │   ├── config_manager.py    # YAML config management
│       │   └── presets.py       # Ready-to-use presets
│       ├── abliteration/        # Abliteration tools
│       │   ├── abliterator.py
│       │   ├── hooks.py
│       │   ├── prompts.py
│       │   └── testing.py
│       ├── integrations/        # External integrations
│       │   ├── mcp_server.py    # MCP server
│       │   └── ollama.py
│       └── main.py              # Entry point
├── tools/                       # External tools (llama.cpp)
├── models/                      # Downloaded/converted models
├── datasets/                    # Training datasets
├── config/                      # Configuration files
├── venv/                        # Virtual environment
├── toolbox.bat                  # Windows launcher
├── toolbox.sh                   # Linux/macOS launcher
├── setup.bat                    # Windows setup
├── pyproject.toml               # Project config
├── README.md                    # This file
├── LESMINU.md                   # Finnish documentation
└── CLAUDE_MCP_SETUP.md          # MCP setup guide
```

---

## Quantization Reference

| Type | Bits | Quality | Best For |
|------|------|---------|----------|
| F16 | 16 | Highest | Maximum quality |
| Q8_0 | 8 | Very High | Good balance |
| Q6_K | 6.5 | High | Quality focused |
| Q5_K_M | 5.5 | High | Recommended |
| **Q4_K_M** | **4.5** | **Medium-High** | **Most popular** |
| Q4_K_S | 4.5 | Medium | Smaller files |
| Q3_K_M | 3.5 | Low | Limited RAM |
| Q2_K | 2.5 | Very Low | Extreme compression |

---

## VRAM Requirements

### Inference (GGUF)

| Model Size | Q4_K_M | Q5_K_M | Q8_0 |
|------------|--------|--------|------|
| 7B | ~6 GB | ~7 GB | ~9 GB |
| 13B | ~10 GB | ~12 GB | ~16 GB |
| 30B | ~22 GB | ~26 GB | ~35 GB |
| 70B | ~45 GB | ~52 GB | ~75 GB |

### Merging (Mergekit)

| Models | Recommended VRAM |
|--------|------------------|
| 7B x 2 | 10 GB (with optimizations) |
| 13B x 2 | 16 GB |
| 30B x 2 | 32 GB |

### Training (LoRA)

| Model | QLoRA 4-bit | QLoRA 8-bit | Full |
|-------|-------------|-------------|------|
| 7B | ~8 GB | ~12 GB | ~28 GB |
| 13B | ~12 GB | ~20 GB | ~52 GB |

---

## Portable Mode

AI Toolbox is designed to be portable:

1. Copy the entire `AI Toolbox` folder to a USB drive
2. On the target PC, run `setup.bat` or `toolbox.bat`
3. Dependencies install automatically on first run

**Note:** Target PC must have Python 3.9+ installed

---

## Troubleshooting

### "Python not found"

Install Python 3.9+ from https://python.org and ensure "Add to PATH" is checked.

### "Model not found"

- Verify the HuggingFace model ID is correct (case-sensitive)
- Some models require authentication - set `HF_TOKEN` environment variable

### "Out of memory"

- Use more aggressive quantization (Q4 instead of Q8)
- Enable VRAM optimizations in Mergekit (automatic for <12GB)
- Use QLoRA (4-bit) for training
- Close other applications

### "CUDA out of memory" during merge

Mergekit automatically enables optimizations for 10GB VRAM:
- `--lazy-unpickle` - Deferred model loading
- `--low-cpu-memory` - Intermediate tensors on GPU
- `--out-shard-size 4B` - Smaller output shards

### Architecture mismatch in merge

The Mergekit Wizard automatically detects architecture compatibility:
- Same architecture required for merging
- Different vocab sizes handled automatically

---

## Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## License

GPL-3.0 - see [LICENSE](LICENSE). You may use, modify, and distribute this
software, but any distributed modifications must also be released under the
GPL-3.0.

---

## Credits

AI Toolbox builds on excellent open-source projects. It calls these as
external tools (installed/downloaded separately, not bundled here):

- [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT) - GGUF conversion & quantization
- [mergekit](https://github.com/arcee-ai/mergekit) - model merging
- [Hugging Face](https://huggingface.co) `transformers`, `huggingface-hub`, `safetensors`, `tokenizers` (Apache-2.0) - model download & handling
- [Ollama](https://ollama.com) - local model serving
- [Unsloth](https://github.com/unslothai/unsloth) / [PEFT](https://github.com/huggingface/peft) (Apache-2.0) - LoRA training
- [Rich](https://github.com/Textualize/rich), [Questionary](https://github.com/tmbo/questionary) (MIT) - terminal UI

---

**Made with love for the local AI community**
