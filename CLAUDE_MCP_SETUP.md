# AI Toolbox MCP Server Setup for Claude Code

> **Version 3.0.0** - MCP integration for AI Toolbox

This guide explains how to configure Claude Code to use AI Toolbox as an MCP (Model Context Protocol) server.

## Prerequisites

1. **Python 3.9+** with AI Toolbox installed
2. **MCP library** installed: `pip install mcp`
3. **Claude Code** CLI installed
4. **Git** (for llama.cpp setup)

## Installation

### 1. Install dependencies

```bash
cd /path/to/AI-Toolbox
pip install -e .
pip install mcp
```

### 2. Configure Claude Code

Add the following to your Claude Code MCP configuration file.

**On Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**On macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ai-toolbox": {
      "command": "python",
      "args": ["-m", "ai_toolbox.mcp_server"],
      "cwd": "/path/to/AI-Toolbox",
      "env": {
        "PYTHONPATH": "/path/to/AI-Toolbox/src"
      }
    }
  }
}
```

### 3. Restart Claude Code

After adding the configuration, restart Claude Code to load the MCP server.

## Available Tools (20+ total)

### Model Search & Download

| Tool | Description |
|------|-------------|
| `search_models` | Search HuggingFace Hub for models |
| `get_model_info` | Get detailed model information |
| `download_model` | Download model from HuggingFace |
| `check_downloaded` | Check if model is already downloaded |

### Model Library

| Tool | Description |
|------|-------------|
| `list_library` | List all models in local library |
| `library_stats` | Get library statistics |
| `cleanup_library` | Remove duplicates and missing files |

### GGUF Conversion & Quantization

| Tool | Description |
|------|-------------|
| `convert_to_gguf` | Convert HF model to GGUF format |
| `quantize_gguf` | Quantize GGUF model to smaller size |
| `convert_and_quantize` | Convert and quantize in one step (recommended) |
| `list_quantization_types` | List all available quantization types |
| `recommend_quantization` | Get recommended quantization for your system |
| `estimate_model_size` | Estimate output size after conversion |

### Model Merging (Mergekit)

| Tool | Description |
|------|-------------|
| `merge_models` | Merge models using Mergekit |
| `list_merge_methods` | List available merge methods |
| `list_merge_presets` | List ready-to-use merge presets |
| `check_merge_compatibility` | Check if models can be merged |

### Converter Setup

| Tool | Description |
|------|-------------|
| `check_converter_status` | Check if llama.cpp is installed |
| `setup_llama_cpp` | Download and set up llama.cpp |

### Utilities

| Tool | Description |
|------|-------------|
| `calculate_vram` | Calculate VRAM requirements |
| `get_system_info` | Get system RAM/CPU info |

## Tool Details

### `search_models`
Search for AI models on HuggingFace Hub.
```
Parameters:
- query (required): Search query (e.g., "llama", "mistral", "qwen")
- limit: Maximum results (default: 10)
- task: Filter by task type ("text-generation", "text2text-generation", "feature-extraction")
```

### `download_model`
Download a model from HuggingFace Hub.
```
Parameters:
- model_id (required): HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
- safetensors_only: Download only safetensors files (default: true)
```

### `convert_and_quantize`
Convert HF model to GGUF and quantize in one step. **This is the recommended way to create GGUF models.**
```
Parameters:
- model_path (required): Path to HuggingFace model directory
- quantization: Target quantization (default: "q4_k_m")
  Options: q8_0, q6_k, q5_k_m, q5_k_s, q4_k_m, q4_k_s, q4_0, q3_k_m, q3_k_s, q2_k
- keep_f16: Keep intermediate F16 file (default: false)
```

### `quantize_gguf`
Quantize an existing GGUF model to a smaller size.
```
Parameters:
- input_path (required): Path to input GGUF file
- quantization: Target quantization (default: "q4_k_m")
```

### `merge_models`
Merge multiple models using Mergekit.
```
Parameters:
- models (required): List of model paths to merge
- method: Merge method (default: "slerp")
  Options: slerp, dare_ties, dare_linear, ties, della
- output_name: Name for merged model
- t: Interpolation factor for SLERP (default: 0.5)
- density: Density for DARE methods (default: 0.5)
```

### `list_merge_presets`
Get ready-to-use merge configurations.
```
Available presets:
- slerp_balanced: Balanced 2-model merge (SLERP t=0.5)
- slerp_light_finetune: Light finetune weight (SLERP t=0.3)
- dare_ties_language: Language model merging
- reasoning_r1_style: Reasoning optimized (SLERP t=0.35)
- della_efficient: Pruning + merge
```

### `recommend_quantization`
Get recommended quantization types based on model size and available RAM.
```
Parameters:
- model_params_billion (required): Model parameters in billions (e.g., 7, 13, 70)
- available_ram_gb: Available RAM in GB (auto-detected if not provided)
```

### `setup_llama_cpp`
Download and set up llama.cpp for GGUF conversion.
```
Parameters:
- force: Force re-download even if exists (default: false)
```

## Example Usage in Claude

Once configured, you can ask Claude things like:

### Search & Download
- "Search for Qwen models on HuggingFace"
- "Download the mistralai/Mistral-7B-v0.1 model"
- "Is llama-2-7b already downloaded?"

### GGUF Conversion
- "Convert my downloaded Qwen model to GGUF with Q4_K_M quantization"
- "What quantization types are available?"
- "Recommend a quantization for a 13B model on my system"
- "Set up llama.cpp for me"

### Model Merging
- "List available merge methods"
- "Check if these two models can be merged"
- "Merge Model1 and Model2 using SLERP with t=0.4"
- "Show me the merge presets"
- "Use the reasoning_r1_style preset to merge my models"

### Library & Info
- "What's in my model library?"
- "Show me only GGUF models in my library"
- "How much VRAM do I need for a 70B model?"
- "Clean up my library (remove duplicates)"

## Typical Workflows

### Workflow 1: Download and Convert

1. **Search for a model:**
   "Search for small LLMs under 3B parameters"

2. **Download the model:**
   "Download Qwen/Qwen2.5-0.5B"

3. **Check converter status:**
   "Is the GGUF converter ready?"

4. **Set up converter (if needed):**
   "Set up llama.cpp"

5. **Convert and quantize:**
   "Convert the Qwen model I just downloaded to GGUF with Q4_K_M"

6. **Verify in library:**
   "Show me my GGUF models"

### Workflow 2: Merge Models

1. **Check compatibility:**
   "Can I merge Model1 and Model2?"

2. **List methods:**
   "What merge methods are available?"

3. **Merge models:**
   "Merge Model1 and Model2 using DARE-TIES"

4. **Verify result:**
   "Show the merged model in my library"

### Workflow 3: Library Management

1. **Check library:**
   "What's in my model library?"

2. **Find duplicates:**
   "Are there any duplicate models?"

3. **Cleanup:**
   "Clean up my library"

## Troubleshooting

### MCP server not connecting
1. Ensure Python and MCP are installed correctly
2. Check that PYTHONPATH points to the correct directory
3. Try running manually: `python -m ai_toolbox.mcp_server`

### Import errors
Make sure AI Toolbox is installed:
```bash
cd /path/to/AI-Toolbox
pip install -e .
pip install mcp psutil
```

### GGUF conversion fails
1. Run `check_converter_status` to see if llama.cpp is installed
2. Run `setup_llama_cpp` to install it
3. For quantization, you may need to build llama.cpp:
   ```bash
   cd ~/.ai-toolbox/llama.cpp
   make llama-quantize
   ```

### Merge fails with architecture mismatch
- Only models with the same architecture can be merged
- Use `check_merge_compatibility` before merging
- Different vocab sizes are handled automatically

### Windows path issues
Use forward slashes in paths:
- `C:/path/to/AI-Toolbox` (recommended)
- `C:\\path\\to\\AI-Toolbox` (escaped backslashes)

## File Locations

- **Downloads:** `~/.ai-toolbox/downloads/`
- **GGUF Models:** `~/.ai-toolbox/gguf_models/`
- **Model Library:** `~/.ai-toolbox/models/`
- **Merged Models:** `~/.ai-toolbox/merged/`
- **llama.cpp:** `~/.ai-toolbox/llama.cpp/`
- **Merge Configs:** `~/.ai-toolbox/merge_configs/`

## Available Merge Methods

| Method | Models | Description |
|--------|--------|-------------|
| SLERP | 2 | Spherical linear interpolation (most popular) |
| DARE-TIES | 2+ | Advanced pruning + TIES (recommended) |
| DARE-LINEAR | 2+ | Linear DARE variant |
| TIES | 2+ | Task-specific interpolation |
| DELLA | 2+ | Efficient pruning method |

## Quantization Reference

| Type | Bits | Quality | Best For |
|------|------|---------|----------|
| Q8_0 | 8 | Very High | Good balance |
| Q6_K | 6.5 | High | Quality focused |
| Q5_K_M | 5.5 | High | Recommended |
| **Q4_K_M** | **4.5** | **Medium-High** | **Most popular** |
| Q4_K_S | 4.5 | Medium | Smaller files |
| Q3_K_M | 3.5 | Low | Limited RAM |
| Q2_K | 2.5 | Very Low | Extreme compression |
