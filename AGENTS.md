# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AI Toolbox is a Python CLI application for working with local AI models. It provides tools for downloading, converting to GGUF, quantizing, LoRA training, model merging (via mergekit), abliteration (refusal removal), and running inference via llama.cpp or Ollama.

## Commands

```bash
# Install in development mode
pip install -e .

# Run the application (after install)
ai-toolbox
aitb

# Run without installing
python -m ai_toolbox

# Windows launcher (creates venv if needed)
toolbox.bat

# Linux/macOS launcher
./toolbox.sh
```

## Architecture

### Entry Point Flow
`main.py` → `cli/app.py:AIToolbox` → creates all services → routes menu choices to command handlers

### Package Structure
- **core/** - Foundation: `paths.py` (singleton path manager), `ui.py` (Rich theming), `config.py`, `exceptions.py`
- **cli/** - Command handlers: each `*_cmd.py` file is a menu subsystem (ModelHubCommands, GGUFToolsCommands, etc.)
- **models/** - Model library management and HuggingFace downloading
- **conversion/** - GGUF conversion and quantization via llama.cpp
- **training/** - LoRA training with Unsloth/PEFT, dataset preparation
- **merging/** - Model merging via mergekit with presets and YAML config
- **abliteration/** - Refusal direction removal using activation analysis
- **inference/** - Chat interface and benchmarking via llama-cpp-python
- **integrations/** - Ollama integration and MCP server

### Key Patterns

**Path Management**: All paths go through `core/paths.py:ToolboxPaths` singleton. Detects root via `AITOOLBOX_ROOT` env var or by finding `src/ai_toolbox/`. Model subdirectories: `models/safetensors/`, `models/gguf/`, `models/merged/`, `models/abliterated/`, `models/lora/adapters/`.

**Service Injection**: `cli/app.py:AIToolbox.__init__()` creates service instances (ModelLibrary, ModelDownloader, GGUFConverter, etc.) and passes them to command handlers.

**UI Consistency**: Orange/amber branding throughout. Use `core/ui.py` exports: `console`, `print_success/error/warning/info`, `MENU_STYLE` for questionary, `print_mini_banner()` for tool headers.

**Backward Compatibility**: `__init__.py` re-exports all public classes for code using the old flat import structure.

## External Tools

- **llama.cpp**: Located at `tools/llama.cpp/`. Used for GGUF conversion (`convert_hf_to_gguf.py`) and quantization (`llama-quantize`).
- **mergekit**: External pip package. Wrapper in `merging/mergekit_wrapper.py`.

## UI Language

Interface text is in Finnish with English technical terms. Example: "Lataa malli" (Download model), "GGUF Tools", "Training Center".

---

## Learnings & Common Mistakes

### Path Management
- **ALWAYS** use `get_paths()` from `core/paths.py`
- **NEVER** hardcode paths like `"D:/AI Toolbox/models/"`
- Use appropriate property: `paths.safetensors_dir`, `paths.gguf_dir`, `paths.abliterated_dir`, etc.

```python
# CORRECT
from ..core.paths import get_paths
paths = get_paths()
output_path = paths.abliterated_dir / model_name

# WRONG
output_path = Path("D:/AI Toolbox/models/abliterated") / model_name
```

### Service Pattern
All services follow this structure:
```python
class ServiceName:
    def __init__(self):
        self.paths = get_paths()
        self._check_dependencies()

    def operation(self, config, progress_callback: Optional[Callable] = None):
        """Main operation with optional progress callback."""
        if progress_callback:
            progress_callback("Starting...", 0.0)
        # ... work ...
        if progress_callback:
            progress_callback("Complete!", 1.0)
```

### UI Patterns
```python
# Import from core/ui.py - NEVER create new Console()
from ..core.ui import console, print_success, print_error, print_warning, MENU_STYLE, format_menu_item

# Menu items: English term (24 chars), Finnish description
format_menu_item("SLERP Merge", "Yhdistä kaksi mallia")

# Messages - use helpers, not raw print
print_success("Valmis!")      # Not: print("[green]Done![/green]")
print_error("Virhe: ...")     # Not: print("[red]Error[/red]")
print_warning("Varoitus...")  # Not: print("[yellow]Warning[/yellow]")
```

### Common Mistakes to Avoid
1. **Business logic in CLI commands** - CLI files should only handle user interaction, delegate to services
2. **Creating new Console()** - Use shared `console` from core/ui.py
3. **Forgetting AIToolbox registration** - New services must be added to `cli/app.py:AIToolbox.__init__()`
4. **Using print() directly** - Use `print_success/error/warning/info` for consistent styling
5. **Hardcoding model paths** - Always use ToolboxPaths properties
6. **Ignoring progress_callback=None** - Always check before calling: `if progress_callback: progress_callback(...)`
7. **Missing __init__.py exports** - Public classes must be exported for backward compatibility

---

## Common Workflows

### Full Abliteration Pipeline
```
1. Model Hub → Lataa HuggingFacesta     (download SafeTensors)
2. Training Center → Abliteration        (remove refusal)
3. GGUF Tools → Muunna GGUF:ksi         (convert to GGUF)
4. GGUF Tools → Kvantisoi               (quantize Q4_K_M/Q8_0)
5. Ollama Manager → Luo malli           (create Ollama model)
```

### LoRA Training Workflow
```
1. Training Center → Dataset Preparation  (prepare/convert dataset)
2. Training Center → LoRA Training        (configure & train)
3. Training Center → Merge Adapter        (merge into base model)
4. GGUF Tools → Muunna & Kvantisoi       (convert to GGUF)
```

### Model Merging Workflow
```
1. Model Hub → Download both models      (SafeTensors format)
2. Training Center → Model Merge         (SLERP or TIES)
3. GGUF Tools → Convert & Quantize       (to GGUF)
```

---

## Architecture Decisions

### Why Service Injection?
`AIToolbox` creates all service instances in `__init__` and passes them to command handlers:
- **Testing**: Easy to mock services for unit tests
- **Lifecycle**: Single instance per service, consistent state
- **Dependencies**: Clear dependency graph, no circular imports

### Why ToolboxPaths Singleton?
- **Single source of truth** for all paths
- **Auto-creates directories** on first access
- **Portable**: Supports `AITOOLBOX_ROOT` env var for relocation
- **Consistent**: All modules use same path resolution

### Why Finnish + English?
- **Target audience**: Finnish developers working with local AI
- **Technical terms in English**: "SLERP", "LoRA", "GGUF" - universal terms
- **Descriptions in Finnish**: "Yhdistä kaksi mallia" - user-friendly
- **Format**: `format_menu_item("Technical Term", "Finnish description")`

### Why Rich + Questionary?
- **Rich**: Beautiful tables, panels, progress bars with consistent theming
- **Questionary**: Keyboard-navigable menus with our orange brand color (#ff9d00)
- **Shared console**: Single Rich Console instance for consistent output

---

## Module Responsibilities

| Module | Responsibility | NOT Allowed |
|--------|---------------|-------------|
| `core/` | Foundation (paths, ui, config, exceptions) | Business logic |
| `cli/` | User interaction, menu routing | Business logic, direct file ops |
| `models/` | Library management, HF downloading | Training, conversion |
| `conversion/` | GGUF conversion, quantization | Training, merging |
| `training/` | LoRA training, dataset prep | Conversion, merging |
| `merging/` | Model merging (SLERP, TIES) | Training, conversion |
| `abliteration/` | Refusal removal | Training, conversion |
| `inference/` | Chat, benchmarking | Model modification |
| `integrations/` | Ollama, MCP server | Core functionality |
