"""
AI TOOLBOX - Model Library
==========================

Manage and browse your local AI models.
"""

import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..core.paths import get_paths, get_models_dir, get_library_file
from ..core.ui import format_size
from .types import ModelEntry, ModelTreeNode, ModelCategory

console = Console()


# Category icons for display
CATEGORY_ICONS = {
    "base": "🏠",
    "adapter": "🔧",
    "merged": "🔀",
    "ollama": "🤖",
}


# ============================================================================
# NAME FORMATTING UTILITIES
# ============================================================================

# Tokens dropped from descriptive display names - architecture / format noise
# that does not help tell two local models apart.
_DISPLAY_DROP_TOKENS = {"llama", "meta", "hf"}
_DISPLAY_ARCH_TOKENS = {"llama", "meta"}

# Descriptor tokens normalized to short, consistent labels. These are the parts
# that actually distinguish two variants of the same base model (Instruct vs
# SFT, chat vs base, etc.).
_DISPLAY_DESCRIPTORS = {
    "instruct": "Instr", "instr": "Instr", "it": "Instr",
    "sft": "SFT", "dpo": "DPO", "rlhf": "RLHF",
    "base": "Base", "chat": "Chat",
}

_DISPLAY_ORG_PREFIXES = (
    "LumiOpen_", "aifeifei798_", "meta-llama_", "mistralai_",
    "Qwen_", "deepseek-ai_", "NousResearch_", "teknium_",
    "Open-Orca_", "WizardLM_", "lmsys_", "THUDM_",
    "tiiuae_", "bigscience_", "EleutherAI_", "microsoft_",
    "google_", "facebook_", "stabilityai_", "databricks_",
)


def _descriptive_identity(name: str) -> str:
    """Build a display identity that keeps the parts which distinguish
    variants of the same base model.

    Unlike extract_model_identity (which collapses to the shortest unique
    token and is used for terse merge names), this preserves family version,
    training type and modifier tags so that e.g. "Poro-2-Instr-Ablit" and
    "Poro-2-Ablit-v2" stay distinct in selection menus.
    """
    # Strip a leading organization prefix (case-insensitive).
    for prefix in _DISPLAY_ORG_PREFIXES:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
            break

    out: List[str] = []
    prev_low = ""
    for token in re.split(r'[-_]', name):
        if not token:
            continue
        low = token.lower()

        # Size tags like "8B" / "1.5B" carry no distinguishing value.
        if re.fullmatch(r'\d+(\.\d+)?b', low):
            prev_low = low
            continue
        # Architecture / format noise.
        if low in _DISPLAY_DROP_TOKENS:
            prev_low = low
            continue
        # A bare number right after an architecture word is an arch version
        # (e.g. Llama-3.1) -> drop. After a family word (Poro-2) -> keep.
        if re.fullmatch(r'\d+(\.\d+)?', low) and prev_low in _DISPLAY_ARCH_TOKENS:
            prev_low = low
            continue

        # Normalize known descriptor / modifier tokens to short labels.
        if low.startswith("abliter"):
            value = "Ablit"
        elif low.startswith("uncens"):
            value = "Uncens"
        elif low in _DISPLAY_DESCRIPTORS:
            value = _DISPLAY_DESCRIPTORS[low]
        else:
            value = token

        if value.lower() not in [o.lower() for o in out]:
            out.append(value)
        prev_low = low

    if not out:
        # Nothing meaningful survived - fall back to the terse identity.
        return extract_model_identity(name)

    result = "-".join(out)
    if result[0].islower():
        result = result[0].upper() + result[1:]
    return result


def format_display_name(
    full_name: str,
    max_length: int = 40,
    include_quant: bool = False
) -> str:
    """
    Create a clean, readable display name from a full model name.

    Args:
        full_name: The full model name
        max_length: Maximum length for display
        include_quant: Whether to include quantization in name (False if shown separately)

    Returns:
        Shortened, cleaned display name
    """
    name = full_name

    # Strip path components
    if "/" in name:
        name = name.split("/")[-1]
    if "\\" in name:
        name = name.split("\\")[-1]

    # Remove file extension
    if name.endswith(".gguf"):
        name = name[:-5]

    # Extract and remove quantization suffix
    quant_suffix = ""
    detected_quant = ""

    # Pattern 1: Bracketed format " [Q8_0]" or "[Q8_0]"
    bracket_match = re.search(r'\s*\[([QFqf]\d+[_]?\d*[KkMmSsLl]*)\]$', name)
    if bracket_match:
        detected_quant = bracket_match.group(1).upper()
        name = name[:bracket_match.start()]

    # Pattern 2: Suffix format "-q8_0" or "_q8_0"
    if not detected_quant:
        quant_patterns = [
            "-q8_0", "-q6_k", "-q5_k_m", "-q5_k_s", "-q4_k_m", "-q4_k_s",
            "-q4_0", "-q3_k_m", "-q3_k_s", "-q2_k", "-f16", "-f32",
            "_q8_0", "_q6_k", "_q5_k_m", "_q5_k_s", "_q4_k_m", "_q4_k_s",
            "_q4_0", "_q3_k_m", "_q3_k_s", "_q2_k", "_f16", "_f32",
            "-iq4_nl", "-iq4_xs", "-iq3_s", "-iq3_m", "-iq3_xs",
            "-iq2_s", "-iq2_xs", "-iq1_s", "-iq1_m",
        ]
        for pattern in quant_patterns:
            if name.lower().endswith(pattern):
                detected_quant = pattern[1:].upper()
                name = name[:-len(pattern)]
                break

    # Add quant suffix only if requested
    if detected_quant and include_quant:
        quant_suffix = f" [{detected_quant}]"

    # Build a descriptive identity that keeps variant-distinguishing parts
    # (training type, family version, abliterated/uncensored tags).
    clean_name = _descriptive_identity(name)

    # Check if it's a merge (has method prefix)
    merge_prefixes = {
        "slerp_": "SLERP",
        "dare_ties_": "DARE",
        "dare_linear_": "DARE-L",
        "dare_": "DARE",
        "ties_": "TIES",
        "linear_": "LINEAR",
        "della_": "DELLA",
        "merged_": "Merged",
        "advanced_": "ADV",
    }

    method = None
    name_lower = name.lower()
    for prefix, method_name in merge_prefixes.items():
        if name_lower.startswith(prefix):
            method = method_name
            # Extract model parts after prefix
            rest = name[len(prefix):]

            # Handle various separators: _, -, +, " + "
            # First normalize "+" with spaces
            rest = re.sub(r'\s*\+\s*', '_', rest)

            # Split by common separators
            parts = re.split(r'[-_]', rest)

            # Filter and extract identities
            skip_words = {
                'and', 'more', 'abliterated', 'abliter', 'uncensored',
                'instruct', 'chat', 'base', 'sft', 'hf', 'llama',
                '8b', '7b', '13b', '14b', '32b', '70b', '3', '3.1', '2'
            }

            identities = []
            for part in parts:
                part_clean = part.strip()
                # Skip short parts and common words
                if len(part_clean) < 2 or part_clean.lower() in skip_words:
                    continue
                # Skip parts that are just numbers
                if part_clean.isdigit():
                    continue

                ident = extract_model_identity(part_clean)
                # Only add if meaningful and not duplicate
                if ident and len(ident) >= 2 and ident.lower() not in [i.lower() for i in identities]:
                    identities.append(ident)

            if identities:
                clean_name = "-".join(identities[:3])
                if len(identities) > 3:
                    clean_name += f"+{len(identities)-3}"
            break

    # Build final name
    if method:
        final_name = f"{method}: {clean_name}"
    else:
        final_name = clean_name

    # Add quant suffix if requested
    final_name += quant_suffix

    # Truncate if needed
    if len(final_name) > max_length:
        final_name = final_name[:max_length - 3] + "..."

    return final_name


def extract_model_identity(full_name: str) -> str:
    """
    Extract the unique identifying name from a full model name.

    Examples:
        "LumiOpen_Llama-Poro-2-8B-SFT-abliterated" -> "Poro"
        "DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored" -> "DarkIdol"
        "aifeifei798_DarkIdol-Llama-3.1-8B" -> "DarkIdol"
        "meta-llama/Llama-2-7b-chat-hf" -> "Llama2"
        "mistralai/Mistral-7B-Instruct-v0.2" -> "Mistral"
        "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" -> "DeepSeek-R1"

    Args:
        full_name: The full model name/path

    Returns:
        Short, unique identifier for the model
    """
    # Start with the base name
    name = full_name

    # Remove path components
    if "/" in name:
        name = name.split("/")[-1]
    if "\\" in name:
        name = name.split("\\")[-1]

    # Remove file extension
    name = Path(name).stem if "." in name else name

    # Remove common organization prefixes (at start of name)
    org_prefixes = [
        "LumiOpen_", "aifeifei798_", "meta-llama_", "mistralai_",
        "Qwen_", "deepseek-ai_", "NousResearch_", "teknium_",
        "Open-Orca_", "WizardLM_", "lmsys_", "THUDM_",
        "tiiuae_", "bigscience_", "EleutherAI_", "microsoft_",
        "google_", "facebook_", "stabilityai_", "databricks_",
    ]
    for prefix in org_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Remove common suffixes (order matters - longer first)
    suffixes_to_remove = [
        # Quantization
        "-abliterated", "_abliterated", "-abliter", "_abliter",
        "-Uncensored", "_Uncensored", "-uncensored", "_uncensored",
        # Size indicators
        "-8B", "-7B", "-13B", "-14B", "-32B", "-70B", "-72B",
        "-1B", "-3B", "-0.5B", "-1.5B", "-2B", "-4B",
        "_8B", "_7B", "_13B", "_14B", "_32B", "_70B", "_72B",
        # Training type
        "-Instruct", "-Chat", "-Base", "-SFT", "-RLHF", "-DPO",
        "_Instruct", "_Chat", "_Base", "_SFT", "_RLHF", "_DPO",
        "-instruct", "-chat", "-base", "-sft",
        # Version indicators
        "-hf", "_hf", "-HF", "_HF",
        "-v0.1", "-v0.2", "-v0.3", "-v1", "-v2", "-v3",
        "-1.0", "-1.1", "-1.2", "-2.0", "-3.0", "-3.1", "-3.2",
        # Llama architecture indicators (keep but will handle specially)
        "-Llama-3.1", "-Llama-3", "-Llama-2", "-Llama",
        "_Llama-3.1", "_Llama-3", "_Llama-2", "_Llama",
    ]

    for suffix in suffixes_to_remove:
        if suffix.lower() in name.lower():
            idx = name.lower().find(suffix.lower())
            name = name[:idx] + name[idx + len(suffix):]

    # Remove trailing version numbers like "-1", "-2", "-1.2" at end of name
    name = re.sub(r'-\d+\.?\d*$', '', name)

    # Handle special known model names to extract the key identity
    # These are patterns where the unique name is embedded
    special_patterns = [
        # Pattern: "Something-Llama-..." -> "Something"
        (r'^([A-Za-z0-9]+)-Llama', r'\1'),
        # Pattern: "Llama-Something-..." -> "Something" (if Something is not a version)
        (r'^Llama-([A-Za-z][A-Za-z0-9]*)', r'\1'),
        # Pattern: "Something-Mistral-..." -> "Something"
        (r'^([A-Za-z0-9]+)-Mistral', r'\1'),
        # Pattern: "Qwen2.5" or "Qwen2" -> keep as is
        (r'^(Qwen\d+\.?\d*)', r'\1'),
        # Pattern: "DeepSeek-R1" -> keep as is
        (r'^(DeepSeek-R\d+)', r'\1'),
        # Pattern: "Mistral-..." -> "Mistral"
        (r'^(Mistral)', r'\1'),
    ]

    for pattern, replacement in special_patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            name = match.expand(replacement)
            break

    # Clean up any remaining dashes/underscores at edges
    name = name.strip("-_")

    # If still too long or contains numbers at end, try to simplify
    if len(name) > 15:
        # Remove trailing numbers and dashes
        name = re.sub(r'[-_]?\d+[-_]?\d*$', '', name)

    # Final cleanup - capitalize first letter
    if name and name[0].islower():
        name = name[0].upper() + name[1:]

    # If we ended up with nothing useful, use first word
    if not name or len(name) < 2:
        parts = full_name.replace("-", "_").split("_")
        for part in parts:
            if len(part) >= 3 and not part.isdigit():
                name = part
                break

    return name or "Model"


def generate_merge_name(
    method: str,
    model_names: list,
    base_model_name: str = None,
    short: bool = True
) -> str:
    """
    Generate a clean default name for a merged model.

    Examples:
        SLERP + ["Poro-8B", "DarkIdol-8B"] -> "Slerp_Poro-DarkIdol"
        DARE_TIES + ["Model1", "Model2", "Model3"] -> "Dare_Model1-Model2-Model3"

    Args:
        method: Merge method (slerp, dare_ties, etc.)
        model_names: List of model names being merged
        base_model_name: Name of the base model (for DARE/TIES)
        short: If True, use abbreviated names

    Returns:
        A clean, descriptive merge name like "Slerp_Poro-DarkIdol"
    """
    # Extract clean identities from all models
    if short:
        identities = [extract_model_identity(m) for m in model_names]
    else:
        identities = [Path(m).stem for m in model_names]

    # Remove duplicates while preserving order
    seen = set()
    unique_identities = []
    for ident in identities:
        if ident.lower() not in seen:
            seen.add(ident.lower())
            unique_identities.append(ident)
    identities = unique_identities

    # Method prefix - capitalize nicely
    method_names = {
        "slerp": "Slerp",
        "dare_ties": "Dare",
        "dare_linear": "DareL",
        "ties": "Ties",
        "linear": "Linear",
        "della": "Della",
        "task_arithmetic": "Task",
    }
    method_prefix = method_names.get(method.lower(), method.capitalize()[:6])

    # Build name with hyphen separator between models
    if len(identities) >= 2:
        models_part = "-".join(identities[:4])  # Max 4 models in name
        if len(identities) > 4:
            models_part += f"-+{len(identities)-4}"
        return f"{method_prefix}_{models_part}"
    elif identities:
        return f"{method_prefix}_{identities[0]}"
    else:
        return f"{method_prefix}_Merge"


def get_sort_key(model: 'ModelEntry', sort_by: str = "date"):
    """
    Get sort key for a model based on sort criteria.

    Args:
        model: ModelEntry to get sort key for
        sort_by: One of 'date', 'name', 'size', 'quant', 'format'

    Returns:
        Sort key value
    """
    if sort_by == "date":
        return model.added_date or "0000-00-00"
    elif sort_by == "name":
        return model.name.lower()
    elif sort_by == "size":
        return model.size_bytes
    elif sort_by == "quant":
        # Sort by quantization level (larger = better quality)
        quant_order = {
            "F32": 100, "F16": 90, "BF16": 85,
            "Q8_0": 80, "Q6_K": 70, "Q5_K_M": 65, "Q5_K_S": 60,
            "Q4_K_M": 55, "Q4_K_S": 50, "Q4_0": 45,
            "Q3_K_M": 40, "Q3_K_S": 35, "Q2_K": 30,
        }
        return quant_order.get(model.quantization, 0) if model.quantization else 0
    elif sort_by == "format":
        format_order = {"safetensors": 1, "pytorch": 2, "gguf": 3}
        return format_order.get(model.format, 99)
    else:
        return model.added_date or "0000-00-00"


class ModelLibrary:
    """Manages a library of AI models."""

    def __init__(self, library_path: Optional[str] = None, auto_scan: bool = True):
        """
        Initialize the model library.

        Args:
            library_path: Path to the library directory (uses portable path if not specified)
            auto_scan: Whether to auto-scan for models if the index is empty
        """
        if library_path:
            self.library_path = Path(library_path)
            self.index_file = self.library_path / "library_index.json"
        else:
            self.library_path = get_models_dir()
            self.index_file = get_library_file()

        self.library_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelEntry] = {}
        self._load_index()

        if auto_scan and not self._models:
            self._auto_scan()

    def _load_index(self):
        """Load the library index from disk (falls back to .bak if corrupt)."""
        self._models = {}

        for candidate in (self.index_file, self.index_file.with_suffix('.json.bak')):
            if not candidate.exists():
                continue
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for model_id, model_data in data.get('models', {}).items():
                    self._models[model_id] = ModelEntry(**model_data)
                if candidate != self.index_file:
                    console.print("[yellow]Kirjastoindeksi palautettu varmuuskopiosta (.bak)[/yellow]")
                return
            except (json.JSONDecodeError, TypeError) as e:
                console.print(f"[yellow]Warning: Could not load {candidate.name}: {e}[/yellow]")
                self._models = {}

    def _save_index(self):
        """Save the library index to disk atomically (temp file + replace).

        library.json is the single source of truth for the whole library -
        a crash mid-write must never corrupt it. The previous version is
        kept as library.json.bak for recovery.
        """
        data = {
            'version': '1.0',
            'updated': datetime.now().isoformat(),
            'models': {mid: asdict(model) for mid, model in self._models.items()}
        }

        temp_file = self.index_file.with_suffix('.json.tmp')
        backup_file = self.index_file.with_suffix('.json.bak')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if self.index_file.exists():
                shutil.copy2(self.index_file, backup_file)
            temp_file.replace(self.index_file)
        except OSError as e:
            if temp_file.exists():
                temp_file.unlink()
            console.print(f"[red]Virhe tallennettaessa kirjastoindeksia: {e}[/red]")
            raise

    def _auto_scan(self):
        """Automatically scan the models directory for existing models."""
        found = self.scan_directory(str(self.library_path), add_found=True)
        if found:
            console.print(f"[green]Auto-discovered {len(found)} model(s) in library[/green]")

    def _generate_id(self, name: str) -> str:
        """Generate a unique ID for a model."""
        base_id = name.lower().replace('/', '_').replace(' ', '_')
        counter = 1
        model_id = base_id
        while model_id in self._models:
            model_id = f"{base_id}_{counter}"
            counter += 1
        return model_id

    def _detect_format(self, path: Path) -> str:
        """Detect the format of a model."""
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix == '.gguf':
                return 'gguf'
            elif suffix == '.ggml':
                return 'ggml'
            elif suffix == '.safetensors':
                return 'safetensors'
            elif suffix in ['.bin', '.pt', '.pth']:
                return 'pytorch'
        elif path.is_dir():
            if (path / 'model.safetensors').exists() or list(path.glob('*.safetensors')):
                return 'safetensors'
            elif (path / 'pytorch_model.bin').exists() or list(path.glob('*.bin')):
                return 'pytorch'
            elif list(path.glob('*.gguf')):
                return 'gguf'
        return 'unknown'

    def _detect_quantization(self, path: Path) -> Optional[str]:
        """Detect quantization from filename."""
        name = path.stem.lower()
        quant_types = [
            'f32', 'f16', 'bf16',
            'q8_0', 'q6_k', 'q5_k_m', 'q5_k_s', 'q5_0', 'q5_1',
            'q4_k_m', 'q4_k_s', 'q4_0', 'q4_1',
            'q3_k_l', 'q3_k_m', 'q3_k_s', 'q2_k',
            'iq4_nl', 'iq4_xs', 'iq3_s', 'iq3_m', 'iq3_xs',
            'iq2_s', 'iq2_xs', 'iq1_s', 'iq1_m'
        ]
        for qt in quant_types:
            if qt in name or qt.replace('_', '-') in name:
                return qt.upper()
        return None

    def _calculate_size(self, path: Path) -> int:
        """Calculate total size of a model."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
            return total
        return 0

    def add_model(
        self,
        path: str,
        name: Optional[str] = None,
        source: str = "local",
        source_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        copy_to_library: bool = False,
        category: str = "base",
        parent_id: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        merge_info: Optional[Dict[str, Any]] = None,
        abliteration_info: Optional[Dict[str, Any]] = None,
        ollama_info: Optional[Dict[str, Any]] = None,
    ) -> ModelEntry:
        """
        Add a model to the library.

        Args:
            path: Path to the model file or directory
            name: Display name for the model
            source: Source type (local, huggingface, converted, merged, trained, ollama)
            source_id: Original source ID (e.g., HuggingFace model ID)
            tags: List of tags for the model
            copy_to_library: Whether to copy files to the library directory
            category: Model category (base, adapter, merged, ollama)
            parent_id: ID of the parent model (for adapters/merged)
            training_info: Training metadata (for LoRA adapters)
            merge_info: Merge metadata (for merged models)
            abliteration_info: Abliteration metadata (for abliterated models)
            ollama_info: Ollama metadata (for Ollama models)

        Returns:
            The created ModelEntry
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Check for duplicates - same path already in library
        normalized_path = str(model_path.absolute())
        for existing_id, existing_model in self._models.items():
            try:
                if Path(existing_model.path).absolute() == model_path.absolute():
                    # Already exists - return existing entry instead of creating duplicate
                    console.print(f"[dim]Malli on jo kirjastossa: {existing_model.name}[/dim]")
                    return existing_model
            except Exception:
                pass

        if not name:
            name = model_path.stem if model_path.is_file() else model_path.name

        if copy_to_library:
            dest_path = self.library_path / model_path.name
            if model_path.is_file():
                shutil.copy2(model_path, dest_path)
            else:
                shutil.copytree(model_path, dest_path, dirs_exist_ok=True)
            model_path = dest_path

        model_id = self._generate_id(name)
        entry = ModelEntry(
            id=model_id,
            name=name,
            source=source,
            source_id=source_id,
            path=str(model_path.absolute()),
            format=self._detect_format(model_path),
            size_bytes=self._calculate_size(model_path),
            quantization=self._detect_quantization(model_path),
            added_date=datetime.now().isoformat(),
            tags=tags or [],
            metadata={},
            category=category,
            parent_id=parent_id,
            children_ids=[],
            training_info=training_info,
            merge_info=merge_info,
            abliteration_info=abliteration_info,
            ollama_info=ollama_info,
        )

        self._models[model_id] = entry

        # Update parent's children_ids if parent exists
        if parent_id and parent_id in self._models:
            parent = self._models[parent_id]
            if model_id not in parent.children_ids:
                parent.children_ids.append(model_id)

        self._save_index()
        return entry

    def remove_model(self, model_id: str, delete_files: bool = False) -> bool:
        """Remove a model from the library."""
        if model_id not in self._models:
            return False

        model = self._models[model_id]

        if delete_files:
            model_path = Path(model.path)
            if model_path.exists():
                if model_path.is_file():
                    model_path.unlink()
                else:
                    shutil.rmtree(model_path)

        del self._models[model_id]
        self._save_index()
        return True

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get a model by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        format_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        sort_by: str = "date",
        reverse: bool = True,
    ) -> List[ModelEntry]:
        """
        List models with optional filters and sorting.

        Args:
            format_filter: Filter by format (gguf, safetensors, etc.)
            source_filter: Filter by source (huggingface, local, merged, etc.)
            tag_filter: Filter by tag
            category_filter: Filter by category (base, adapter, merged, ollama)
            sort_by: Sort by 'date', 'name', 'size', 'quant', or 'format'
            reverse: Reverse sort order (default True for newest first)

        Returns:
            Sorted, filtered list of ModelEntry objects
        """
        models = list(self._models.values())

        if format_filter:
            models = [m for m in models if m.format == format_filter.lower()]

        if source_filter:
            models = [m for m in models if m.source == source_filter.lower()]

        if tag_filter:
            models = [m for m in models if tag_filter.lower() in [t.lower() for t in m.tags]]

        if category_filter:
            models = [m for m in models if m.category == category_filter.lower()]

        # Sort by specified criteria
        models.sort(key=lambda m: get_sort_key(m, sort_by), reverse=reverse)
        return models

    def search_models(self, query: str) -> List[ModelEntry]:
        """Search models by name or tags."""
        query = query.lower()
        results = []

        for model in self._models.values():
            if query in model.name.lower():
                results.append(model)
            elif model.source_id and query in model.source_id.lower():
                results.append(model)
            elif any(query in tag.lower() for tag in model.tags):
                results.append(model)

        return results

    def get_gguf_models(self) -> List[ModelEntry]:
        """Get all GGUF format models."""
        return self.list_models(format_filter='gguf')

    def get_hf_models(self) -> List[ModelEntry]:
        """Get all HuggingFace source models."""
        return [m for m in self._models.values() if m.source == 'huggingface']

    def scan_directory(self, directory: str, add_found: bool = False) -> List[Dict[str, Any]]:
        """
        Scan a directory for models.

        Identifies models as:
        - Individual .gguf files (standalone GGUF models)
        - Directories containing config.json (HuggingFace models)

        Does NOT add individual .safetensors/.bin files as separate models
        if they're part of a model directory.
        """
        scan_path = Path(directory)
        if not scan_path.exists():
            return []

        found = []
        seen_paths = set(m.path for m in self._models.values())

        # First pass: Find model DIRECTORIES (HuggingFace format with config.json)
        model_dirs = set()
        for config_file in scan_path.rglob('config.json'):
            model_dir = config_file.parent
            # Skip if this is a nested config (like in a tokenizer subfolder)
            if model_dir.parent in model_dirs:
                continue
            model_dirs.add(model_dir)

            info = {
                'path': str(model_dir),
                'name': model_dir.name,
                'format': self._detect_format(model_dir),
                'size': self._calculate_size(model_dir),
                'quantization': None,
                'is_directory': True,
            }
            found.append(info)

            if add_found and str(model_dir) not in seen_paths:
                self.add_model(str(model_dir))
                seen_paths.add(str(model_dir))

        # Second pass: Find standalone GGUF files (not inside model directories)
        for gguf_file in scan_path.rglob('*.gguf'):
            # Skip if this GGUF is inside a model directory
            if any(gguf_file.is_relative_to(md) for md in model_dirs):
                continue

            info = {
                'path': str(gguf_file),
                'name': gguf_file.stem,
                'format': 'gguf',
                'size': self._calculate_size(gguf_file),
                'quantization': self._detect_quantization(gguf_file),
                'is_directory': False,
            }
            found.append(info)

            if add_found and str(gguf_file) not in seen_paths:
                self.add_model(str(gguf_file))
                seen_paths.add(str(gguf_file))

        return found

    def update_model(self, model_id: str, **kwargs) -> bool:
        """Update model metadata."""
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)

        self._save_index()
        return True

    def add_tag(self, model_id: str, tag: str) -> bool:
        """Add a tag to a model."""
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        if tag not in model.tags:
            model.tags.append(tag)
            self._save_index()
        return True

    def remove_tag(self, model_id: str, tag: str) -> bool:
        """Remove a tag from a model."""
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        if tag in model.tags:
            model.tags.remove(tag)
            self._save_index()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        models = list(self._models.values())
        total_size = sum(m.size_bytes for m in models)

        format_counts = {}
        source_counts = {}

        for model in models:
            format_counts[model.format] = format_counts.get(model.format, 0) + 1
            source_counts[model.source] = source_counts.get(model.source, 0) + 1

        return {
            'total_models': len(models),
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / (1024**3), 2),
            'format_counts': format_counts,
            'source_counts': source_counts,
        }

    def print_library(self, limit: int = 20):
        """Print the library as a formatted table."""
        models = self.list_models()[:limit]

        if not models:
            console.print("[yellow]No models in library. Use 'add' to add models.[/yellow]")
            return

        table = Table(
            title=f"Model Library ({len(self._models)} models)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Name", style="white", max_width=30)
        table.add_column("Format", style="green", width=12)
        table.add_column("Quant", style="yellow", width=8)
        table.add_column("Size", style="cyan", justify="right", width=10)
        table.add_column("Source", style="dim", width=12)

        for i, model in enumerate(models, 1):
            size_str = format_size(model.size_bytes)
            quant = model.quantization or "-"
            table.add_row(
                str(i),
                model.name[:28] + ".." if len(model.name) > 30 else model.name,
                model.format.upper(),
                quant,
                size_str,
                model.source
            )

        console.print(table)

        if len(self._models) > limit:
            console.print(f"[dim]... and {len(self._models) - limit} more models[/dim]")

    # ==========================================
    # Relationship Management Methods
    # ==========================================

    def set_parent(self, model_id: str, parent_id: str) -> bool:
        """
        Set the parent model for a given model.

        Args:
            model_id: The model to set the parent for
            parent_id: The parent model ID

        Returns:
            True if successful, False otherwise
        """
        if model_id not in self._models or parent_id not in self._models:
            return False

        model = self._models[model_id]
        parent = self._models[parent_id]

        # Remove from old parent if exists
        if model.parent_id and model.parent_id in self._models:
            old_parent = self._models[model.parent_id]
            if model_id in old_parent.children_ids:
                old_parent.children_ids.remove(model_id)

        # Set new parent
        model.parent_id = parent_id
        if model_id not in parent.children_ids:
            parent.children_ids.append(model_id)

        self._save_index()
        return True

    def remove_parent(self, model_id: str) -> bool:
        """Remove the parent relationship for a model."""
        if model_id not in self._models:
            return False

        model = self._models[model_id]
        if model.parent_id and model.parent_id in self._models:
            parent = self._models[model.parent_id]
            if model_id in parent.children_ids:
                parent.children_ids.remove(model_id)

        model.parent_id = None
        self._save_index()
        return True

    def get_parent(self, model_id: str) -> Optional[ModelEntry]:
        """Get the parent model of a given model."""
        if model_id not in self._models:
            return None

        model = self._models[model_id]
        if model.parent_id:
            return self._models.get(model.parent_id)
        return None

    def get_children(self, model_id: str) -> List[ModelEntry]:
        """Get all child models (adapters, merged, ollama) of a given model."""
        if model_id not in self._models:
            return []

        model = self._models[model_id]
        return [self._models[cid] for cid in model.children_ids if cid in self._models]

    def get_adapters(self, model_id: str) -> List[ModelEntry]:
        """Get all LoRA adapters trained on a given base model."""
        children = self.get_children(model_id)
        return [c for c in children if c.category == "adapter"]

    def get_merged_models(self, model_id: str) -> List[ModelEntry]:
        """Get all merged models derived from a given model."""
        children = self.get_children(model_id)
        return [c for c in children if c.category == "merged"]

    def get_ollama_models(self, model_id: str) -> List[ModelEntry]:
        """Get all Ollama models created from a given model."""
        children = self.get_children(model_id)
        return [c for c in children if c.category == "ollama"]

    # ==========================================
    # Tree View Methods
    # ==========================================

    def get_model_tree(self) -> List[ModelTreeNode]:
        """
        Build a tree structure of all models showing relationships.

        Returns:
            List of root ModelTreeNode objects (models without parents)
        """
        # Find root models (no parent)
        root_models = [m for m in self._models.values() if not m.parent_id]

        # Build tree recursively
        def build_node(entry: ModelEntry, depth: int = 0) -> ModelTreeNode:
            children_entries = [
                self._models[cid] for cid in entry.children_ids
                if cid in self._models
            ]
            children_nodes = [
                build_node(child, depth + 1) for child in children_entries
            ]
            return ModelTreeNode(entry=entry, children=children_nodes, depth=depth)

        return [build_node(m) for m in root_models]

    def print_tree(self, category_filter: Optional[str] = None):
        """
        Print the model library as a hierarchical tree showing relationships.

        Shows parent-child relationships between models:
        - Base models at root
        - Converted/quantized models as children
        - Ollama models as grandchildren
        """
        tree_nodes = self.get_model_tree()

        if not tree_nodes:
            console.print("[yellow]Kirjasto on tyhjä.[/yellow]")
            return

        # Calculate totals
        total = len(self._models)
        total_size = sum(m.size_bytes for m in self._models.values())

        console.print()
        console.print(Panel(
            f"[bold white]Yhteensä {total} mallia[/bold white]  •  [cyan]{format_size(total_size)}[/cyan]\n"
            f"[dim]Puu näyttää mallien väliset suhteet (parent → child)[/dim]",
            title="[bold cyan]🌳 MALLIPUU[/bold cyan]",
            border_style="cyan",
            padding=(0, 2)
        ))
        console.print()

        def get_icon(entry):
            """Get icon based on format and source."""
            if entry.category == 'ollama':
                return '🤖'
            elif entry.category == 'adapter':
                return '🔧'
            elif entry.source == 'merged':
                return '🔀'
            elif entry.format == 'gguf':
                if entry.quantization and entry.quantization not in ('F16', 'F32', 'BF16'):
                    return '⚡'  # Quantized
                else:
                    return '📦'  # F16
            else:
                return '🏠'  # SafeTensors/base

        def get_color(entry):
            """Get color based on type."""
            if entry.category == 'ollama':
                return 'blue'
            elif entry.format == 'gguf':
                if entry.quantization and entry.quantization not in ('F16', 'F32', 'BF16'):
                    return 'cyan'
                return 'yellow'
            elif entry.source == 'merged':
                return 'magenta'
            return 'green'

        def print_node(node, prefix="", is_last=True, is_root=True):
            """Print a single node with proper tree formatting."""
            entry = node.entry
            icon = get_icon(entry)
            color = get_color(entry)

            # Tree branch characters
            if is_root:
                branch = ""
                child_prefix = ""
            else:
                branch = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

            # Format model info
            size_str = format_size(entry.size_bytes)
            quant = f" [{color}][{entry.quantization}][/{color}]" if entry.quantization else ""

            # Shorten very long names
            name = entry.name if len(entry.name) <= 45 else entry.name[:42] + "..."

            # Source tag for clarity
            source_tag = ""
            if entry.source in ('converted', 'quantized', 'merged'):
                source_tag = f" [dim]({entry.source})[/dim]"

            console.print(
                f"{prefix}{branch}{icon} [{color}]{name}[/{color}]"
                f"{quant}  [dim]{size_str}[/dim]{source_tag}"
            )

            # Print children
            children = node.children
            for i, child in enumerate(children):
                print_node(child, child_prefix, i == len(children) - 1, is_root=False)

        # Print each root node
        for i, node in enumerate(tree_nodes):
            print_node(node, "", i == len(tree_nodes) - 1, is_root=True)
            if node.children:  # Add spacing after trees with children
                console.print()

        # Legend
        console.print()
        console.print(Panel(
            "[green]🏠 SafeTensors[/green]  "
            "[yellow]📦 GGUF F16[/yellow]  "
            "[cyan]⚡ Kvantisoidut[/cyan]  "
            "[magenta]🔀 Merged[/magenta]  "
            "[blue]🤖 Ollama[/blue]  "
            "[white]🔧 LoRA[/white]",
            title="[dim]Selitykset[/dim]",
            border_style="dim",
            padding=(0, 1)
        ))

    def get_base_models(self) -> List[ModelEntry]:
        """Get all base/foundation models."""
        return self.list_models(category_filter="base")

    def get_all_adapters(self) -> List[ModelEntry]:
        """Get all LoRA adapters."""
        return self.list_models(category_filter="adapter")

    def get_all_merged(self) -> List[ModelEntry]:
        """Get all merged models."""
        return self.list_models(category_filter="merged")

    def get_all_ollama(self) -> List[ModelEntry]:
        """Get all Ollama models."""
        return self.list_models(category_filter="ollama")

    # ==========================================
    # Orphan Detection and Cleanup Methods
    # ==========================================

    def find_orphaned_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find files on disk that are not in the library.

        Returns:
            Dict with category keys and lists of orphan file info
        """
        paths = get_paths()

        orphans = {
            'safetensors': [],
            'gguf': [],
            'lora': [],
            'merged': [],
            'abliterated': [],
            'ollama': []
        }

        # Get all registered paths (normalized)
        registered_paths = set()
        for m in self._models.values():
            try:
                registered_paths.add(Path(m.path).resolve())
            except Exception:
                pass

        # Scan directories
        scan_dirs = {
            'safetensors': paths.safetensors_dir,
            'gguf': paths.gguf_dir,
            'lora': paths.loras_dir,
            'merged': paths.merged_dir,
            'abliterated': paths.abliterated_dir,
            'ollama': paths.ollama_dir
        }

        for category, scan_dir in scan_dirs.items():
            if not scan_dir or not scan_dir.exists():
                continue

            for item in scan_dir.iterdir():
                try:
                    item_path = item.resolve()

                    # Check if this path or any parent/child is registered
                    is_registered = False
                    for reg_path in registered_paths:
                        if (item_path == reg_path or
                            item_path in reg_path.parents or
                            reg_path in item_path.parents):
                            is_registered = True
                            break

                    if not is_registered:
                        # Skip modelfiles in ollama dir - they're managed by OllamaManager
                        if category == 'ollama' and item.suffix == '.modelfile':
                            continue

                        size = self._calculate_size(item)
                        orphans[category].append({
                            'path': str(item),
                            'name': item.name,
                            'size_bytes': size,
                            'is_directory': item.is_dir()
                        })
                except Exception:
                    pass

        return orphans

    def get_orphan_stats(self) -> Dict[str, Any]:
        """Get statistics about orphaned files."""
        orphans = self.find_orphaned_files()

        stats = {
            'total_count': 0,
            'total_size_bytes': 0,
            'by_category': {}
        }

        for category, files in orphans.items():
            cat_size = sum(f['size_bytes'] for f in files)
            cat_count = len(files)
            stats['by_category'][category] = {
                'count': cat_count,
                'size_bytes': cat_size
            }
            stats['total_count'] += cat_count
            stats['total_size_bytes'] += cat_size

        return stats

    def cleanup_orphans(
        self,
        categories: Optional[List[str]] = None,
        dry_run: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Clean up orphaned files.

        Args:
            categories: List of categories to clean (None = all)
            dry_run: If True, only report what would be deleted

        Returns:
            List of deleted (or would-be-deleted) items
        """
        orphans = self.find_orphaned_files()
        deleted = []

        for category, files in orphans.items():
            if categories and category not in categories:
                continue

            for file_info in files:
                if not dry_run:
                    try:
                        path = Path(file_info['path'])
                        if path.exists():
                            if path.is_file():
                                path.unlink()
                            else:
                                shutil.rmtree(path)
                    except Exception as e:
                        console.print(f"[red]Error deleting {file_info['path']}: {e}[/red]")
                        continue

                deleted.append(file_info)

        return deleted

    # ==========================================
    # Relationship Validation Methods
    # ==========================================

    def validate_relationships(self) -> Dict[str, List[str]]:
        """
        Validate all parent-child relationships in the library.

        Returns:
            Dict with 'broken_parents', 'broken_children', 'orphaned_children' lists
        """
        issues = {
            'broken_parents': [],      # Children pointing to non-existent parents
            'broken_children': [],     # Parents listing non-existent children
            'orphaned_children': [],   # Children whose parent doesn't list them
        }

        for model_id, model in self._models.items():
            # Check parent reference
            if model.parent_id:
                if model.parent_id not in self._models:
                    issues['broken_parents'].append(
                        f"{model.name} -> missing parent {model.parent_id}"
                    )
                else:
                    parent = self._models[model.parent_id]
                    if model_id not in parent.children_ids:
                        issues['orphaned_children'].append(
                            f"{model.name} has parent {parent.name} but not listed as child"
                        )

            # Check children references
            for child_id in model.children_ids:
                if child_id not in self._models:
                    issues['broken_children'].append(
                        f"{model.name} -> missing child {child_id}"
                    )

        return issues

    def repair_relationships(self) -> int:
        """Repair broken relationships. Returns count of repairs made."""
        repairs = 0

        for model_id, model in self._models.items():
            # Fix broken parent references
            if model.parent_id and model.parent_id not in self._models:
                model.parent_id = None
                repairs += 1

            # Fix broken children references
            valid_children = [cid for cid in model.children_ids if cid in self._models]
            if len(valid_children) != len(model.children_ids):
                repairs += len(model.children_ids) - len(valid_children)
                model.children_ids = valid_children

            # Ensure bidirectional consistency
            if model.parent_id and model.parent_id in self._models:
                parent = self._models[model.parent_id]
                if model_id not in parent.children_ids:
                    parent.children_ids.append(model_id)
                    repairs += 1

        if repairs > 0:
            self._save_index()

        return repairs

    def check_deletion_impact(self, model_id: str) -> Dict[str, Any]:
        """
        Analyze the impact of deleting a model.

        Returns:
            Dict with 'children', 'ollama_name', 'size_freed', 'warnings'
        """
        if model_id not in self._models:
            return {'error': 'Model not found'}

        model = self._models[model_id]
        children = self.get_children(model_id)

        # Calculate cascade size
        cascade_size = model.size_bytes
        cascade_models = [model]

        def add_descendants(mid):
            nonlocal cascade_size
            for child in self.get_children(mid):
                cascade_size += child.size_bytes
                cascade_models.append(child)
                add_descendants(child.id)

        add_descendants(model_id)

        warnings = []
        if children:
            ollama_children = [c for c in children if c.category == 'ollama']
            if ollama_children:
                warnings.append(
                    f"Talla mallilla on {len(ollama_children)} Ollama-lapsimallia"
                )

        return {
            'model': model,
            'children': children,
            'all_descendants': cascade_models,
            'ollama_name': model.ollama_info.get('ollama_name') if model.ollama_info else None,
            'size_freed': cascade_size,
            'warnings': warnings
        }

    def find_missing_files(self) -> List[ModelEntry]:
        """Find models whose files no longer exist on disk."""
        missing = []
        for model in self._models.values():
            if not Path(model.path).exists():
                missing.append(model)
        return missing

    def cleanup_duplicates(self) -> int:
        """
        Remove duplicate entries (same file path).

        Returns:
            Number of duplicates removed
        """
        seen_paths = {}
        duplicates = []

        for model_id, model in self._models.items():
            try:
                abs_path = str(Path(model.path).absolute())
                if abs_path in seen_paths:
                    # This is a duplicate - mark for removal
                    duplicates.append(model_id)
                else:
                    seen_paths[abs_path] = model_id
            except Exception:
                pass

        # Remove duplicates
        for dup_id in duplicates:
            del self._models[dup_id]

        if duplicates:
            self._save_index()
            console.print(f"[green]Poistettu {len(duplicates)} duplikaattia[/green]")

        return len(duplicates)

    def cleanup_library(self) -> Dict[str, int]:
        """
        Full library cleanup: remove missing files and duplicates.

        Returns:
            Dict with 'missing' and 'duplicates' counts
        """
        results = {'missing': 0, 'duplicates': 0}

        # Remove missing files
        missing = self.find_missing_files()
        for m in missing:
            del self._models[m.id]
        results['missing'] = len(missing)

        # Remove duplicates
        results['duplicates'] = self.cleanup_duplicates()

        if results['missing'] > 0 or results['duplicates'] > 0:
            self._save_index()

        return results

    def get_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics broken down by category."""
        category_stats = {}

        for model in self._models.values():
            if model.category not in category_stats:
                category_stats[model.category] = {
                    'count': 0,
                    'size_bytes': 0
                }
            category_stats[model.category]['count'] += 1
            category_stats[model.category]['size_bytes'] += model.size_bytes

        return category_stats

    # ==========================================
    # Categorized Model Query Methods
    # ==========================================

    def get_convertible_models(self) -> List[ModelEntry]:
        """
        Get models that can be converted to GGUF.

        Returns SafeTensors and merged models (not GGUF).
        """
        return [
            m for m in self._models.values()
            if m.format in ('safetensors', 'pytorch') and m.format != 'gguf'
        ]

    def get_quantizable_models(self) -> List[ModelEntry]:
        """
        Get GGUF models that can be quantized.

        Returns F16/F32 GGUF models (not already heavily quantized).
        """
        return [
            m for m in self._models.values()
            if m.format == 'gguf' and m.quantization in (None, 'F16', 'F32', 'BF16')
        ]

    def get_already_quantized_models(self) -> List[ModelEntry]:
        """
        Get GGUF models that are already quantized.

        Returns GGUF models with quantization other than F16/F32.
        """
        return [
            m for m in self._models.values()
            if m.format == 'gguf' and m.quantization not in (None, 'F16', 'F32', 'BF16')
        ]

    def get_models_grouped_by_format(self) -> Dict[str, List[ModelEntry]]:
        """
        Group all models by their format/type for categorized display.

        Returns:
            Dict with keys: 'safetensors', 'gguf_f16', 'gguf_quantized',
                           'merged', 'ollama', 'adapter', 'other'
        """
        groups: Dict[str, List[ModelEntry]] = {
            'safetensors': [],
            'gguf_f16': [],
            'gguf_quantized': [],
            'merged': [],
            'ollama': [],
            'adapter': [],
            'other': []
        }

        for model in self._models.values():
            # Check category first
            if model.category == 'ollama':
                groups['ollama'].append(model)
            elif model.category == 'adapter':
                groups['adapter'].append(model)
            elif model.category == 'merged':
                groups['merged'].append(model)
            # Then check format
            elif model.format == 'gguf':
                if model.quantization in (None, 'F16', 'F32', 'BF16'):
                    groups['gguf_f16'].append(model)
                else:
                    groups['gguf_quantized'].append(model)
            elif model.format in ('safetensors', 'pytorch'):
                groups['safetensors'].append(model)
            else:
                groups['other'].append(model)

        # Sort each group by name
        for key in groups:
            groups[key].sort(key=lambda m: m.name.lower())

        return groups

    def get_all_gguf_models(self) -> List[ModelEntry]:
        """Get all GGUF format models (both F16 and quantized)."""
        return [m for m in self._models.values() if m.format == 'gguf']

    def refresh(self):
        """Reload the library index from disk."""
        self._load_index()
