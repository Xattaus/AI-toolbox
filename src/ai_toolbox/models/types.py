"""
AI TOOLBOX - Model Types
========================

Common types and dataclasses for model handling.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelFormat(Enum):
    """Supported model formats."""
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    GGML = "ggml"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, ext: str) -> 'ModelFormat':
        """Get format from file extension."""
        ext = ext.lower().lstrip('.')
        mapping = {
            'safetensors': cls.SAFETENSORS,
            'bin': cls.PYTORCH,
            'pt': cls.PYTORCH,
            'pth': cls.PYTORCH,
            'gguf': cls.GGUF,
            'ggml': cls.GGML,
        }
        return mapping.get(ext, cls.UNKNOWN)


class ModelSource(Enum):
    """Model source types."""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    CONVERTED = "converted"
    MERGED = "merged"
    TRAINED = "trained"
    ABLITERATED = "abliterated"
    OLLAMA = "ollama"


class ModelCategory(Enum):
    """Model category for organization."""
    BASE = "base"           # Base/foundation models (SafeTensors, GGUF)
    ADAPTER = "adapter"     # LoRA adapters
    MERGED = "merged"       # Merged models
    OLLAMA = "ollama"       # Ollama models

    @classmethod
    def from_source(cls, source: str) -> 'ModelCategory':
        """Infer category from source type."""
        mapping = {
            "huggingface": cls.BASE,
            "local": cls.BASE,
            "converted": cls.BASE,
            "merged": cls.MERGED,
            "trained": cls.ADAPTER,
            "abliterated": cls.BASE,  # Abliterated models are base models
            "ollama": cls.OLLAMA,
        }
        return mapping.get(source.lower(), cls.BASE)


@dataclass
class ModelEntry:
    """Represents a model in the library."""
    # Basic info
    id: str
    name: str
    source: str
    source_id: Optional[str]
    path: str
    format: str
    size_bytes: int
    quantization: Optional[str]
    added_date: str
    tags: List[str]
    metadata: Dict[str, Any]

    # Organization (NEW)
    category: str = "base"  # "base", "adapter", "merged", "ollama"
    parent_id: Optional[str] = None  # ID of parent model (for adapters/merged)
    children_ids: List[str] = field(default_factory=list)  # IDs of derived models

    # Training info (for adapters)
    training_info: Optional[Dict[str, Any]] = None  # epochs, loss, backend, base_model

    # Merge info (for merged models)
    merge_info: Optional[Dict[str, Any]] = None  # method, ratio, source_models

    # Abliteration info (for abliterated models)
    abliteration_info: Optional[Dict[str, Any]] = None  # method, strength, source_model

    # Ollama info
    ollama_info: Optional[Dict[str, Any]] = None  # ollama_name, system_prompt, params


@dataclass
class ModelSearchResult:
    """Represents a model search result from HuggingFace."""
    model_id: str
    author: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    last_modified: str


@dataclass
class ModelDetails:
    """Detailed information about a model."""
    model_id: str
    author: str
    sha: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    files: List[Dict[str, Any]]
    total_size: int
    siblings: List[Any]


@dataclass
class LoRAInfo:
    """Information about a LoRA adapter."""
    name: str
    path: str
    base_model: Optional[str]
    size_bytes: int
    rank: Optional[int] = None
    alpha: Optional[float] = None
    target_modules: List[str] = field(default_factory=list)


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str                           # Ollama model name
    gguf_path: Optional[str] = None     # Source GGUF file path
    modelfile_path: Optional[str] = None  # Stored Modelfile path
    system_prompt: Optional[str] = None # System prompt used
    template_name: Optional[str] = None # Template name (assistant, coder, etc.)
    parameters: Dict[str, Any] = field(default_factory=dict)  # temperature, context, etc.
    created_date: Optional[str] = None


@dataclass
class ModelTreeNode:
    """Node in the model relationship tree."""
    entry: ModelEntry
    children: List['ModelTreeNode'] = field(default_factory=list)
    depth: int = 0

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0
