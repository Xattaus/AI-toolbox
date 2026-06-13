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


@dataclass
class ExtendedModelInfo:
    """Extended model information with full metadata from HuggingFace."""

    # Basic info
    model_id: str
    author: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    last_modified: str = ""

    # Extended fields
    gated: Optional[str] = None  # "auto", "manual", or None
    license: Optional[str] = None  # "mit", "apache-2.0", "llama3", etc.
    library_name: Optional[str] = None  # "transformers", "gguf", etc.
    card_data: Optional[Dict[str, Any]] = None  # Full YAML frontmatter

    # Computed/analyzed fields
    model_size_estimate: Optional[str] = None  # "7B", "13B", "70B"
    parameter_count: Optional[int] = None  # Actual parameter count
    compatible_apps: List[str] = field(default_factory=list)  # ["ollama", "llama.cpp"]
    inference_providers: List[str] = field(default_factory=list)  # ["hf-inference", "together"]

    # File information
    has_safetensors: bool = False
    has_gguf: bool = False
    total_size_bytes: int = 0
    gguf_variants: List[Dict[str, Any]] = field(default_factory=list)

    # Technical details
    architecture: Optional[str] = None  # "LlamaForCausalLM"
    context_length: Optional[int] = None  # Max context window
    base_model: Optional[str] = None  # For fine-tuned models
    languages: List[str] = field(default_factory=list)  # ["en", "fi"]


@dataclass
class HFSearchResult:
    """Simplified search result from HuggingFace with extended metadata."""

    model_id: str
    author: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    last_modified: str

    # Extended fields for rich display
    gated: Optional[str] = None
    license: Optional[str] = None
    library_name: Optional[str] = None
    model_size: Optional[str] = None
    compatible_apps: List[str] = field(default_factory=list)
    has_gguf: bool = False
    has_safetensors: bool = False


@dataclass
class GGUFFileInfo:
    """Information about a specific GGUF file variant."""

    filename: str
    size_bytes: int
    quantization: Optional[str] = None  # "Q4_K_M", "Q5_K_S", etc.
    quality_score: float = 3.0  # 1-5 scale based on perplexity
    estimated_vram_gb: float = 0.0  # Estimated VRAM needed
