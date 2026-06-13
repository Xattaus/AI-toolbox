"""
AI TOOLBOX - HuggingFace Search Engine
======================================

Professional-grade HuggingFace model search with advanced filtering,
model card analysis, and metadata extraction.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from huggingface_hub import HfApi, ModelInfo
from rich.console import Console

# Import errors - try new location first, fall back to old
try:
    from huggingface_hub.errors import RepositoryNotFoundError, GatedRepoError
except ImportError:
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from .hf_filters import (
    APPS,
    QUANTIZATION_QUALITY,
    LICENSES,
    parse_model_size_from_name,
    detect_quantization_from_filename,
    get_app_compatibility,
)

console = Console()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SearchFilters:
    """All search filters for HuggingFace model search."""

    query: Optional[str] = None  # Text search query
    tasks: List[str] = field(default_factory=list)  # pipeline_tag filter
    libraries: List[str] = field(default_factory=list)  # transformers, gguf, etc.
    apps: List[str] = field(default_factory=list)  # ollama, llama.cpp, vllm
    author: Optional[str] = None  # meta-llama, mistralai, etc.
    gated: Optional[bool] = None  # True/False/None for any
    inference: Optional[str] = None  # "warm" for active inference
    min_downloads: Optional[int] = None  # Minimum download count
    min_likes: Optional[int] = None  # Minimum like count
    language: Optional[str] = None  # Language filter (en, fi, etc.)
    license_filter: Optional[str] = None  # License filter

    def has_post_filters(self) -> bool:
        """True if any filter must be applied client-side after the API call."""
        return bool(self.apps or self.min_downloads or self.min_likes)


@dataclass
class SearchResult:
    """Extended search result with full metadata."""

    model_id: str
    author: str
    downloads: int
    likes: int
    pipeline_tag: Optional[str]
    tags: List[str]
    last_modified: str

    # Extended fields
    gated: Optional[str] = None  # "auto", "manual", False
    license: Optional[str] = None  # "mit", "apache-2.0", etc.
    library_name: Optional[str] = None  # "transformers", "gguf"
    inference_status: Optional[str] = None  # "warm", "cold"
    compatible_apps: List[str] = field(default_factory=list)  # ["ollama", "llama.cpp"]
    model_size: Optional[str] = None  # "7B", "13B", "70B"
    has_gguf: bool = False
    has_safetensors: bool = False


@dataclass
class ModelCardInfo:
    """Parsed model card information."""

    # Basic info
    model_id: str
    description: Optional[str] = None  # README excerpt

    # Card data (YAML frontmatter)
    license: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    base_model: Optional[str] = None
    datasets: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Technical info
    model_type: Optional[str] = None  # "causal-lm", "seq2seq", etc.
    architecture: Optional[str] = None  # "LlamaForCausalLM"
    parameter_count: Optional[int] = None
    quantization: Optional[str] = None  # "Q4_K_M", "fp16", etc.
    context_length: Optional[int] = None
    model_size: Optional[str] = None  # "7B", "13B", etc.
    library_name: Optional[str] = None  # "transformers", "gguf"

    # Files
    files: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    has_safetensors: bool = False
    has_gguf: bool = False
    gguf_variants: List[Dict[str, Any]] = field(default_factory=list)

    # Compatibility
    compatible_apps: List[str] = field(default_factory=list)
    inference_providers: List[str] = field(default_factory=list)

    # Metadata
    downloads: int = 0
    likes: int = 0
    last_modified: Optional[str] = None
    gated: Optional[str] = None

    # Benchmark results (if available)
    benchmark_results: Optional[Dict[str, Any]] = None


@dataclass
class GGUFVariant:
    """Information about a GGUF file variant."""

    filename: str
    size_bytes: int
    quantization: Optional[str]
    quality_score: float
    estimated_vram_gb: float


# =============================================================================
# HF SEARCH ENGINE
# =============================================================================


class HFSearchEngine:
    """Professional HuggingFace search engine with advanced filtering."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the search engine.

        Args:
            token: Optional HuggingFace API token for authenticated requests
        """
        self.api = HfApi(token=token)
        self.token = token

    def search(
        self,
        filters: SearchFilters,
        sort: str = "downloads",
        direction: int = -1,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[SearchResult], int]:
        """
        Search for models with advanced filtering.

        Args:
            filters: SearchFilters object with all filter criteria
            sort: Sort field (downloads, likes, lastModified, trending)
            direction: Sort direction (-1 descending, 1 ascending)
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Tuple of (results list, total count estimate)
        """
        try:
            # Client-side filters (apps, min_downloads, min_likes) drop results
            # after the API call, so over-fetch to still fill the requested page.
            fetch_limit = limit + offset
            if filters.has_post_filters():
                fetch_limit = min(fetch_limit * 3, 1000)

            # Build API parameters
            api_kwargs = {
                "search": filters.query,
                "sort": sort,
                "direction": -1 if direction < 0 else None,
                "limit": fetch_limit,
            }

            # Library/format filter (native API param, accepts str or list)
            if filters.libraries:
                api_kwargs["library"] = (
                    filters.libraries[0] if len(filters.libraries) == 1 else filters.libraries
                )

            # Add task filter (pipeline_tag) - still works as direct param
            if filters.tasks:
                api_kwargs["pipeline_tag"] = filters.tasks[0]

            # Add author filter (direct param)
            if filters.author:
                api_kwargs["author"] = filters.author

            # Add gated filter (direct param)
            if filters.gated is not None:
                api_kwargs["gated"] = filters.gated

            # Add language filter (direct param)
            if filters.language:
                api_kwargs["language"] = filters.language

            # Inference availability filter ("warm" = served by a provider)
            if filters.inference:
                api_kwargs["inference"] = filters.inference

            filter_tags = []

            # License filter (models carry a license:<id> tag on the Hub)
            if filters.license_filter:
                filter_tags.append(f"license:{filters.license_filter.lower()}")

            # If every selected app requires GGUF, narrow the API query to
            # gguf-tagged models so the page isn't dominated by incompatible hits
            if filters.apps and not filters.libraries:
                if all(
                    APPS.get(app, {}).get("requires") == ["gguf"]
                    for app in filters.apps
                ):
                    filter_tags.append("gguf")

            if filter_tags:
                api_kwargs["filter"] = (
                    filter_tags[0] if len(filter_tags) == 1 else filter_tags
                )

            # Use the list_models API
            models = self.api.list_models(**api_kwargs)

            results = []
            count = 0

            for model in models:
                count += 1

                # Skip until we reach offset
                if count <= offset:
                    continue

                # Stop at limit
                if len(results) >= limit:
                    break

                # Apply post-filters (min_downloads, min_likes)
                if filters.min_downloads and (model.downloads or 0) < filters.min_downloads:
                    continue
                if filters.min_likes and (model.likes or 0) < filters.min_likes:
                    continue

                result = self._model_to_search_result(model)

                # App compatibility filter (client-side: requires format info
                # that is only available after result conversion)
                if filters.apps:
                    if not any(app in result.compatible_apps for app in filters.apps):
                        continue

                results.append(result)

            # Estimate total (may be higher)
            total_estimate = max(count, len(results))

            return results, total_estimate

        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return [], 0

    def search_quick(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
    ) -> List[SearchResult]:
        """
        Quick search with just a text query.

        Args:
            query: Search query
            limit: Maximum results
            sort: Sort field

        Returns:
            List of search results
        """
        filters = SearchFilters(query=query)
        results, _ = self.search(filters, sort=sort, limit=limit)
        return results

    def search_by_preset(
        self,
        preset_filters: Dict[str, Any],
        limit: int = 20,
    ) -> List[SearchResult]:
        """
        Search using a preset filter configuration.

        Args:
            preset_filters: Dict with filter values
            limit: Maximum results

        Returns:
            List of search results
        """
        filters = SearchFilters(
            query=preset_filters.get("query"),
            tasks=preset_filters.get("tasks", []),
            libraries=preset_filters.get("libraries", []),
            apps=preset_filters.get("apps", []),
            author=preset_filters.get("author"),
        )

        results, _ = self.search(filters, limit=limit)
        return results

    def get_model_card(self, model_id: str) -> Optional[ModelCardInfo]:
        """
        Get detailed model card information.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")

        Returns:
            ModelCardInfo with full metadata, or None if not found
        """
        try:
            info = self.api.model_info(model_id, files_metadata=True)
            return self._parse_model_card(info)

        except GatedRepoError:
            console.print(f"[yellow]Gated model - limited info available: {model_id}[/yellow]")
            # Try to get basic info without files
            try:
                info = self.api.model_info(model_id, files_metadata=False)
                return self._parse_model_card(info, limited=True)
            except Exception:
                return None
        except RepositoryNotFoundError:
            console.print(f"[red]Model not found: {model_id}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Error fetching model card: {e}[/red]")
            return None

    def get_gguf_variants(self, model_id: str) -> List[GGUFVariant]:
        """
        Get all GGUF variants available for a model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            List of GGUFVariant objects sorted by quality
        """
        card = self.get_model_card(model_id)
        if not card:
            return []

        variants = []
        for gguf_info in card.gguf_variants:
            variant = GGUFVariant(
                filename=gguf_info["filename"],
                size_bytes=gguf_info["size"],
                quantization=gguf_info.get("quantization"),
                quality_score=gguf_info.get("quality", 3.0),
                estimated_vram_gb=gguf_info.get("vram_estimate", 0),
            )
            variants.append(variant)

        # Sort by quality (descending)
        variants.sort(key=lambda x: x.quality_score, reverse=True)
        return variants

    def get_popular_by_task(
        self,
        task: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Get popular models for a specific task.

        Args:
            task: Pipeline tag (e.g., "text-generation")
            limit: Maximum results

        Returns:
            List of popular models
        """
        filters = SearchFilters(tasks=[task])
        results, _ = self.search(filters, sort="downloads", limit=limit)
        return results

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _model_to_search_result(self, model: ModelInfo) -> SearchResult:
        """Convert HfApi ModelInfo to SearchResult."""
        tags = model.tags or []

        # Detect library and format
        library_name = getattr(model, "library_name", None)
        has_gguf = "gguf" in tags or library_name == "gguf"
        has_safetensors = "safetensors" in tags

        # Determine compatible apps
        libraries = [library_name] if library_name else []
        if has_gguf:
            libraries.append("gguf")
        if has_safetensors:
            libraries.append("safetensors")

        compatible_apps = get_app_compatibility(libraries, has_gguf)

        # Parse model size from name
        model_size = parse_model_size_from_name(model.id)

        # Get license from card_data if available
        license_info = None
        if hasattr(model, "card_data") and model.card_data:
            license_info = getattr(model.card_data, "license", None)

        # Get gated status
        gated = getattr(model, "gated", None)
        if gated is True:
            gated = "manual"
        elif gated == "auto":
            gated = "auto"
        elif gated is False or gated is None:
            gated = None

        # Format last modified
        last_mod = model.lastModified
        if isinstance(last_mod, datetime):
            last_modified = last_mod.strftime("%Y-%m-%d")
        else:
            last_modified = str(last_mod) if last_mod else ""

        return SearchResult(
            model_id=model.id,
            author=model.author or "Unknown",
            downloads=model.downloads or 0,
            likes=model.likes or 0,
            pipeline_tag=model.pipeline_tag,
            tags=tags,
            last_modified=last_modified,
            gated=gated,
            license=license_info,
            library_name=library_name,
            inference_status=None,  # Would need additional API call
            compatible_apps=compatible_apps,
            model_size=model_size,
            has_gguf=has_gguf,
            has_safetensors=has_safetensors,
        )

    def _parse_model_card(
        self, info: ModelInfo, limited: bool = False
    ) -> ModelCardInfo:
        """Parse ModelInfo into ModelCardInfo."""
        card = ModelCardInfo(model_id=info.id)

        # Basic metadata
        card.downloads = info.downloads or 0
        card.likes = info.likes or 0
        card.tags = info.tags or []

        # Format last modified
        last_mod = info.lastModified
        if isinstance(last_mod, datetime):
            card.last_modified = last_mod.strftime("%Y-%m-%d %H:%M")
        else:
            card.last_modified = str(last_mod) if last_mod else None

        # Gated status
        gated = getattr(info, "gated", None)
        if gated is True:
            card.gated = "manual"
        elif gated == "auto":
            card.gated = "auto"

        # Parse card_data (YAML frontmatter)
        if info.card_data:
            card.license = getattr(info.card_data, "license", None)
            card.languages = getattr(info.card_data, "language", []) or []
            if isinstance(card.languages, str):
                card.languages = [card.languages]

            card.base_model = getattr(info.card_data, "base_model", None)
            card.datasets = getattr(info.card_data, "datasets", []) or []

            # Model index (benchmark results)
            model_index = getattr(info.card_data, "model_index", None)
            if model_index:
                card.benchmark_results = model_index

        # Parse files if available
        if not limited and info.siblings:
            card.files = []
            card.total_size_bytes = 0

            for sibling in info.siblings:
                file_info = {
                    "filename": sibling.rfilename,
                    "size": sibling.size or 0,
                }
                card.files.append(file_info)
                card.total_size_bytes += file_info["size"]

                # Check file types
                fname = sibling.rfilename.lower()
                if fname.endswith(".safetensors"):
                    card.has_safetensors = True
                elif fname.endswith(".gguf"):
                    card.has_gguf = True
                    # Extract GGUF variant info
                    quant = detect_quantization_from_filename(fname)
                    quality = QUANTIZATION_QUALITY.get(quant, 3.0) if quant else 3.0
                    vram_est = self._estimate_vram(sibling.size or 0)

                    card.gguf_variants.append({
                        "filename": sibling.rfilename,
                        "size": sibling.size or 0,
                        "quantization": quant,
                        "quality": quality,
                        "vram_estimate": vram_est,
                    })

        # Extract model size from safetensors metadata or name
        card.model_size = parse_model_size_from_name(info.id)

        # Try to get parameter count from safetensors metadata
        if hasattr(info, "safetensors") and info.safetensors:
            param_count = info.safetensors.get("total", 0)
            if param_count:
                card.parameter_count = param_count
                card.model_size = self._format_param_count(param_count)

        # Determine compatible apps
        libraries = []
        if info.library_name:
            libraries.append(info.library_name)
            card.library_name = info.library_name
        if card.has_gguf:
            libraries.append("gguf")
        if card.has_safetensors:
            libraries.append("safetensors")

        card.compatible_apps = get_app_compatibility(libraries, card.has_gguf)

        # Extract architecture from config if available
        card.architecture = self._detect_architecture(card.tags)

        return card

    def _estimate_vram(self, file_size_bytes: int) -> float:
        """Estimate VRAM needed in GB."""
        # File size + ~30% overhead for KV cache
        return (file_size_bytes * 1.3) / (1024 ** 3)

    def _format_param_count(self, count: int) -> str:
        """Format parameter count as human-readable string."""
        if count >= 1_000_000_000_000:
            return f"{count / 1_000_000_000_000:.1f}T"
        elif count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.0f}M"
        else:
            return f"{count / 1_000:.0f}K"

    def _detect_architecture(self, tags: List[str]) -> Optional[str]:
        """Detect model architecture from tags."""
        arch_patterns = {
            "llama": "LlamaForCausalLM",
            "mistral": "MistralForCausalLM",
            "qwen": "Qwen2ForCausalLM",
            "phi": "PhiForCausalLM",
            "gemma": "GemmaForCausalLM",
            "falcon": "FalconForCausalLM",
            "mpt": "MptForCausalLM",
            "gpt-neox": "GPTNeoXForCausalLM",
            "gpt2": "GPT2LMHeadModel",
            "bloom": "BloomForCausalLM",
            "opt": "OPTForCausalLM",
            "stablelm": "StableLmForCausalLM",
        }

        tags_lower = [t.lower() for t in tags]

        for pattern, arch in arch_patterns.items():
            if any(pattern in tag for tag in tags_lower):
                return arch

        return None


# =============================================================================
# MODEL CARD ANALYZER
# =============================================================================


class ModelCardAnalyzer:
    """Analyzes and interprets HuggingFace model cards."""

    def __init__(self, api: Optional[HfApi] = None, token: Optional[str] = None):
        """Initialize the analyzer."""
        self.api = api or HfApi(token=token)

    def analyze(self, model_id: str) -> Optional[ModelCardInfo]:
        """
        Analyze a model card in depth.

        Args:
            model_id: HuggingFace model ID

        Returns:
            ModelCardInfo with comprehensive analysis
        """
        engine = HFSearchEngine(token=getattr(self.api, "token", None))
        return engine.get_model_card(model_id)

    def get_recommended_gguf(
        self,
        model_id: str,
        max_vram_gb: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get recommended GGUF variant based on available VRAM.

        Args:
            model_id: HuggingFace model ID
            max_vram_gb: Maximum available VRAM in GB (None for best quality)

        Returns:
            Dict with recommended variant info, or None
        """
        engine = HFSearchEngine(token=getattr(self.api, "token", None))
        variants = engine.get_gguf_variants(model_id)

        if not variants:
            return None

        # If no VRAM constraint, return highest quality
        if max_vram_gb is None:
            best = variants[0]
            return {
                "filename": best.filename,
                "size_bytes": best.size_bytes,
                "quantization": best.quantization,
                "quality": best.quality_score,
                "vram_gb": best.estimated_vram_gb,
                "reason": "Highest quality available",
            }

        # Find best quality that fits in VRAM
        for variant in variants:
            if variant.estimated_vram_gb <= max_vram_gb:
                return {
                    "filename": variant.filename,
                    "size_bytes": variant.size_bytes,
                    "quantization": variant.quantization,
                    "quality": variant.quality_score,
                    "vram_gb": variant.estimated_vram_gb,
                    "reason": f"Best quality within {max_vram_gb}GB VRAM",
                }

        # Nothing fits, return smallest
        smallest = min(variants, key=lambda x: x.estimated_vram_gb)
        return {
            "filename": smallest.filename,
            "size_bytes": smallest.size_bytes,
            "quantization": smallest.quantization,
            "quality": smallest.quality_score,
            "vram_gb": smallest.estimated_vram_gb,
            "reason": f"Smallest available (needs {smallest.estimated_vram_gb:.1f}GB VRAM)",
            "warning": f"May exceed your {max_vram_gb}GB VRAM limit",
        }

    def compare_quantizations(
        self,
        model_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Compare all quantization variants for a model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            List of dicts with comparison data
        """
        engine = HFSearchEngine(token=getattr(self.api, "token", None))
        variants = engine.get_gguf_variants(model_id)

        comparisons = []
        for variant in variants:
            quality_stars = int(variant.quality_score)
            comparisons.append({
                "filename": variant.filename,
                "quantization": variant.quantization or "Unknown",
                "size_gb": variant.size_bytes / (1024 ** 3),
                "vram_gb": variant.estimated_vram_gb,
                "quality": variant.quality_score,
                "quality_stars": "*" * quality_stars + "." * (5 - quality_stars),
                "recommended": variant.quantization in ["Q4_K_M", "Q5_K_M"],
            })

        return comparisons

    def get_license_info(self, license_id: Optional[str]) -> Dict[str, Any]:
        """
        Get detailed license information.

        Args:
            license_id: License identifier

        Returns:
            Dict with license details
        """
        if not license_id:
            return {
                "name": "Unknown",
                "open": None,
                "commercial": None,
                "requires_attribution": None,
            }

        license_data = LICENSES.get(license_id.lower(), {})

        return {
            "id": license_id,
            "name": license_data.get("name", license_id),
            "open": license_data.get("open"),
            "commercial": license_data.get("commercial"),
            "gated": license_data.get("gated", False),
        }
