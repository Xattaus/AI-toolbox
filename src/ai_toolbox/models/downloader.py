"""
AI TOOLBOX - Model Downloader
=============================

Download models from HuggingFace Hub with duplicate detection.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    snapshot_download,
    model_info as get_model_info,
)
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TaskProgressColumn,
)
from rich import box

from ..core.paths import get_downloads_dir, get_loras_dir
from ..core.ui import format_size
from .types import ModelSearchResult, ModelDetails, ExtendedModelInfo, HFSearchResult
from .hf_search import HFSearchEngine, SearchFilters, SearchResult, ModelCardInfo

console = Console()


class ModelDownloader:
    """Downloads models from HuggingFace Hub."""

    MODEL_EXTENSIONS = ['.safetensors', '.bin', '.pt', '.pth', '.gguf', '.ggml']
    CONFIG_FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'vocab.json', 'merges.txt', 'special_tokens_map.json',
                    'generation_config.json']

    def __init__(
        self,
        download_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """Initialize the downloader."""
        if download_dir:
            self.download_dir = Path(download_dir)
        else:
            self.download_dir = get_downloads_dir()

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.token = (
            token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or self._token_from_config()
        )
        self.api = HfApi(token=self.token)
        self._search_engine = None  # Lazy-loaded HFSearchEngine

    @staticmethod
    def _token_from_config() -> Optional[str]:
        """Read HF token from the toolbox config file (Settings menu)."""
        try:
            from ..core.config import get_config
            return get_config().hf_token
        except Exception:
            return None

    def set_token(self, token: Optional[str]):
        """Set or clear the HF token at runtime (recreates API clients)."""
        self.token = token
        self.api = HfApi(token=self.token)
        self._search_engine = None  # Recreate lazily with new token

    @property
    def search_engine(self) -> HFSearchEngine:
        """Get or create the HFSearchEngine instance."""
        if self._search_engine is None:
            self._search_engine = HFSearchEngine(token=self.token)
        return self._search_engine

    def search_models(
        self,
        query: str,
        limit: int = 20,
        filter_task: Optional[str] = None,
        sort: str = "downloads",
    ) -> List[ModelSearchResult]:
        """
        Search for models on HuggingFace Hub.

        This is a backward-compatible wrapper around the new HFSearchEngine.
        For advanced filtering, use search_models_advanced() instead.
        """
        # Build filters for the new engine
        filters = SearchFilters(
            query=query,
            tasks=[filter_task] if filter_task else [],
        )

        # Use new search engine
        results, _ = self.search_engine.search(filters, sort=sort, limit=limit)

        # Convert to legacy format for backward compatibility
        legacy_results = []
        for result in results:
            legacy_results.append(ModelSearchResult(
                model_id=result.model_id,
                author=result.author,
                downloads=result.downloads,
                likes=result.likes,
                pipeline_tag=result.pipeline_tag,
                tags=result.tags,
                last_modified=result.last_modified,
            ))

        return legacy_results

    def search_models_advanced(
        self,
        filters: SearchFilters,
        sort: str = "downloads",
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[SearchResult], int]:
        """
        Advanced model search with full filtering capabilities.

        Args:
            filters: SearchFilters object with all filter criteria
            sort: Sort field (downloads, likes, lastModified)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (results list, total count estimate)
        """
        return self.search_engine.search(filters, sort=sort, limit=limit, offset=offset)

    def get_model_card(self, model_id: str) -> Optional[ModelCardInfo]:
        """
        Get detailed model card information with full metadata.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")

        Returns:
            ModelCardInfo with comprehensive model information
        """
        return self.search_engine.get_model_card(model_id)

    def get_gguf_variants(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get all GGUF quantization variants for a model.

        Args:
            model_id: HuggingFace model ID

        Returns:
            List of dicts with variant info (filename, size, quantization, quality)
        """
        variants = self.search_engine.get_gguf_variants(model_id)
        return [
            {
                "filename": v.filename,
                "size_bytes": v.size_bytes,
                "quantization": v.quantization,
                "quality_score": v.quality_score,
                "estimated_vram_gb": v.estimated_vram_gb,
            }
            for v in variants
        ]

    def get_model_details(self, model_id: str) -> Optional[ModelDetails]:
        """Get detailed information about a model."""
        try:
            info = get_model_info(model_id, token=self.token)

            files = []
            total_size = 0

            for sibling in info.siblings or []:
                file_size = sibling.size or 0
                files.append({
                    'name': sibling.rfilename,
                    'size': file_size,
                })
                total_size += file_size

            return ModelDetails(
                model_id=info.id,
                author=info.author or "Unknown",
                sha=info.sha or "",
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                pipeline_tag=info.pipeline_tag,
                tags=info.tags or [],
                files=files,
                total_size=total_size,
                siblings=info.siblings or [],
            )

        except RepositoryNotFoundError:
            console.print(f"[red]Model not found: {model_id}[/red]")
            return None
        except GatedRepoError:
            console.print(f"[yellow]This model requires authentication.[/yellow]")
            console.print("[dim]Set HF_TOKEN environment variable or login with 'huggingface-cli login'[/dim]")
            return None
        except Exception as e:
            console.print(f"[red]Error getting model info: {e}[/red]")
            return None

    def check_exists(self, model_id: str) -> Optional[Path]:
        """Check if a model is already downloaded."""
        folder_name = model_id.replace("/", "_")
        model_path = self.download_dir / folder_name

        if model_path.exists():
            if (model_path / "config.json").exists():
                return model_path

        return None

    def download_model(
        self,
        model_id: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Path]:
        """Download a model from HuggingFace Hub."""
        existing = self.check_exists(model_id)
        if existing and not force:
            console.print(f"[yellow]Model already exists:[/yellow] {existing}")
            return existing

        folder_name = model_id.replace("/", "_")
        output_path = self.download_dir / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        if include_patterns is None:
            include_patterns = []

        if exclude_patterns is None:
            exclude_patterns = ["*.md", "*.txt", ".gitattributes"]

        console.print(f"\n[cyan]Downloading:[/cyan] {model_id}")
        console.print(f"[cyan]Destination:[/cyan] {output_path}\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=None)

                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                    token=self.token,
                    allow_patterns=include_patterns if include_patterns else None,
                    ignore_patterns=exclude_patterns,
                )

                progress.update(task, completed=True)

            console.print(f"\n[green]Download complete![/green]")
            return Path(downloaded_path)

        except GatedRepoError:
            console.print(f"\n[red]This model is gated and requires authentication.[/red]")
            console.print("[yellow]To access this model:[/yellow]")
            console.print("1. Create a HuggingFace account at https://huggingface.co")
            console.print("2. Accept the model's license on its page")
            console.print("3. Create an access token at https://huggingface.co/settings/tokens")
            console.print("4. Set the HF_TOKEN environment variable or run 'huggingface-cli login'")
            return None
        except RepositoryNotFoundError:
            console.print(f"\n[red]Model not found: {model_id}[/red]")
            return None
        except Exception as e:
            console.print(f"\n[red]Download failed: {e}[/red]")
            return None

    def download_specific_files(
        self,
        model_id: str,
        files: List[str],
    ) -> Optional[Path]:
        """Download specific files from a model."""
        folder_name = model_id.replace("/", "_")
        output_path = self.download_dir / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[cyan]Downloading {len(files)} files from:[/cyan] {model_id}\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=len(files))

                for filename in files:
                    progress.update(task, description=f"Downloading {filename[:40]}...")

                    hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        local_dir=str(output_path),
                        local_dir_use_symlinks=False,
                        token=self.token,
                    )

                    progress.advance(task)

            console.print(f"\n[green]Download complete![/green]")
            return output_path

        except Exception as e:
            console.print(f"\n[red]Download failed: {e}[/red]")
            return None

    def list_downloaded(self) -> List[Dict[str, Any]]:
        """List all downloaded models."""
        models = []

        if not self.download_dir.exists():
            return models

        for item in self.download_dir.iterdir():
            if item.is_dir():
                config_file = item / "config.json"
                if config_file.exists():
                    total_size = sum(
                        f.stat().st_size for f in item.rglob('*') if f.is_file()
                    )
                    model_id = item.name.replace("_", "/", 1)
                    models.append({
                        'model_id': model_id,
                        'path': str(item),
                        'size': total_size,
                    })

        return models

    def delete_download(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        import shutil

        folder_name = model_id.replace("/", "_")
        model_path = self.download_dir / folder_name

        if model_path.exists():
            shutil.rmtree(model_path)
            return True
        return False

    def get_download_size_estimate(self, model_id: str, include_safetensors_only: bool = True) -> int:
        """Estimate download size for a model."""
        details = self.get_model_details(model_id)
        if not details:
            return 0

        total = 0
        has_safetensors = any(f['name'].endswith('.safetensors') for f in details.files)

        for file_info in details.files:
            name = file_info['name']
            size = file_info['size']

            if include_safetensors_only and has_safetensors:
                if name.endswith('.bin') and not name.endswith('tokenizer.bin'):
                    continue

            if name.endswith(('.md', '.txt', '.gitattributes')):
                continue

            total += size

        return total

    def print_search_results(self, results: List[ModelSearchResult]):
        """Print search results as a table."""
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(
            title="Search Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Model ID", style="white", max_width=40)
        table.add_column("Downloads", style="green", justify="right", width=12)
        table.add_column("Likes", style="yellow", justify="right", width=8)
        table.add_column("Task", style="cyan", width=18)

        for i, result in enumerate(results, 1):
            downloads = f"{result.downloads:,}" if result.downloads else "0"
            likes = f"{result.likes:,}" if result.likes else "0"
            task = result.pipeline_tag or "-"

            table.add_row(
                str(i),
                result.model_id[:38],
                downloads,
                likes,
                task[:16],
            )

        console.print(table)

    def print_model_details(self, details: ModelDetails):
        """Print detailed model information."""
        size_str = format_size(details.total_size)

        safetensors_count = sum(1 for f in details.files if f['name'].endswith('.safetensors'))
        bin_count = sum(1 for f in details.files if f['name'].endswith('.bin'))
        gguf_count = sum(1 for f in details.files if f['name'].endswith('.gguf'))

        panel_content = f"""[bold white]{details.model_id}[/bold white]

[cyan]Author:[/cyan]      {details.author}
[cyan]Downloads:[/cyan]   {details.downloads:,}
[cyan]Likes:[/cyan]       {details.likes:,}
[cyan]Task:[/cyan]        {details.pipeline_tag or 'N/A'}
[cyan]Total Size:[/cyan]  {size_str}

[cyan]Files:[/cyan]
  Safetensors: {safetensors_count}
  PyTorch bin: {bin_count}
  GGUF:        {gguf_count}
  Total:       {len(details.files)}"""

        if details.tags:
            tags_str = ", ".join(details.tags[:8])
            if len(details.tags) > 8:
                tags_str += f" (+{len(details.tags) - 8} more)"
            panel_content += f"\n\n[cyan]Tags:[/cyan] {tags_str}"

        console.print(Panel(
            panel_content,
            title="[bold]Model Details[/bold]",
            border_style="cyan",
            padding=(1, 2)
        ))

    def print_files_table(self, details: ModelDetails, limit: int = 20):
        """Print model files as a table."""
        table = Table(
            title="Model Files",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("File", style="white", max_width=50)
        table.add_column("Size", style="yellow", justify="right", width=12)
        table.add_column("Type", style="green", width=12)

        sorted_files = sorted(details.files, key=lambda x: x['size'], reverse=True)

        for file_info in sorted_files[:limit]:
            name = file_info['name']
            size = format_size(file_info['size'])

            if name.endswith('.safetensors'):
                ftype = "safetensors"
            elif name.endswith('.bin'):
                ftype = "pytorch"
            elif name.endswith('.gguf'):
                ftype = "gguf"
            elif name.endswith('.json'):
                ftype = "config"
            else:
                ftype = name.split('.')[-1] if '.' in name else "other"

            table.add_row(name[:48], size, ftype)

        console.print(table)

        if len(details.files) > limit:
            console.print(f"[dim]... and {len(details.files) - limit} more files[/dim]")

    # LoRA methods
    def download_lora(self, model_id: str, force: bool = False) -> Optional[Path]:
        """Download a LoRA adapter from HuggingFace Hub."""
        loras_dir = get_loras_dir()
        folder_name = model_id.replace("/", "_")
        output_path = loras_dir / folder_name

        if output_path.exists() and not force:
            if (output_path / "adapter_config.json").exists():
                console.print(f"[yellow]LoRA already exists:[/yellow] {output_path}")
                return output_path

        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[cyan]Downloading LoRA:[/cyan] {model_id}")
        console.print(f"[cyan]Destination:[/cyan] {output_path}\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading LoRA...", total=None)

                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                    token=self.token,
                    ignore_patterns=["*.md", "*.txt", ".gitattributes"],
                )

                progress.update(task, completed=True)

            # Validate required LoRA files exist
            adapter_config = output_path / "adapter_config.json"
            adapter_weights = output_path / "adapter_model.safetensors"
            adapter_weights_bin = output_path / "adapter_model.bin"

            if not adapter_config.exists():
                console.print(f"\n[red]ERROR: adapter_config.json not found![/red]")
                console.print("[dim]This may not be a valid LoRA adapter.[/dim]")
                return None

            if not adapter_weights.exists() and not adapter_weights_bin.exists():
                console.print(f"\n[red]ERROR: adapter weights not found![/red]")
                console.print("[dim]Expected adapter_model.safetensors or adapter_model.bin[/dim]")
                return None

            # Show LoRA info
            try:
                import json
                with open(adapter_config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                base_model = config.get('base_model_name_or_path', 'Unknown')
                rank = config.get('r', config.get('rank', '?'))
                console.print(f"\n[green]LoRA downloaded successfully![/green]")
                console.print(f"[dim]Base model: {base_model}[/dim]")
                console.print(f"[dim]Rank: {rank}[/dim]")
            except Exception:
                console.print(f"\n[green]LoRA downloaded successfully![/green]")

            console.print(f"[dim]Location: {output_path}[/dim]")
            return Path(downloaded_path)

        except GatedRepoError:
            console.print(f"\n[red]This LoRA is gated and requires authentication.[/red]")
            console.print("[yellow]Set HF_TOKEN or run 'huggingface-cli login'[/yellow]")
            return None
        except RepositoryNotFoundError:
            console.print(f"\n[red]LoRA not found: {model_id}[/red]")
            return None
        except Exception as e:
            console.print(f"\n[red]Download failed: {e}[/red]")
            return None

    def list_downloaded_loras(self) -> List[Dict[str, Any]]:
        """List all downloaded LoRA adapters."""
        import json
        loras = []
        loras_dir = get_loras_dir()

        if not loras_dir.exists():
            return loras

        for item in loras_dir.iterdir():
            if item.is_dir():
                adapter_config = item / "adapter_config.json"
                if adapter_config.exists():
                    total_size = sum(
                        f.stat().st_size for f in item.rglob('*') if f.is_file()
                    )
                    model_id = item.name.replace("_", "/", 1)

                    base_model = None
                    try:
                        with open(adapter_config, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            base_model = config.get('base_model_name_or_path', None)
                    except:
                        pass

                    loras.append({
                        'model_id': model_id,
                        'name': item.name,
                        'path': str(item),
                        'size': total_size,
                        'base_model': base_model,
                    })

        return loras

    def delete_lora(self, model_id: str) -> bool:
        """Delete a downloaded LoRA adapter."""
        import shutil

        loras_dir = get_loras_dir()
        folder_name = model_id.replace("/", "_")
        lora_path = loras_dir / folder_name

        if lora_path.exists():
            shutil.rmtree(lora_path)
            return True
        return False

    def search_loras(self, query: str, limit: int = 20) -> List[ModelSearchResult]:
        """Search for LoRA adapters on HuggingFace Hub."""
        search_query = f"{query} lora adapter"

        try:
            models = self.api.list_models(
                search=search_query,
                limit=limit,
                sort="downloads",
                direction=-1,
            )

            results = []
            for model in models:
                tags = model.tags or []
                is_lora = any(
                    tag.lower() in ['lora', 'peft', 'adapter']
                    for tag in tags
                ) or 'lora' in model.id.lower() or 'adapter' in model.id.lower()

                if is_lora:
                    results.append(ModelSearchResult(
                        model_id=model.id,
                        author=model.author or "Unknown",
                        downloads=model.downloads or 0,
                        likes=model.likes or 0,
                        pipeline_tag=model.pipeline_tag,
                        tags=tags,
                        last_modified=str(model.lastModified) if model.lastModified else "",
                    ))

            return results[:limit]

        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []

    def get_lora_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get details about a LoRA adapter from HuggingFace Hub."""
        try:
            model_info = self.api.model_info(model_id)

            # Get file list
            files = []
            total_size = 0
            has_adapter_config = False
            has_adapter_weights = False

            for sibling in model_info.siblings or []:
                files.append({
                    'filename': sibling.rfilename,
                    'size': sibling.size or 0,
                })
                total_size += sibling.size or 0

                if sibling.rfilename == 'adapter_config.json':
                    has_adapter_config = True
                if 'adapter_model' in sibling.rfilename:
                    has_adapter_weights = True

            # Try to get base model from card data
            base_model = None
            if model_info.card_data:
                base_model = getattr(model_info.card_data, 'base_model', None)

            return {
                'model_id': model_id,
                'author': model_info.author or 'Unknown',
                'downloads': model_info.downloads or 0,
                'likes': model_info.likes or 0,
                'tags': model_info.tags or [],
                'files': files,
                'total_size': total_size,
                'base_model': base_model,
                'has_adapter_config': has_adapter_config,
                'has_adapter_weights': has_adapter_weights,
                'last_modified': str(model_info.lastModified) if model_info.lastModified else '',
                'is_valid_lora': has_adapter_config and has_adapter_weights,
            }

        except Exception as e:
            console.print(f"[red]Error fetching details: {e}[/red]")
            return None

    def print_lora_details(self, details: Dict[str, Any]):
        """Print LoRA adapter details."""
        from rich.panel import Panel

        model_id = details.get('model_id', 'Unknown')
        base_model = details.get('base_model', 'Tuntematon')
        downloads = details.get('downloads', 0)
        likes = details.get('likes', 0)
        total_size = details.get('total_size', 0)
        is_valid = details.get('is_valid_lora', False)

        size_str = format_size(total_size) if total_size > 0 else 'Tuntematon'

        # List important files
        files = details.get('files', [])
        important_files = [f['filename'] for f in files if f['filename'] in
                         ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']]

        validity_str = "[green]Validi LoRA[/green]" if is_valid else "[yellow]Ei validi LoRA?[/yellow]"

        content = (
            f"[white]Model:[/white] {model_id}\n"
            f"[white]Base model:[/white] {base_model or 'Ei maaritelty'}\n"
            f"[white]Koko:[/white] {size_str}\n"
            f"[white]Downloads:[/white] {downloads:,}\n"
            f"[white]Likes:[/white] {likes:,}\n"
            f"[white]Tiedostot:[/white] {', '.join(important_files) or 'Ei loydetty'}\n"
            f"[white]Status:[/white] {validity_str}"
        )

        console.print(Panel(content, title="[bold]LoRA Details[/bold]", border_style="cyan"))

    def print_loras_table(self, loras: List[Dict[str, Any]]):
        """Print downloaded LoRAs as a table."""
        if not loras:
            console.print("[yellow]No LoRA adapters downloaded yet.[/yellow]")
            return

        table = Table(
            title=f"Downloaded LoRA Adapters ({len(loras)})",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Name", style="white", max_width=35)
        table.add_column("Base Model", style="cyan", max_width=30)
        table.add_column("Size", style="yellow", justify="right", width=10)

        for i, lora in enumerate(loras, 1):
            base = lora.get('base_model', '-')
            if base and len(base) > 28:
                base = "..." + base[-25:]

            table.add_row(
                str(i),
                lora['name'][:33],
                base or "-",
                format_size(lora['size']),
            )

        console.print(table)
