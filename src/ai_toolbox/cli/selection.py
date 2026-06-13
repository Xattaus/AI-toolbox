"""
AI TOOLBOX - Selection Helpers
==============================

Model and dataset selection helpers for CLI menus.
Extracted common patterns for selecting models from library,
downloads, and datasets.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import questionary

from .helpers import MENU_STYLE, create_choice, create_separator, press_any_key
from ..core.ui import format_size, print_warning, console
from ..core.paths import get_paths


# Type aliases for selection results
SelectionResult = Optional[Path]
TaggedSelectionResult = Optional[Tuple[str, Path]]


def select_model(
    library,
    prompt: str = "Select model:",
    format_filter: Optional[str] = None,
    include_downloads: bool = False,
    downloader=None,
    max_items: int = 15,
) -> SelectionResult:
    """
    Select a model from the library.

    Args:
        library: ModelLibrary instance
        prompt: Prompt message to display
        format_filter: Filter by format ('gguf', 'safetensors', 'pytorch')
        include_downloads: Whether to include downloaded HF models
        downloader: ModelDownloader instance (required if include_downloads=True)
        max_items: Maximum number of items to show per category

    Returns:
        Path to selected model, or None if cancelled
    """
    choices = []

    # Library models
    if format_filter:
        models = library.list_models(format_filter=format_filter)
    else:
        models = library.list_models()

    if models:
        choices.append(create_separator("Library Models"))
        for m in models[:max_items]:
            size = format_size(m.size_bytes)
            fmt = m.format.upper() if m.format else "?"
            title = f"[lib] {m.name[:30]:<30} {fmt:<12} {size:>10}"
            choices.append(create_choice(title, ("library", m.path)))

    # Downloaded HF models
    if include_downloads and downloader:
        downloaded = downloader.list_downloaded()
        if downloaded:
            choices.append(create_separator("Downloaded HF Models"))
            for d in downloaded[:max_items]:
                size = format_size(d['size'])
                name = d['model_id'].split('/')[-1] if '/' in d['model_id'] else d['model_id']
                title = f"[hf]  {name[:30]:<30} {'HF':<12} {size:>10}"
                choices.append(create_choice(title, ("download", d['path'])))

    if not choices:
        print_warning("No models found in library.")
        console.print("[dim]Add models via Model Download or refresh library.[/dim]")
        press_any_key()
        return None

    choices.append(create_separator())
    choices.append(create_choice("<-  Cancel", ("cancel", None)))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result[0] == "cancel":
        return None

    return Path(result[1])


def select_model_for_merge(
    library,
    downloader,
    prompt: str = "Select model:"
) -> SelectionResult:
    """
    Select a model suitable for merging (safetensors/pytorch).

    Args:
        library: ModelLibrary instance
        downloader: ModelDownloader instance
        prompt: Prompt message to display

    Returns:
        Path to selected model, or None if cancelled
    """
    choices = []

    # Get safetensors and pytorch models from library
    models = library.list_models(format_filter="safetensors")
    models.extend(library.list_models(format_filter="pytorch"))

    if models:
        choices.append(create_separator("Library Models"))
        for m in models[:15]:
            size = format_size(m.size_bytes)
            fmt = m.format.upper() if m.format else "?"
            title = f"[lib] {m.name[:30]:<30} {fmt:<12} {size:>10}"
            choices.append(create_choice(title, ("library", m.path)))

    # Downloaded HF models
    downloaded = downloader.list_downloaded()
    if downloaded:
        choices.append(create_separator("Downloaded HF Models"))
        for d in downloaded[:10]:
            size = format_size(d['size'])
            name = d['model_id'].split('/')[-1] if '/' in d['model_id'] else d['model_id']
            title = f"[hf]  {name[:30]:<30} {'HF':<12} {size:>10}"
            choices.append(create_choice(title, ("download", d['path'])))

    if not models and not downloaded:
        print_warning("No models in library or downloads.")
        console.print("[dim]Download a model first: Model Download[/dim]")
        press_any_key()
        return None

    choices.append(create_separator())
    choices.append(create_choice("<-  Cancel", ("cancel", None)))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result[0] == "cancel":
        return None

    return Path(result[1])


def select_model_for_training(
    library,
    downloader,
    prompt: str = "Select base model:"
) -> SelectionResult:
    """
    Select a model suitable for training (safetensors/pytorch).

    This is essentially the same as select_model_for_merge but with
    training-specific messaging.

    Args:
        library: ModelLibrary instance
        downloader: ModelDownloader instance
        prompt: Prompt message to display

    Returns:
        Path to selected model, or None if cancelled
    """
    choices = []

    # Get safetensors and pytorch models from library
    models = library.list_models(format_filter="safetensors")
    models.extend(library.list_models(format_filter="pytorch"))

    if models:
        choices.append(create_separator("Library Models"))
        for m in models[:15]:
            size = format_size(m.size_bytes)
            fmt = m.format.upper() if m.format else "?"
            title = f"[lib] {m.name[:30]:<30} {fmt:<12} {size:>10}"
            choices.append(create_choice(title, ("library", m.path)))

    # Downloaded HF models
    downloaded = downloader.list_downloaded()
    if downloaded:
        choices.append(create_separator("Downloaded HF Models"))
        for d in downloaded[:10]:
            size = format_size(d['size'])
            name = d['model_id'].split('/')[-1] if '/' in d['model_id'] else d['model_id']
            title = f"[hf]  {name[:30]:<30} {'HF':<12} {size:>10}"
            choices.append(create_choice(title, ("download", d['path'])))

    if not models and not downloaded:
        console.print("[yellow]No models in library or downloads.[/yellow]")
        console.print("[dim]Download a model first: Model Download -> Search/Download[/dim]\n")

    choices.append(create_separator())
    choices.append(create_choice("<-  Cancel", ("cancel", None)))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result[0] == "cancel":
        return None

    return Path(result[1])


def select_gguf_model(
    library,
    prompt: str = "Select GGUF model:",
    max_items: int = 20
) -> SelectionResult:
    """
    Select a GGUF model from the library.

    Args:
        library: ModelLibrary instance
        prompt: Prompt message to display
        max_items: Maximum number of items to show

    Returns:
        Path to selected GGUF model, or None if cancelled
    """
    gguf_models = library.list_models(format_filter="gguf")

    if not gguf_models:
        print_warning("No GGUF models in library.")
        console.print("[dim]Add models via Model Library -> Refresh Library or GGUF Converter[/dim]")
        console.print(f"[dim]GGUF directory: {get_paths().gguf_dir}[/dim]")
        press_any_key()
        return None

    choices = []
    for model in gguf_models[:max_items]:
        size = format_size(model.size_bytes)
        quant = model.quantization or "-"
        title = f"{model.name[:35]:<35} {quant:<8} {size:>10}"
        choices.append(create_choice(title, model.path))

    choices.append(create_separator())
    choices.append(create_choice("<- Palaa", "back"))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result == "back":
        return None

    return Path(result)


def select_dataset(
    dataset_prep,
    prompt: str = "Select dataset:",
    include_processed: bool = True
) -> SelectionResult:
    """
    Select a dataset from the datasets directory.

    Args:
        dataset_prep: DatasetPrep instance
        prompt: Prompt message to display
        include_processed: Whether to include processed datasets

    Returns:
        Path to selected dataset, or None if cancelled
    """
    datasets = dataset_prep.list_datasets(include_processed=include_processed)

    if not datasets:
        print_warning("No datasets found. Add .jsonl/.json/.csv files to datasets/ directory.")
        console.print(f"\n[dim]Datasets directory: {dataset_prep.datasets_dir}[/dim]")
        press_any_key()
        return None

    choices = []
    for ds in datasets:
        size = format_size(ds["size_bytes"])
        processed_tag = " [processed]" if ds.get("is_processed") else ""
        fmt = ds['format'] or '?'
        title = f"{ds['name']:<35} {fmt:<10} {size:>10}{processed_tag}"
        choices.append(create_choice(title, ds["path"]))

    choices.append(create_separator())
    choices.append(create_choice("<- Palaa", "back"))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result == "back":
        return None

    return result


def select_dataset_for_training(
    trainer,
    prompt: str = "Select dataset:"
) -> SelectionResult:
    """
    Select a dataset suitable for training.

    Args:
        trainer: LoRATrainer instance
        prompt: Prompt message to display

    Returns:
        Path to selected dataset, or None if cancelled
    """
    choices = []
    paths = get_paths()

    # Training datasets
    datasets = trainer.list_datasets()
    if datasets:
        choices.append(create_separator("Training Datasets"))
        for ds in datasets:
            size = format_size(ds["size_bytes"])
            fmt = ds['format'] or '?'
            title = f"[ds]  {ds['name'][:30]:<30} {fmt:<10} {size:>10}"
            choices.append(create_choice(title, ds["path"]))

    # Processed datasets
    processed_dir = paths.processed_dir
    if processed_dir.exists():
        processed = list(processed_dir.glob("*.jsonl")) + list(processed_dir.glob("*.json"))
        if processed:
            choices.append(create_separator("Processed Datasets"))
            for p in processed[:10]:
                size = format_size(p.stat().st_size)
                title = f"[pr]  {p.name[:30]:<30} {'JSONL':<10} {size:>10}"
                choices.append(create_choice(title, p))

    if not datasets and not (processed_dir.exists() and any(processed_dir.iterdir())):
        console.print("[yellow]No datasets found.[/yellow]")
        console.print("[dim]Create or process datasets: Dataset Prep[/dim]\n")

    choices.append(create_separator())
    choices.append(create_choice("<-  Cancel", "cancel"))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result == "cancel":
        return None

    return Path(result) if isinstance(result, str) else result


def select_lora_adapter(
    library=None,
    loras_dir: Path = None,
    prompt: str = "Select LoRA adapter:",
    max_items: int = 20
) -> SelectionResult:
    """
    Select a LoRA adapter.

    Args:
        library: Optional ModelLibrary instance (for library-indexed LoRAs)
        loras_dir: Directory to search for LoRA adapters
        prompt: Prompt message to display
        max_items: Maximum number of items to show

    Returns:
        Path to selected LoRA adapter, or None if cancelled
    """
    if loras_dir is None:
        loras_dir = get_paths().loras_dir

    choices = []

    # From library if available
    if library:
        lora_models = library.list_models(format_filter="lora")
        if lora_models:
            choices.append(create_separator("Library LoRAs"))
            for m in lora_models[:max_items]:
                size = format_size(m.size_bytes)
                title = f"[lib] {m.name[:35]:<35} {size:>10}"
                choices.append(create_choice(title, m.path))

    # Direct from loras directory
    if loras_dir.exists():
        lora_dirs = [d for d in loras_dir.iterdir() if d.is_dir()]
        if lora_dirs:
            choices.append(create_separator("LoRA Directory"))
            for d in lora_dirs[:max_items]:
                # Check if it's a valid LoRA directory
                adapter_config = d / "adapter_config.json"
                if adapter_config.exists():
                    # Calculate size
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    title = f"[dir] {d.name[:35]:<35} {format_size(size):>10}"
                    choices.append(create_choice(title, d))

    if not choices:
        print_warning("No LoRA adapters found.")
        console.print(f"[dim]LoRA directory: {loras_dir}[/dim]")
        press_any_key()
        return None

    choices.append(create_separator())
    choices.append(create_choice("<- Palaa", "back"))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result == "back":
        return None

    return Path(result)


def select_from_list(
    items: List[dict],
    prompt: str = "Select item:",
    name_key: str = "name",
    value_key: str = "path",
    format_item: Callable[[dict], str] = None,
    empty_message: str = "No items found.",
    back_label: str = "Back"
) -> Optional[Any]:
    """
    Generic selection helper for any list of items.

    Args:
        items: List of dictionaries with item data
        prompt: Prompt message to display
        name_key: Key for item name in dictionary
        value_key: Key for item value in dictionary
        format_item: Optional function to format item display
        empty_message: Message to show when list is empty
        back_label: Label for the back/cancel option

    Returns:
        Selected item value, or None if cancelled
    """
    if not items:
        print_warning(empty_message)
        press_any_key()
        return None

    choices = []
    for item in items:
        if format_item:
            title = format_item(item)
        else:
            title = str(item.get(name_key, "Unknown"))

        value = item.get(value_key, item)
        choices.append(create_choice(title, value))

    choices.append(create_separator())
    choices.append(create_choice(f"<-  {back_label}", "back"))

    result = questionary.select(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()

    if result is None or result == "back":
        return None

    return result


def multi_select_models(
    library,
    prompt: str = "Select models:",
    format_filter: Optional[str] = None,
    min_selections: int = 1,
    max_selections: Optional[int] = None
) -> List[Path]:
    """
    Select multiple models from the library.

    Args:
        library: ModelLibrary instance
        prompt: Prompt message to display
        format_filter: Filter by format ('gguf', 'safetensors', 'pytorch')
        min_selections: Minimum required selections
        max_selections: Maximum allowed selections

    Returns:
        List of paths to selected models (empty if cancelled)
    """
    if format_filter:
        models = library.list_models(format_filter=format_filter)
    else:
        models = library.list_models()

    if not models:
        print_warning("No models found in library.")
        press_any_key()
        return []

    choices = []
    for m in models:
        size = format_size(m.size_bytes)
        fmt = m.format.upper() if m.format else "?"
        title = f"{m.name[:30]:<30} {fmt:<12} {size:>10}"
        choices.append(create_choice(title, m.path))

    def validate_selection(selected):
        if len(selected) < min_selections:
            return f"Select at least {min_selections} model(s)"
        if max_selections and len(selected) > max_selections:
            return f"Select at most {max_selections} model(s)"
        return True

    result = questionary.checkbox(
        prompt,
        choices=choices,
        style=MENU_STYLE,
        validate=validate_selection
    ).ask()

    if result is None:
        return []

    return [Path(p) for p in result]
