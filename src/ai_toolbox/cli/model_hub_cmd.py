"""
AI TOOLBOX - Model Hub Commands
===============================

Unified CLI for model browsing, downloading, and library management.
Combines the functionality of download_cmd.py and library_cmd.py.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..core.ui import (
    console,
    print_mini_banner,
    print_branded_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    format_size,
    format_menu_item,
    menu_separator,
    create_model_preview_card,
    create_model_table,
    select_model_from_table,
    build_model_choices,
    CATEGORY_CONFIG,
    MENU_STYLE,
)
from ..core.paths import get_paths
from ..models.downloader import ModelDownloader
from ..models.library import ModelLibrary
from ..models.hf_search import SearchFilters, SearchResult, ModelCardInfo
from ..models.hf_filters import (
    TASK_CATEGORIES,
    LIBRARIES,
    LICENSES,
    APP_CHOICES,
    SEARCH_PRESETS,
    SORT_OPTIONS,
    QUANTIZATION_QUALITY,
    get_quality_stars,
)

# Use unified menu style
custom_style = MENU_STYLE


class ModelHubCommands:
    """Unified CLI commands for model management."""

    def __init__(self, library: ModelLibrary, downloader: ModelDownloader, converter=None):
        """
        Initialize Model Hub commands.

        Args:
            library: Model library instance
            downloader: Model downloader instance
            converter: Optional GGUF converter for model conversion
        """
        self.library = library
        self.downloader = downloader
        self.converter = converter

    def model_hub_menu(self):
        """Model Hub main menu."""
        from ..core.ui import print_branded_header

        while True:
            print_branded_header("Model Hub", "Lataa, selaa ja hallitse malleja")

            # Show stats
            stats = self.library.get_stats()
            downloaded = self.downloader.list_downloaded()
            console.print(
                f"[dim]Kirjasto: {stats['total_models']} mallia ({stats['total_size_gb']:.1f} GB) | Ladattu: {len(downloaded)}[/dim]\n"
            )

            choices = [
                questionary.Choice(
                    title=format_menu_item("Selaa kirjastoa", "Kaikki mallit"), value="browse"
                ),
                menu_separator("Lataa"),
                questionary.Choice(
                    title=format_menu_item("Hae HuggingFacesta", "Etsi ja lataa malleja"),
                    value="search",
                ),
                questionary.Choice(
                    title=format_menu_item("Lataa ID:llä", "Suora lataus model ID:llä"),
                    value="direct",
                ),
                questionary.Choice(
                    title=format_menu_item("Suositut mallit", "Eniten ladatut"), value="popular"
                ),
                questionary.Choice(
                    title=format_menu_item("Lataa LoRA", "LoRA-adapterit"), value="lora"
                ),
                menu_separator("Hallinta"),
                questionary.Choice(
                    title=format_menu_item("Lisää malli", "Lisää paikallinen malli"), value="add"
                ),
                questionary.Choice(
                    title=format_menu_item("Päivitä kirjasto", "Skannaa uudet mallit"),
                    value="refresh",
                ),
                questionary.Choice(
                    title=format_menu_item("Skannaa kansio", "Etsi malleja kansiosta"), value="scan"
                ),
                questionary.Choice(
                    title=format_menu_item("Kirjaston terveys", "Tarkista ja siivoa"),
                    value="health",
                ),
                menu_separator(),
                questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
            ]

            choice = questionary.select(
                "Valitse toiminto:", choices=choices, style=custom_style, qmark="#", pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "browse":
                self._browse_library_menu()
            elif choice == "search":
                self._search_hf_models()
            elif choice == "direct":
                self._download_by_id()
            elif choice == "popular":
                self._browse_popular_models()
            elif choice == "lora":
                self._download_lora_wizard()
            elif choice == "add":
                self._add_model_to_library()
            elif choice == "refresh":
                self._refresh_library()
            elif choice == "scan":
                self._scan_for_models()
            elif choice == "health":
                self._library_health_check()

    # ==================== BROWSE LIBRARY ====================

    def _browse_library_menu(self):
        """Browse library sub-menu."""
        while True:
            print_mini_banner("Library Browser", "Selaa ja hallinnoi malleja")

            stats = self.library.get_stats()
            console.print(
                f"  [dim]Models:[/dim] [cyan]{stats['total_models']}[/cyan]  |  "
                f"[dim]Size:[/dim] [cyan]{stats['total_size_gb']:.1f} GB[/cyan]\n"
            )

            choices = [
                questionary.Separator("--- Browse ---"),
                questionary.Choice(
                    title=format_menu_item("Categorized View", "Ryhmitelty nakyma (suositeltu)"),
                    value="categorized",
                ),
                questionary.Choice(
                    title=format_menu_item("All Models", "Kaikki mallit listana"), value="all"
                ),
                questionary.Choice(
                    title=format_menu_item("Tree View", "Hierarkkinen puunakyma"), value="tree"
                ),
                questionary.Separator("--- Filter ---"),
                questionary.Choice(
                    title=format_menu_item("GGUF Models", "Vain GGUF-mallit"), value="gguf"
                ),
                questionary.Choice(
                    title=format_menu_item("SafeTensors", "HuggingFace-mallit"), value="safetensors"
                ),
                questionary.Choice(
                    title=format_menu_item("LoRA Adapters", "Koulutetut adapterit"),
                    value="adapters",
                ),
                questionary.Choice(
                    title=format_menu_item("Merged Models", "Yhdistetyt mallit"), value="merged"
                ),
                questionary.Choice(
                    title=format_menu_item("Ollama Models", "Ollama-mallit"), value="ollama"
                ),
                questionary.Separator("--- Tools ---"),
                questionary.Choice(
                    title=format_menu_item("Search", "Hae kirjastosta"), value="search"
                ),
                questionary.Choice(
                    title=format_menu_item("Cleanup", "Siivoa duplikaatit ja puuttuvat"),
                    value="cleanup",
                ),
                questionary.Separator("-----------------------------------"),
                questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
            ]

            choice = questionary.select(
                "",
                choices=choices,
                style=custom_style,
                qmark="",
                pointer=">",
                instruction="(↑↓ valitse)",
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "categorized":
                self._browse_models_categorized()
            elif choice == "all":
                self._browse_models()
            elif choice == "tree":
                self._show_tree_view()
            elif choice == "gguf":
                self._browse_models(format_filter="gguf")
            elif choice == "safetensors":
                self._browse_models(format_filter="safetensors")
            elif choice == "adapters":
                self._browse_models(category_filter="adapter")
            elif choice == "merged":
                self._browse_models(category_filter="merged")
            elif choice == "ollama":
                self._browse_models(category_filter="ollama")
            elif choice == "search":
                self._search_library()
            elif choice == "cleanup":
                self._cleanup_library()

    def _browse_models_categorized(self):
        """Browse models in categorized view using table."""
        # Get grouped models
        grouped = self.library.get_models_grouped_by_format()

        # Count total
        total = sum(len(models) for models in grouped.values())

        if total == 0:
            print_branded_header("Kirjasto", "Tyhjä")
            print_warning("Kirjasto on tyhjä.")
            console.print("[dim]Lataa malleja Model Hub -> Hae HuggingFacesta[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        print_branded_header("Kirjasto", f"{total} mallia")

        # Category config
        category_icons = {
            "safetensors": ("🏠", "SafeTensors", "green"),
            "gguf_f16": ("📦", "GGUF F16/F32", "yellow"),
            "gguf_quantized": ("⚡", "GGUF Quantized", "cyan"),
            "merged": ("🔀", "Merged", "magenta"),
            "ollama": ("🤖", "Ollama", "blue"),
            "adapter": ("🔧", "Adapters", "white"),
            "other": ("📄", "Other", "dim"),
        }

        # Build unified list for selection
        all_models = []
        idx = 1

        # Create table
        table = Table(
            title=f"[bold orange1]Mallikirjasto ({total})[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("Malli", style="white", min_width=30)
        table.add_column("Tyyppi", style="green", width=8, justify="center")
        table.add_column("Quant", style="yellow", width=8, justify="center")
        table.add_column("Koko", style="cyan", width=10, justify="right")

        # Order categories logically
        category_order = [
            "safetensors",
            "gguf_f16",
            "gguf_quantized",
            "merged",
            "ollama",
            "adapter",
            "other",
        ]

        for cat_key in category_order:
            models = grouped.get(cat_key, [])
            if not models:
                continue

            icon, cat_name, color = category_icons.get(cat_key, ("📄", cat_key, "dim"))
            table.add_row(
                "",
                f"[bold {color}]--- {icon} {cat_name} ({len(models)}) ---[/bold {color}]",
                "",
                "",
                "",
            )

            for m in models[:8]:  # Max 8 per category
                size = format_size(m.size_bytes) if m.size_bytes else "-"
                quant = m.quantization if m.quantization else "-"
                name = m.name[:30] if len(m.name) <= 30 else m.name[:27] + "..."
                fmt = m.format.upper() if m.format else "?"
                table.add_row(str(idx), f"{icon} {name}", fmt, quant, size)
                all_models.append(m)
                idx += 1

        console.print(table)
        console.print()

        # Get selection by number
        while True:
            answer = questionary.text(
                f"Valitse numero [1-{len(all_models)}] (0 = palaa)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                return

            try:
                sel_idx = int(answer.strip())
                if 1 <= sel_idx <= len(all_models):
                    selected = all_models[sel_idx - 1]
                    # Show detailed preview card
                    console.print()
                    console.print(create_model_preview_card(selected, show_path=True))
                    console.print()
                    # Model actions
                    self._model_actions_menu(selected)
                    return
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(all_models)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

    def _model_actions_menu(self, model):
        """Show actions menu for a selected model."""
        # Guard against string being passed instead of ModelEntry
        if isinstance(model, str):
            print_error(f"Virhe: Odotettiin ModelEntry-objektia, saatiin merkkijono: {model}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices = [
            questionary.Choice(title="📂 Avaa kansio          Open folder", value="folder"),
            questionary.Choice(title="🏷️  Muokkaa tageja       Edit tags", value="tags"),
        ]

        # Add conversion option for non-GGUF models
        if model.format != "gguf":
            choices.append(
                questionary.Choice(title="🔄 Muunna GGUF:ksi      Convert to GGUF", value="convert")
            )

        # Add Ollama option for GGUF models
        if model.format == "gguf" and model.category != "ollama":
            choices.append(
                questionary.Choice(
                    title="🤖 Luo Ollama-malli     Create Ollama model", value="ollama"
                )
            )

        choices.extend(
            [
                questionary.Separator(),
                questionary.Choice(title="🗑️  Poista              Remove", value="remove"),
                questionary.Choice(title="⬅️  Palaa               Back", value="back"),
            ]
        )

        action = questionary.select(
            "Toiminto:", choices=choices, style=custom_style, qmark=">>", pointer=">"
        ).ask()

        if action == "folder":
            self._open_model_folder(model)
        elif action == "tags":
            self._edit_model_tags(model)
        elif action == "convert":
            self._convert_model(model)
        elif action == "ollama":
            # Navigate to Ollama wizard
            from .ollama_cmd import run_ollama_wizard

            run_ollama_wizard()
        elif action == "remove":
            self._remove_model(model)

    def _browse_models(
        self,
        format_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        filter_models: Optional[List] = None,
    ):
        """Browse models with interactive selection."""
        # Käytä annettuja malleja tai hae kirjastosta
        if filter_models is not None:
            models = filter_models
        else:
            models = self.library.list_models(
                format_filter=format_filter,
                source_filter=source_filter,
                category_filter=category_filter,
            )

        if not models:
            print_warning("Kirjastosta ei loytynyt malleja.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}

        while True:
            print_branded_header("Kirjasto", f"{len(models)} mallia")

            # Create table for models
            table = Table(
                title=f"[bold orange1]Mallit[/bold orange1]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                border_style="orange3",
                padding=(0, 1),
            )
            table.add_column("#", style="bold yellow", width=4, justify="right")
            table.add_column("Malli", style="white", min_width=30)
            table.add_column("Tyyppi", style="green", width=8, justify="center")
            table.add_column("Quant", style="yellow", width=8, justify="center")
            table.add_column("Koko", style="cyan", width=10, justify="right")

            display_models = models[:20]
            for i, model in enumerate(display_models, 1):
                size_str = format_size(model.size_bytes) if model.size_bytes else "-"
                quant = model.quantization if model.quantization else "-"
                icon = category_icons.get(model.category, "📄")
                name = model.name[:35] if len(model.name) <= 35 else model.name[:32] + "..."
                fmt = model.format.upper() if model.format else "?"
                table.add_row(str(i), f"{icon} {name}", fmt, quant, size_str)

            console.print(table)
            console.print()

            # Get selection by number
            answer = questionary.text(
                f"Valitse numero [1-{len(display_models)}] (0 = palaa)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                break

            try:
                idx = int(answer.strip())
                if 1 <= idx <= len(display_models):
                    self._show_model_details(display_models[idx - 1].id)
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(display_models)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

    def _show_model_details(self, model_id: str):
        """Show detailed model information."""
        model = self.library.get_model(model_id)
        if not model:
            print_error("Mallia ei loytynyt")
            return

        category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}
        category_icon = category_icons.get(model.category, "📄")

        parent_name = "Ei"
        if model.parent_id:
            parent = self.library.get_model(model.parent_id)
            if parent:
                parent_name = parent.name

        children = self.library.get_children(model.id)
        children_str = f"{len(children)} mallia" if children else "Ei"

        console.print()
        panel_content = f"""[bold white]{category_icon} {model.name}[/bold white]

[cyan]ID:[/cyan]           {model.id}
[cyan]Kategoria:[/cyan]    {model.category.upper()}
[cyan]Formaatti:[/cyan]    {model.format.upper()}
[cyan]Kvantisointi:[/cyan] {model.quantization or 'Ei'}
[cyan]Koko:[/cyan]         {format_size(model.size_bytes)}
[cyan]Lahde:[/cyan]        {model.source}
[cyan]Lahde-ID:[/cyan]     {model.source_id or 'Ei'}
[cyan]Lisatty:[/cyan]      {model.added_date[:10]}
[cyan]Tagit:[/cyan]        {', '.join(model.tags) if model.tags else 'Ei'}

[yellow]Parent:[/yellow]       {parent_name}
[yellow]Children:[/yellow]     {children_str}

[dim]Polku: {model.path}[/dim]"""

        if model.training_info:
            info = model.training_info
            panel_content += f"""

[green]Training Info:[/green]
  Backend:    {info.get('backend', 'unknown')}
  Base model: {info.get('base_model', 'unknown')}
  Epochs:     {info.get('epochs', 'unknown')}"""

        if model.merge_info:
            info = model.merge_info
            panel_content += f"""

[green]Merge Info:[/green]
  Method:  {info.get('method', 'unknown')}
  Sources: {', '.join(info.get('sources', []))}"""

        if model.ollama_info:
            info = model.ollama_info
            panel_content += f"""

[green]Ollama Info:[/green]
  Name:     {info.get('ollama_name', 'unknown')}
  Template: {info.get('template', 'none')}"""

        console.print(
            Panel(
                panel_content,
                title="[bold]Mallin tiedot[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        choices = [
            questionary.Choice(
                title=format_menu_item("Convert to GGUF", "Muunna GGUF-muotoon"), value="convert"
            ),
            questionary.Choice(title=format_menu_item("Edit Tags", "Muokkaa tageja"), value="tags"),
            questionary.Choice(
                title=format_menu_item("Open Folder", "Avaa kansio"), value="folder"
            ),
            questionary.Choice(
                title=format_menu_item("Remove", "Poista kirjastosta"), value="remove"
            ),
            questionary.Separator(),
            questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
        ]

        action = questionary.select(
            "Toiminnot:", choices=choices, style=custom_style, qmark=">>", pointer=">"
        ).ask()

        if action == "convert":
            self._convert_model(model)
        elif action == "tags":
            self._edit_model_tags(model)
        elif action == "folder":
            self._open_model_folder(model)
        elif action == "remove":
            self._remove_model(model)

    def _show_tree_view(self):
        """Show model library as a hierarchical tree."""
        print_mini_banner("Model Tree View")
        self.library.print_tree()
        console.print("[dim]Selitykset:[/dim]")
        console.print(
            "  [dim]🏠 = Base model  |  🔧 = LoRA adapter  |  🔀 = Merged  |  🤖 = Ollama[/dim]"
        )
        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _search_library(self):
        """Search models in library."""
        query = questionary.text(
            "Hakusana:",
            style=custom_style,
        ).ask()

        if not query:
            return

        results = self.library.search_models(query)

        if not results:
            print_warning(f"Malleja ei loytynyt haulla '{query}'")
        else:
            console.print(f"\n[green]Loytyi {len(results)} vastaavaa mallia:[/green]\n")
            # Näytä vain hakutulokset, ei kaikkia malleja
            self._browse_models(filter_models=results)

        questionary.press_any_key_to_continue(style=custom_style).ask()

    # ==================== DOWNLOAD ====================

    def _search_hf_models(self):
        """Advanced HuggingFace search with multiple modes."""
        print_mini_banner("HuggingFace-haku", "Etsi malleja HuggingFacesta")

        # Search mode selection
        mode_choices = [
            questionary.Choice(
                title=format_menu_item("Pikahaku", "Tekstihaku suosituimmista"), value="quick"
            ),
            questionary.Choice(
                title=format_menu_item("Suodatettu haku", "Kaikki suodattimet"), value="filtered"
            ),
            questionary.Choice(
                title=format_menu_item("Suositut kategoriat", "Valmiit hakupohjat"), value="presets"
            ),
            questionary.Separator(),
            questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
        ]

        mode = questionary.select(
            "Hakutapa:", choices=mode_choices, style=custom_style, qmark="#", pointer=">"
        ).ask()

        if mode is None or mode == "back":
            return
        elif mode == "quick":
            self._search_quick()
        elif mode == "filtered":
            self._search_filtered()
        elif mode == "presets":
            self._search_presets()

    def _search_quick(self):
        """Quick text-based search."""
        query = questionary.text(
            "Hakusana (esim. 'llama', 'mistral', 'qwen', 'coder'):",
            style=custom_style,
        ).ask()

        if not query:
            return

        console.print(f"\n[cyan]Haetaan '{query}'...[/cyan]\n")

        # Use new search engine
        filters = SearchFilters(query=query)
        results, total = self.downloader.search_models_advanced(filters, limit=20)

        if not results:
            print_warning("Malleja ei loytynyt")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        self._display_search_results(results, total, f"Hakutulokset: '{query}'")

    def _search_filtered(self):
        """Full filter search interface."""
        filters = SearchFilters()

        # 1. Text search (optional)
        query = questionary.text(
            "Hakusana (valinnainen, Enter ohittaa):",
            style=custom_style,
        ).ask()
        filters.query = query if query else None

        # 2. Task category selection
        category_choices = [
            questionary.Choice(title="Kaikki tehtavat", value=None),
            questionary.Choice(title="NLP / Tekstimallit", value="NLP"),
            questionary.Choice(title="Kuva & Video", value="Vision"),
            questionary.Choice(title="Audio / Puhe", value="Audio"),
            questionary.Choice(title="Multimodaaliset (VLM)", value="Multimodal"),
        ]

        category = questionary.select(
            "Tehtavakategoria:",
            choices=category_choices,
            style=custom_style,
        ).ask()

        if category:
            # Show tasks in category
            tasks = TASK_CATEGORIES.get(category, {})
            task_choices = [questionary.Choice(title=v, value=k) for k, v in tasks.items()]

            if task_choices:
                selected_tasks = questionary.checkbox(
                    "Valitse tehtavat:",
                    choices=task_choices,
                    style=custom_style,
                ).ask()
                filters.tasks = selected_tasks or []

        # 3. Library/format filter
        library_choices = [
            questionary.Choice(title="Kaikki formaatit", value=None),
            questionary.Choice(title="GGUF (llama.cpp, Ollama)", value="gguf"),
            questionary.Choice(title="SafeTensors (HF Transformers)", value="safetensors"),
            questionary.Choice(title="Diffusers (kuvagenerointi)", value="diffusers"),
            questionary.Choice(title="PEFT/LoRA (adapterit)", value="peft"),
            questionary.Choice(title="ONNX", value="onnx"),
        ]

        library = questionary.select(
            "Formaatti/kirjasto:",
            choices=library_choices,
            style=custom_style,
        ).ask()
        filters.libraries = [library] if library else []

        # 4. Application compatibility
        app_choices = [
            questionary.Choice(title="Ei rajoitusta", value=None),
        ] + [questionary.Choice(title=name, value=app_id) for app_id, name in APP_CHOICES]

        app = questionary.select(
            "Yhteensopiva sovellus:",
            choices=app_choices,
            style=custom_style,
        ).ask()
        filters.apps = [app] if app else []

        # 5. Author/organization (optional)
        author = questionary.text(
            "Tekija/organisaatio (valinnainen, esim. 'meta-llama', 'mistralai'):",
            style=custom_style,
        ).ask()
        filters.author = author if author else None

        # 6. Gated filter
        gated_choices = [
            questionary.Choice(title="Kaikki mallit", value=None),
            questionary.Choice(title="Vain avoimet (ei kirjautumista)", value=False),
            questionary.Choice(title="Vain gated (vaatii hyvaksynnan)", value=True),
        ]

        gated = questionary.select(
            "Saatavuus:",
            choices=gated_choices,
            style=custom_style,
        ).ask()
        filters.gated = gated

        # 7. License filter
        common_licenses = [
            "apache-2.0",
            "mit",
            "llama3.1",
            "llama3.2",
            "llama3.3",
            "gemma",
            "cc-by-4.0",
            "cc-by-nc-4.0",
            "openrail",
        ]
        license_choices = [
            questionary.Choice(title="Kaikki lisenssit", value=None),
        ] + [
            questionary.Choice(
                title=LICENSES.get(lic, {}).get("name", lic),
                value=lic,
            )
            for lic in common_licenses
        ]

        license_id = questionary.select(
            "Lisenssi:",
            choices=license_choices,
            style=custom_style,
        ).ask()
        filters.license_filter = license_id

        # 8. Sort order
        sort_choices = [
            questionary.Choice(title="Lataukset (suosituin)", value="downloads"),
            questionary.Choice(title="Tykkäykset", value="likes"),
            questionary.Choice(title="Viimeksi paivitetty", value="lastModified"),
        ]

        sort = questionary.select(
            "Jarjestys:",
            choices=sort_choices,
            style=custom_style,
        ).ask()

        # Execute search
        console.print("\n[cyan]Haetaan suodatetuilla kriteereilla...[/cyan]\n")

        results, total = self.downloader.search_models_advanced(filters, sort=sort, limit=25)

        if not results:
            print_warning("Malleja ei loytynyt annetuilla suodattimilla")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        self._display_search_results(results, total, "Suodatetut tulokset")

    def _search_presets(self):
        """Search using predefined presets."""
        preset_choices = []
        for preset_id, preset_data in SEARCH_PRESETS.items():
            preset_choices.append(
                questionary.Choice(
                    title=format_menu_item(preset_data["name"], preset_data["description"]),
                    value=preset_id,
                )
            )

        preset_choices.append(questionary.Separator())
        preset_choices.append(
            questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back")
        )

        preset_id = questionary.select(
            "Valitse kategoria:", choices=preset_choices, style=custom_style, qmark="#", pointer=">"
        ).ask()

        if preset_id is None or preset_id == "back":
            return

        preset = SEARCH_PRESETS.get(preset_id)
        if not preset:
            return

        console.print(f"\n[cyan]Haetaan: {preset['name']}...[/cyan]\n")

        # Build filters from preset
        preset_filters = preset.get("filters", {})
        filters = SearchFilters(
            query=preset_filters.get("query"),
            tasks=preset_filters.get("tasks", []),
            libraries=preset_filters.get("libraries", []),
        )

        results, total = self.downloader.search_models_advanced(filters, limit=25)

        if not results:
            print_warning("Malleja ei loytynyt")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        self._display_search_results(results, total, preset["name"])

    def _display_search_results(self, results: List[SearchResult], total: int, title: str):
        """Display search results with rich metadata."""
        while True:
            print_branded_header("Hakutulokset", f"{len(results)}/{total} mallia")

            # Create rich table
            table = Table(
                title=f"[bold orange1]{title}[/bold orange1]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                border_style="orange3",
                padding=(0, 1),
            )

            table.add_column("#", style="bold yellow", width=4, justify="right")
            table.add_column("Malli", style="white", min_width=35)
            table.add_column("Latauksia", style="cyan", width=12, justify="right")
            table.add_column("Koko", style="yellow", width=8, justify="center")
            table.add_column("Lisenssi", style="green", width=12)
            table.add_column("Apps", style="magenta", width=14)

            for i, result in enumerate(results, 1):
                downloads = f"{result.downloads:,}" if result.downloads else "0"

                # Model name with truncation
                model_name = result.model_id
                if len(model_name) > 35:
                    model_name = model_name[:32] + "..."

                # Model size
                size = result.model_size or "-"

                # License with gated indicator
                license_str = result.license[:10] if result.license else "-"
                if result.gated:
                    license_str = f"[yellow]{license_str}*[/yellow]"

                # Compatible apps (first 2)
                if result.compatible_apps:
                    apps = ", ".join(result.compatible_apps[:2])
                    if len(result.compatible_apps) > 2:
                        apps += f" +{len(result.compatible_apps) - 2}"
                else:
                    apps = "-"

                # Format indicators
                format_icons = ""
                if result.has_gguf:
                    format_icons += "[cyan]G[/cyan]"
                if result.has_safetensors:
                    format_icons += "[green]S[/green]"

                model_display = f"{model_name} {format_icons}"

                table.add_row(str(i), model_display, downloads, size, license_str, apps)

            console.print(table)
            console.print("[dim]* = vaatii kirjautumisen | G = GGUF | S = SafeTensors[/dim]")
            console.print()

            # Selection
            answer = questionary.text(
                f"Valitse numero [1-{len(results)}] (0 = palaa)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                break

            try:
                idx = int(answer.strip())
                if 1 <= idx <= len(results):
                    selected = results[idx - 1]
                    self._show_model_card_details(selected.model_id)
                else:
                    print_warning(f"Valitse numero valilta 1-{len(results)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

    def _show_model_card_details(self, model_id: str):
        """Show detailed model card with rich metadata."""
        console.print(f"\n[cyan]Haetaan mallikortti: {model_id}...[/cyan]\n")

        card = self.downloader.get_model_card(model_id)
        if not card:
            print_error(f"Mallikortin hakeminen epaonnistui: {model_id}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Build model card display
        self._display_model_card(card)

        # Actions menu
        self._model_card_actions(card)

    def _display_model_card(self, card: ModelCardInfo):
        """Display a rich model card panel."""
        # Header
        gated_warning = ""
        if card.gated:
            gated_warning = "\n[yellow]Rajoitettu malli - vaatii hyvaksynnan[/yellow]"

        # Technical info table
        tech_info = []
        if card.parameter_count:
            tech_info.append(
                f"[cyan]Parametreja:[/cyan]     {card.model_size or self._format_params(card.parameter_count)}"
            )
        elif card.model_size:
            tech_info.append(f"[cyan]Koko:[/cyan]            {card.model_size}")

        if card.architecture:
            tech_info.append(f"[cyan]Arkkitehtuuri:[/cyan]   {card.architecture}")
        if card.context_length:
            tech_info.append(f"[cyan]Kontekstipituus:[/cyan] {card.context_length:,} tokenia")
        if card.library_name:
            tech_info.append(f"[cyan]Kirjasto:[/cyan]        {card.library_name}")

        tech_section = "\n".join(tech_info) if tech_info else "[dim]Ei teknisia tietoja[/dim]"

        # Metadata section
        meta_info = []
        if card.license:
            meta_info.append(f"[cyan]Lisenssi:[/cyan]        {card.license}")
        if card.languages:
            langs = ", ".join(card.languages[:5])
            if len(card.languages) > 5:
                langs += f" (+{len(card.languages) - 5})"
            meta_info.append(f"[cyan]Kielet:[/cyan]          {langs}")
        if card.base_model:
            meta_info.append(f"[cyan]Pohjamalli:[/cyan]      {card.base_model}")

        meta_info.append(f"[cyan]Latauksia:[/cyan]       {card.downloads:,}")
        meta_info.append(f"[cyan]Tykkäyksia:[/cyan]      {card.likes:,}")
        if card.last_modified:
            meta_info.append(f"[cyan]Paivitetty:[/cyan]      {card.last_modified[:10]}")

        meta_section = "\n".join(meta_info)

        # File info section
        file_info = []
        file_info.append(f"[cyan]Kokonaiskoko:[/cyan]    {format_size(card.total_size_bytes)}")

        if card.has_safetensors:
            file_info.append("[green]SafeTensors[/green]      saatavilla")
        if card.has_gguf:
            file_info.append(
                f"[green]GGUF[/green]             {len(card.gguf_variants)} varianttia"
            )

        file_section = "\n".join(file_info)

        # Compatibility section
        compat_info = []
        if card.compatible_apps:
            apps_str = ", ".join(card.compatible_apps[:4])
            if len(card.compatible_apps) > 4:
                apps_str += f" +{len(card.compatible_apps) - 4}"
            compat_info.append(f"[cyan]Sovellukset:[/cyan]     {apps_str}")

        compat_section = "\n".join(compat_info) if compat_info else ""

        # Build full panel
        panel_content = f"""[bold white]{card.model_id}[/bold white]{gated_warning}

[bold]--- Tekniset tiedot ---[/bold]
{tech_section}

[bold]--- Metatiedot ---[/bold]
{meta_section}

[bold]--- Tiedostot ---[/bold]
{file_section}"""

        if compat_section:
            panel_content += f"""

[bold]--- Yhteensopivuus ---[/bold]
{compat_section}"""

        if card.tags:
            tags_str = ", ".join(card.tags[:8])
            if len(card.tags) > 8:
                tags_str += f" (+{len(card.tags) - 8})"
            panel_content += f"\n\n[dim]Tagit: {tags_str}[/dim]"

        console.print(
            Panel(
                panel_content, title="[bold]Mallikortti[/bold]", border_style="cyan", padding=(1, 2)
            )
        )

    def _model_card_actions(self, card: ModelCardInfo):
        """Show actions for model card."""
        choices = []

        # Download options based on available formats
        if card.has_gguf and card.gguf_variants:
            choices.append(
                questionary.Choice(
                    title=format_menu_item("Lataa GGUF", f"{len(card.gguf_variants)} varianttia"),
                    value="gguf",
                )
            )

        if card.has_safetensors:
            choices.append(
                questionary.Choice(
                    title=format_menu_item("Lataa SafeTensors", "HuggingFace-muoto"),
                    value="safetensors",
                )
            )

        choices.extend(
            [
                questionary.Choice(
                    title=format_menu_item("Lataa kaikki", format_size(card.total_size_bytes)),
                    value="full",
                ),
                questionary.Choice(
                    title=format_menu_item("Nayta tiedostot", "Kaikki mallin tiedostot"),
                    value="files",
                ),
                questionary.Separator(),
                questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
            ]
        )

        action = questionary.select(
            "Toiminto:", choices=choices, style=custom_style, qmark=">>", pointer=">"
        ).ask()

        if action == "back" or action is None:
            return
        elif action == "gguf":
            self._select_gguf_variant(card)
        elif action == "safetensors":
            self._download_safetensors(card.model_id)
        elif action == "full":
            self._download_full_model(card.model_id)
        elif action == "files":
            self._show_model_files(card)

    def _select_gguf_variant(self, card: ModelCardInfo):
        """Show GGUF variant selection with quality indicators."""
        if not card.gguf_variants:
            print_warning("GGUF-tiedostoja ei loytynyt")
            return

        print_mini_banner("GGUF-variantit", f"{len(card.gguf_variants)} vaihtoehtoa")

        # Find recommended variant (Q4_K_M or highest quality that fits common VRAM)
        recommended_idx = None
        for i, variant in enumerate(card.gguf_variants):
            quant = variant.get("quantization", "")
            if quant in ["Q4_K_M", "Q5_K_M"]:
                recommended_idx = i
                break

        # Create table
        table = Table(
            title="[bold orange1]GGUF-variantit[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
        )

        table.add_column("#", width=4, justify="right")
        table.add_column("Tiedosto", min_width=35)
        table.add_column("Kvantisointi", width=12, justify="center")
        table.add_column("Koko", width=10, justify="right")
        table.add_column("Laatu", width=8, justify="center")
        table.add_column("VRAM", width=10, justify="right")

        for i, variant in enumerate(card.gguf_variants):
            quant = variant.get("quantization", "?")
            quality = variant.get("quality", 3.0)
            vram = variant.get("vram_estimate", 0)

            # Quality stars
            quality_stars = get_quality_stars(quant) if quant else "..."

            # Mark recommended
            mark = "[green]suositeltu[/green]" if i == recommended_idx else ""

            filename = variant["filename"]
            if len(filename) > 35:
                filename = "..." + filename[-32:]

            table.add_row(
                str(i + 1),
                filename,
                f"{quant} {mark}",
                format_size(variant["size"]),
                quality_stars,
                f"~{vram:.1f} GB" if vram else "-",
            )

        console.print(table)
        console.print("[dim]Laatu: perustuu perplexity-testeihin. Q4_K_M = paras laatu/koko[/dim]")
        console.print()

        # Selection
        answer = questionary.text(
            f"Valitse numero [1-{len(card.gguf_variants)}] (0 = palaa)",
            style=custom_style,
        ).ask()

        if answer is None or answer.strip() in ("", "0", "q"):
            return

        try:
            idx = int(answer.strip())
            if 1 <= idx <= len(card.gguf_variants):
                selected = card.gguf_variants[idx - 1]
                self._download_gguf_file(card.model_id, selected["filename"])
            else:
                print_warning(f"Valitse numero valilta 1-{len(card.gguf_variants)}")
        except ValueError:
            print_warning("Anna kelvollinen numero")

    def _download_gguf_file(self, model_id: str, filename: str):
        """Download a specific GGUF file."""
        console.print(f"\n[cyan]Ladataan: {filename}[/cyan]\n")

        downloaded_path = self.downloader.download_specific_files(model_id, files=[filename])

        if downloaded_path:
            # Add to library
            try:
                existing = self.library.search_models(model_id)
                if not existing:
                    entry = self.library.add_model(
                        path=str(downloaded_path / filename),
                        source="huggingface",
                        source_id=model_id,
                    )
                    print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epaonnistui: {e}")
        else:
            print_error("Lataus epaonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _download_safetensors(self, model_id: str):
        """Download SafeTensors format."""
        downloaded_path = self.downloader.download_model(
            model_id,
            include_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"],
            exclude_patterns=["*.bin", "*.md", "*.gguf"],
        )

        if downloaded_path:
            self._auto_add_to_library(downloaded_path, model_id)
        else:
            print_error("Lataus epaonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _download_full_model(self, model_id: str):
        """Download full model with all files."""
        downloaded_path = self.downloader.download_model(model_id)

        if downloaded_path:
            self._auto_add_to_library(downloaded_path, model_id)
        else:
            print_error("Lataus epaonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _auto_add_to_library(self, path: Path, model_id: str):
        """Automatically add downloaded model to library."""
        try:
            existing = self.library.search_models(model_id)
            if not existing:
                entry = self.library.add_model(
                    path=str(path),
                    source="huggingface",
                    source_id=model_id,
                )
                print_success(f"Lisatty automaattisesti kirjastoon: {entry.name}")
            else:
                print_info("Malli on jo kirjastossa")
        except Exception as e:
            print_warning(f"Kirjastolisays epaonnistui: {e}")

    def _show_model_files(self, card: ModelCardInfo):
        """Show all files in the model."""
        print_mini_banner("Mallin tiedostot", f"{len(card.files)} tiedostoa")

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Tiedosto", min_width=50)
        table.add_column("Koko", width=12, justify="right")
        table.add_column("Tyyppi", width=12)

        # Sort by size
        sorted_files = sorted(card.files, key=lambda x: x.get("size", 0), reverse=True)

        for f in sorted_files[:30]:
            filename = f.get("filename", "?")
            size = format_size(f.get("size", 0))

            # Determine type
            if filename.endswith(".safetensors"):
                ftype = "[green]safetensors[/green]"
            elif filename.endswith(".gguf"):
                ftype = "[cyan]gguf[/cyan]"
            elif filename.endswith(".bin"):
                ftype = "[yellow]pytorch[/yellow]"
            elif filename.endswith(".json"):
                ftype = "[dim]config[/dim]"
            else:
                ext = filename.split(".")[-1] if "." in filename else "?"
                ftype = f"[dim]{ext}[/dim]"

            if len(filename) > 50:
                filename = "..." + filename[-47:]

            table.add_row(filename, size, ftype)

        console.print(table)

        if len(card.files) > 30:
            console.print(f"[dim]... ja {len(card.files) - 30} muuta tiedostoa[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _format_params(self, count: int) -> str:
        """Format parameter count."""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.0f}M"
        else:
            return f"{count / 1_000:.0f}K"

    def _browse_popular_models(self):
        """Browse popular/trending models."""
        console.print("\n[cyan]Haetaan suosittuja malleja...[/cyan]\n")

        # Use new search engine for popular models
        filters = SearchFilters()
        results, total = self.downloader.search_models_advanced(filters, sort="downloads", limit=25)

        if results:
            self._display_search_results(results, total, "Suosituimmat mallit")
        else:
            print_warning("Suosittujen mallien hakeminen epaonnistui")
            questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_from_search_results(self, results):
        """Allow user to select from search results using table (legacy compatibility)."""
        # Convert legacy ModelSearchResult to new SearchResult format
        converted = []
        for r in results:
            converted.append(
                SearchResult(
                    model_id=r.model_id,
                    author=r.author,
                    downloads=r.downloads,
                    likes=r.likes,
                    pipeline_tag=r.pipeline_tag,
                    tags=r.tags,
                    last_modified=r.last_modified,
                )
            )

        self._display_search_results(converted, len(converted), "Hakutulokset")

    def _show_download_details(self, model_id: str):
        """Show model details using new model card view."""
        # Use the new model card view
        self._show_model_card_details(model_id)

    def _download_by_id(self):
        """Download model with direct ID input."""
        model_id = questionary.text(
            "Syota HuggingFace-mallin ID (esim. meta-llama/Llama-2-7b-hf):",
            style=custom_style,
        ).ask()

        if not model_id:
            return

        existing = self.downloader.check_exists(model_id)
        if existing:
            console.print(f"\n[yellow]Malli on jo ladattu:[/yellow] {existing}\n")
            if not questionary.confirm("Lataa silti?", style=custom_style, default=False).ask():
                return

        self._show_download_details(model_id)

    # ==================== LoRA DOWNLOAD ====================

    def _download_lora_wizard(self):
        """LoRA adapter download wizard."""
        print_mini_banner("Download LoRA Adapter")

        choices = [
            questionary.Choice(
                title=format_menu_item("Enter LoRA ID", "Syota HuggingFace ID"), value="direct"
            ),
            questionary.Choice(
                title=format_menu_item("Search LoRAs", "Hae LoRA-adaptereita"), value="search"
            ),
            questionary.Choice(
                title=format_menu_item("View Downloaded", "Nayta ladatut LoRAt"), value="view"
            ),
            questionary.Separator(),
            questionary.Choice(title="<- Palaa", value="back"),
        ]

        choice = questionary.select(
            "Download LoRA:", choices=choices, style=custom_style, qmark=">>", pointer=">"
        ).ask()

        if choice is None or choice == "back":
            return
        elif choice == "direct":
            self._download_lora_by_id()
        elif choice == "search":
            self._search_and_download_lora()
        elif choice == "view":
            self._view_downloaded_loras()

    def _download_lora_by_id(self):
        """Download LoRA directly by ID."""
        model_id = questionary.text(
            "Syota LoRA model ID (esim. vpakarinen/llama-8b-lora-uncensored-thinking):",
            style=custom_style,
        ).ask()

        if not model_id:
            return

        console.print(f"\n[cyan]Haetaan tietoja: {model_id}...[/cyan]")
        details = self.downloader.get_lora_details(model_id)

        if details:
            self.downloader.print_lora_details(details)
            console.print()

            if questionary.confirm("Lataa tama LoRA?", style=custom_style, default=True).ask():
                result = self.downloader.download_lora(model_id)
                if result:
                    print_success(f"LoRA ladattu: {result}")
                else:
                    print_error("LoRA-lataus epäonnistui")
        else:
            print_error("LoRA-adapteria ei loytynyt")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _search_and_download_lora(self):
        """Search and download LoRA adapters."""
        query = questionary.text(
            "Hakusana (esim. 'llama uncensored', 'mistral coding'):",
            style=custom_style,
        ).ask()

        if not query:
            return

        console.print(f"\n[cyan]Haetaan LoRA-adaptereita: '{query}'...[/cyan]\n")
        results = self.downloader.search_loras(query, limit=15)

        if not results:
            print_warning("Ei tuloksia. Kokeile eri hakusanoja.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        print_branded_header("LoRA-haku", f"{len(results)} tulosta")

        # Create table for LoRA results
        table = Table(
            title=f"[bold orange1]LoRA-adapterit[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("LoRA", style="white", min_width=40)
        table.add_column("Latauksia", style="cyan", width=12, justify="right")

        for i, result in enumerate(results, 1):
            downloads = f"{result.downloads:,}" if result.downloads else "0"
            name = (
                result.model_id[:45] if len(result.model_id) <= 45 else result.model_id[:42] + "..."
            )
            table.add_row(str(i), f"🔧 {name}", downloads)

        console.print(table)
        console.print()

        # Get selection by number
        answer = questionary.text(
            f"Valitse numero [1-{len(results)}] (0 = palaa)",
            style=custom_style,
        ).ask()

        if answer is None or answer.strip() in ("", "0", "q"):
            return

        try:
            idx = int(answer.strip())
            if not (1 <= idx <= len(results)):
                print_warning(f"Valitse numero väliltä 1-{len(results)}")
                return
            selected = results[idx - 1].model_id
        except ValueError:
            print_warning("Anna kelvollinen numero")
            return

        console.print(f"\n[cyan]Haetaan tietoja: {selected}...[/cyan]")
        details = self.downloader.get_lora_details(selected)

        if details:
            self.downloader.print_lora_details(details)
            console.print()

            if questionary.confirm("Lataa tama LoRA?", style=custom_style, default=True).ask():
                result = self.downloader.download_lora(selected)
                if result:
                    print_success(f"LoRA ladattu: {result}")
                else:
                    print_error("LoRA-lataus epäonnistui")
        else:
            print_error(f"LoRA-tietoja ei saatu: {selected}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _view_downloaded_loras(self):
        """Show downloaded LoRA adapters with table-based deletion."""
        loras = self.downloader.list_downloaded_loras()

        if not loras:
            print_warning("Ladattuja LoRA-adaptereita ei loytynyt")
            console.print(f"\n[dim]LoRA-kansio: {get_paths().loras_dir}[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        print_branded_header("Ladatut LoRAt", f"{len(loras)} adapteria")

        # Create table for downloaded LoRAs
        table = Table(
            title=f"[bold orange1]Ladatut LoRA-adapterit[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("LoRA", style="white", min_width=35)
        table.add_column("Koko", style="cyan", width=10, justify="right")

        for i, lora in enumerate(loras, 1):
            name = lora["name"][:35] if len(lora["name"]) <= 35 else lora["name"][:32] + "..."
            size = format_size(lora.get("size", 0)) if lora.get("size") else "-"
            table.add_row(str(i), f"🔧 {name}", size)

        console.print(table)
        console.print()

        if questionary.confirm(
            "Haluatko poistaa jonkin LoRAn?", style=custom_style, default=False
        ).ask():
            answer = questionary.text(
                f"Poistettavan numero [1-{len(loras)}] (0 = peruuta)",
                style=custom_style,
            ).ask()

            if answer and answer.strip() not in ("", "0", "q"):
                try:
                    idx = int(answer.strip())
                    if 1 <= idx <= len(loras):
                        to_delete = loras[idx - 1]["model_id"]
                        if questionary.confirm(
                            f"Vahvista poisto: {loras[idx - 1]['name']}?",
                            style=custom_style,
                            default=False,
                        ).ask():
                            if self.downloader.delete_lora(to_delete):
                                print_success("LoRA poistettu")
                            else:
                                print_error("Poisto epäonnistui")
                except ValueError:
                    print_warning("Anna kelvollinen numero")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    # ==================== LIBRARY MANAGEMENT ====================

    def _add_model_to_library(self):
        """Add model to library."""
        print_mini_banner("Lisaa malli")

        path = questionary.path(
            "Syota polku mallitiedostoon tai -kansioon:",
            style=custom_style,
        ).ask()

        if not path:
            return

        path = Path(path)
        if not path.exists():
            print_error(f"Polkua ei ole olemassa: {path}")
            return

        name = questionary.text(
            "Mallin nimi (jata tyhjäksi automaattiselle):",
            style=custom_style,
        ).ask()

        source_choices = [
            questionary.Choice(title="Paikallinen tiedosto", value="local"),
            questionary.Choice(title="HuggingFace-lataus", value="huggingface"),
            questionary.Choice(title="Muunnettu malli", value="converted"),
        ]

        source = questionary.select(
            "Mallin lahde:",
            choices=source_choices,
            style=custom_style,
        ).ask()

        source_id = None
        if source == "huggingface":
            source_id = questionary.text(
                "HuggingFace-mallin ID (esim. meta-llama/Llama-2-7b):",
                style=custom_style,
            ).ask()

        try:
            entry = self.library.add_model(
                path=str(path),
                name=name if name else None,
                source=source or "local",
                source_id=source_id,
            )
            print_success(f"Malli lisatty: {entry.name}")
        except Exception as e:
            print_error(f"Mallin lisaaminen epäonnistui: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _refresh_library(self):
        """Refresh library by re-scanning default directory."""
        print_mini_banner("Paivita kirjasto")

        paths = get_paths()
        models_dir = paths.models_dir

        console.print(f"[cyan]Skannataan: {models_dir}[/cyan]\n")

        before_count = len(self.library.list_models())

        found = self.library.scan_directory(str(models_dir), add_found=True)

        after_count = len(self.library.list_models())
        new_models = after_count - before_count

        if new_models > 0:
            print_success(f"Loydettiin {new_models} uutta mallia!")
        else:
            console.print("[green]Kirjasto on ajan tasalla.[/green]")

        console.print(f"\n[dim]Kirjastossa yhteensa: {after_count} mallia[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _scan_for_models(self):
        """Scan directory for models."""
        print_mini_banner("Skannaa malleja")

        directory = questionary.path(
            "Syota skannattava kansio:",
            style=custom_style,
            only_directories=True,
        ).ask()

        if not directory:
            return

        console.print("[cyan]Skannataan...[/cyan]")
        found = self.library.scan_directory(directory)

        if not found:
            print_warning("Malleja ei loytynyt kansiosta.")
        else:
            console.print(f"\n[green]Loytyi {len(found)} mallitiedostoa:[/green]\n")

            table = Table(box=box.SIMPLE)
            table.add_column("Nimi", style="white")
            table.add_column("Formaatti", style="green")
            table.add_column("Koko", style="cyan")

            for model in found[:20]:
                table.add_row(
                    model["name"][:40], model["format"].upper(), format_size(model["size"])
                )

            console.print(table)

            if questionary.confirm(
                "Lisaa kaikki loydetyt mallit kirjastoon?", style=custom_style, default=True
            ).ask():
                self.library.scan_directory(directory, add_found=True)
                print_success(f"Lisattiin {len(found)} mallia kirjastoon")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    # ==================== HELPER METHODS ====================

    def _convert_model(self, model):
        """Convert specific model."""
        if model.format == "gguf":
            print_warning("Tama malli on jo GGUF-muodossa")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        if self.converter:
            from .gguf_tools_cmd import GGUFToolsCommands

            gguf_cmd = GGUFToolsCommands(self.library, self.converter)
            gguf_cmd._run_conversion(model.path, model.name)
        else:
            print_warning("GGUF-muunnin ei ole kaytettavissa")
            questionary.press_any_key_to_continue(style=custom_style).ask()

    def _edit_model_tags(self, model):
        """Edit model tags."""
        console.print(
            f"[cyan]Nykyiset tagit:[/cyan] {', '.join(model.tags) if model.tags else 'Ei'}"
        )
        new_tags = questionary.text(
            "Syota tagit (pilkuilla erotettuna):",
            style=custom_style,
        ).ask()
        if new_tags:
            tags = [t.strip() for t in new_tags.split(",")]
            self.library.update_model(model.id, tags=tags)
            print_success("Tagit paivitetty")

    def _open_model_folder(self, model):
        """Open model folder in file manager."""
        import os
        import sys
        import subprocess

        path = Path(model.path)
        folder = path.parent if path.is_file() else path

        if sys.platform == "win32":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])

        print_success(f"Avattu: {folder}")

    def _remove_model(self, model):
        """Remove model from library with proper cleanup."""
        # Guard against string being passed instead of ModelEntry
        if isinstance(model, str):
            print_error(f"Virhe: Odotettiin ModelEntry-objektia, saatiin merkkijono: {model}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Check for Ollama model - use special deletion flow
        if model.category == "ollama":
            self._remove_ollama_model(model)
            return

        # Check for children that depend on this model
        children = self.library.get_children(model.id)
        if children:
            console.print(
                f"\n[yellow]Varoitus: Talla mallilla on {len(children)} lapsimallia:[/yellow]"
            )
            for child in children:
                category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}
                icon = category_icons.get(child.category, "📄")
                console.print(f"  {icon} {child.name} ({child.category})")

            if not questionary.confirm(
                "Poista silti? Lapsimallit menettavat viittauksen tahan malliin.",
                style=custom_style,
                default=False,
            ).ask():
                return

        if questionary.confirm(
            f"Poista '{model.name}' kirjastosta?", style=custom_style, default=False
        ).ask():
            delete_files = questionary.confirm(
                "Poista myos tiedostot levylta?", style=custom_style, default=False
            ).ask()

            success = self.library.remove_model(model.id, delete_files=delete_files)
            if success:
                print_success("Malli poistettu")
            else:
                print_error("Mallin poisto epäonnistui")

    def _remove_ollama_model(self, model):
        """Remove Ollama model with full cleanup including ollama rm."""
        from ..integrations.ollama import OllamaManager

        ollama_name = model.ollama_info.get("ollama_name") if model.ollama_info else None
        parent = self.library.get_parent(model.id)

        console.print(f"\n[bold cyan]Ollama-mallin poisto: {model.name}[/bold cyan]")

        # Show what will be deleted
        console.print(f"\n[bold]Poistettavat:[/bold]")
        if ollama_name:
            console.print(f"  1. Ollama-rekisterointi: [white]{ollama_name}[/white]")
        console.print(f"  2. Library-merkinta: [white]{model.id}[/white]")

        # Check for parent GGUF
        cascade_option = False
        if parent:
            console.print(f"\n[yellow]Lahde-GGUF: {parent.name}[/yellow]")
            console.print(f"  Polku: [dim]{parent.path}[/dim]")
            console.print(f"  Koko: [cyan]{format_size(parent.size_bytes)}[/cyan]")

            # Check if other Ollama models use the same GGUF
            siblings = [
                c
                for c in self.library.get_children(parent.id)
                if c.id != model.id and c.category == "ollama"
            ]

            if siblings:
                console.print(
                    f"\n[yellow]Huom: Muut Ollama-mallit kayttavat samaa GGUF:ia:[/yellow]"
                )
                for sib in siblings:
                    console.print(f"  - {sib.name}")
            else:
                cascade_option = True

        console.print()

        # Build deletion options
        choices = [
            questionary.Choice(
                title="Poista vain Ollama-malli (sailyta GGUF)", value="ollama_only"
            ),
        ]

        if cascade_option and parent:
            choices.append(
                questionary.Choice(
                    title=f"Poista myos lahde-GGUF ({format_size(parent.size_bytes)} vapautuu)",
                    value="cascade",
                )
            )

        choices.append(questionary.Choice(title="Peruuta", value="cancel"))

        action = questionary.select(
            "Poistotapa:", choices=choices, style=custom_style, qmark=">>", pointer=">"
        ).ask()

        if action == "cancel" or action is None:
            return

        # Initialize Ollama manager and delete from Ollama
        manager = OllamaManager()

        if ollama_name and manager.is_available():
            success, msg = manager.delete_model(ollama_name)
            if success:
                print_success(f"Poistettu Ollamasta: {ollama_name}")
            else:
                print_warning(f"Ollama-poisto epäonnistui: {msg}")
        elif ollama_name:
            print_warning("Ollama ei ole kaytettavissa - ohitetaan Ollama-poisto")

        # Delete from library
        self.library.remove_model(model.id, delete_files=False)
        print_success("Poistettu kirjastosta")

        # Cascade delete if requested
        if action == "cascade" and parent:
            self.library.remove_model(parent.id, delete_files=True)
            print_success(f"Poistettu lahde-GGUF: {parent.name}")

    def _cleanup_library(self):
        """Quick cleanup: remove duplicates and missing files."""
        print_mini_banner("Library Cleanup", "Siivoa duplikaatit ja puuttuvat tiedostot")

        console.print("  [cyan]Tarkistetaan kirjasto...[/cyan]\n")

        # Check for issues before cleanup
        missing = self.library.find_missing_files()
        stats_before = self.library.get_stats()

        # Show what will be cleaned
        console.print(f"  [dim]Malleja yhteensa:[/dim]  {stats_before['total_models']}")
        console.print(f"  [dim]Puuttuvat:[/dim]        {len(missing)}")

        # Run cleanup
        if questionary.confirm("Suorita siivous?", default=True, style=custom_style).ask():
            results = self.library.cleanup_library()

            console.print()
            if results["missing"] > 0 or results["duplicates"] > 0:
                print_success(
                    f"Siivottu: {results['missing']} puuttuvaa, {results['duplicates']} duplikaattia"
                )
            else:
                console.print("  [green]✓[/green] Kirjasto on jo puhdas!")

            # Show stats after
            stats_after = self.library.get_stats()
            if stats_before["total_models"] != stats_after["total_models"]:
                console.print(f"\n  [dim]Malleja nyt:[/dim] {stats_after['total_models']}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _library_health_check(self):
        """Run comprehensive library health check."""
        print_mini_banner("Library Health", "Kirjaston terveys")

        console.print("[cyan]Tarkistetaan kirjaston tila...[/cyan]\n")

        # 1. Relationship validation
        console.print("[bold]1. Suhteiden tarkistus[/bold]")
        issues = self.library.validate_relationships()
        total_issues = sum(len(v) for v in issues.values())

        if total_issues > 0:
            console.print(f"[yellow]   Loytyi {total_issues} suhdeongelma(a):[/yellow]")
            if issues["broken_parents"]:
                console.print(f"   - Rikkinaiset vanhemmat: {len(issues['broken_parents'])}")
            if issues["broken_children"]:
                console.print(f"   - Rikkinaiset lapset: {len(issues['broken_children'])}")
            if issues["orphaned_children"]:
                console.print(f"   - Orvot lapset: {len(issues['orphaned_children'])}")

            if questionary.confirm(
                "Korjaa ongelmat automaattisesti?", default=True, style=custom_style
            ).ask():
                repairs = self.library.repair_relationships()
                print_success(f"   Korjattu {repairs} ongelmaa")
        else:
            console.print("[green]   ✓ Suhteet kunnossa[/green]")

        # 2. File existence check
        console.print("\n[bold]2. Tiedostojen olemassaolo[/bold]")
        missing_files = self.library.find_missing_files()

        if missing_files:
            console.print(f"[yellow]   {len(missing_files)} mallia puuttuu levylta:[/yellow]")
            for m in missing_files[:5]:
                console.print(f"   - {m.name}")
            if len(missing_files) > 5:
                console.print(f"   [dim]... ja {len(missing_files) - 5} muuta[/dim]")

            if questionary.confirm(
                "Poista puuttuvat kirjastosta?", default=True, style=custom_style
            ).ask():
                for m in missing_files:
                    self.library.remove_model(m.id, delete_files=False)
                print_success(f"   Poistettu {len(missing_files)} merkintaa")
        else:
            console.print("[green]   ✓ Kaikki tiedostot loytyvat[/green]")

        # 3. Orphan check
        console.print("\n[bold]3. Orpo-tiedostot[/bold]")
        orphan_stats = self.library.get_orphan_stats()

        if orphan_stats["total_count"] > 0:
            console.print(
                f"[yellow]   {orphan_stats['total_count']} orpo-tiedostoa "
                f"({format_size(orphan_stats['total_size_bytes'])})[/yellow]"
            )

            # Show by category
            for cat, stats in orphan_stats["by_category"].items():
                if stats["count"] > 0:
                    console.print(
                        f"   - {cat}: {stats['count']} kpl ({format_size(stats['size_bytes'])})"
                    )

            if questionary.confirm("Poista orvot?", default=False, style=custom_style).ask():
                deleted = self.library.cleanup_orphans(dry_run=False)
                print_success(f"   Poistettu {len(deleted)} orpo-tiedostoa")
        else:
            console.print("[green]   ✓ Ei orpo-tiedostoja[/green]")

        # 4. Category stats
        console.print("\n[bold]4. Levytilan jakautuminen[/bold]")
        category_stats = self.library.get_category_stats()
        stats = self.library.get_stats()

        categories_display = [
            ("base", "🏠 Base", "green"),
            ("adapter", "🔧 Adapter", "blue"),
            ("merged", "🔀 Merged", "yellow"),
            ("ollama", "🤖 Ollama", "cyan"),
        ]

        total_size = stats["total_size_bytes"]

        for cat_key, cat_name, color in categories_display:
            cat_stats = category_stats.get(cat_key, {"count": 0, "size_bytes": 0})
            size = cat_stats["size_bytes"]
            count = cat_stats["count"]
            pct = (size / total_size * 100) if total_size > 0 else 0

            console.print(
                f"   [{color}]{cat_name:12}[/{color}] "
                f"{count:3} kpl  {format_size(size):>10}  ({pct:5.1f}%)"
            )

        console.print(
            f"\n   [bold]Yhteensa: {stats['total_models']} mallia, {format_size(total_size)}[/bold]"
        )

        console.print()
        questionary.press_any_key_to_continue(style=custom_style).ask()
