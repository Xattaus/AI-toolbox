"""
AI TOOLBOX - Model Hub Commands
===============================

Unified CLI for model browsing, downloading, and library management.
Combines the functionality of download_cmd.py and library_cmd.py.
"""

from pathlib import Path
from typing import Optional, List

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
            console.print(f"[dim]Kirjasto: {stats['total_models']} mallia ({stats['total_size_gb']:.1f} GB) | Ladattu: {len(downloaded)}[/dim]\n")

            choices = [
                questionary.Choice(
                    title=format_menu_item("Selaa kirjastoa", "Kaikki mallit"),
                    value="browse"
                ),
                questionary.Separator("--- Lataa ------------------------------------"),
                questionary.Choice(
                    title=format_menu_item("Hae HuggingFacesta", "Etsi ja lataa malleja"),
                    value="search"
                ),
                questionary.Choice(
                    title=format_menu_item("Lataa ID:llä", "Suora lataus model ID:llä"),
                    value="direct"
                ),
                questionary.Choice(
                    title=format_menu_item("Suositut mallit", "Eniten ladatut"),
                    value="popular"
                ),
                questionary.Choice(
                    title=format_menu_item("Lataa LoRA", "LoRA-adapterit"),
                    value="lora"
                ),
                questionary.Separator("--- Hallinta ---------------------------------"),
                questionary.Choice(
                    title=format_menu_item("Lisää malli", "Lisää paikallinen malli"),
                    value="add"
                ),
                questionary.Choice(
                    title=format_menu_item("Päivitä kirjasto", "Skannaa uudet mallit"),
                    value="refresh"
                ),
                questionary.Choice(
                    title=format_menu_item("Skannaa kansio", "Etsi malleja kansiosta"),
                    value="scan"
                ),
                questionary.Choice(
                    title=format_menu_item("Kirjaston terveys", "Tarkista ja siivoa"),
                    value="health"
                ),
                questionary.Separator("----------------------------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Palaa", ""),
                    value="back"
                ),
            ]

            choice = questionary.select(
                "Valitse toiminto:",
                choices=choices,
                style=custom_style,
                qmark="#",
                pointer=">"
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
            console.print(f"  [dim]Models:[/dim] [cyan]{stats['total_models']}[/cyan]  |  "
                         f"[dim]Size:[/dim] [cyan]{stats['total_size_gb']:.1f} GB[/cyan]\n")

            choices = [
                questionary.Separator("--- Browse ---"),
                questionary.Choice(
                    title=format_menu_item("Categorized View", "Ryhmitelty nakyma (suositeltu)"),
                    value="categorized"
                ),
                questionary.Choice(
                    title=format_menu_item("All Models", "Kaikki mallit listana"),
                    value="all"
                ),
                questionary.Choice(
                    title=format_menu_item("Tree View", "Hierarkkinen puunakyma"),
                    value="tree"
                ),
                questionary.Separator("--- Filter ---"),
                questionary.Choice(
                    title=format_menu_item("GGUF Models", "Vain GGUF-mallit"),
                    value="gguf"
                ),
                questionary.Choice(
                    title=format_menu_item("SafeTensors", "HuggingFace-mallit"),
                    value="safetensors"
                ),
                questionary.Choice(
                    title=format_menu_item("LoRA Adapters", "Koulutetut adapterit"),
                    value="adapters"
                ),
                questionary.Choice(
                    title=format_menu_item("Merged Models", "Yhdistetyt mallit"),
                    value="merged"
                ),
                questionary.Choice(
                    title=format_menu_item("Ollama Models", "Ollama-mallit"),
                    value="ollama"
                ),
                questionary.Separator("--- Tools ---"),
                questionary.Choice(
                    title=format_menu_item("Search", "Hae kirjastosta"),
                    value="search"
                ),
                questionary.Choice(
                    title=format_menu_item("Cleanup", "Siivoa duplikaatit ja puuttuvat"),
                    value="cleanup"
                ),
                questionary.Separator("-----------------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Back", "Palaa"),
                    value="back"
                ),
            ]

            choice = questionary.select(
                "",
                choices=choices,
                style=custom_style,
                qmark="",
                pointer=">",
                instruction="(↑↓ valitse)"
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
            'safetensors': ('🏠', 'SafeTensors', 'green'),
            'gguf_f16': ('📦', 'GGUF F16/F32', 'yellow'),
            'gguf_quantized': ('⚡', 'GGUF Quantized', 'cyan'),
            'merged': ('🔀', 'Merged', 'magenta'),
            'ollama': ('🤖', 'Ollama', 'blue'),
            'adapter': ('🔧', 'Adapters', 'white'),
            'other': ('📄', 'Other', 'dim'),
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
        category_order = ['safetensors', 'gguf_f16', 'gguf_quantized', 'merged', 'ollama', 'adapter', 'other']

        for cat_key in category_order:
            models = grouped.get(cat_key, [])
            if not models:
                continue

            icon, cat_name, color = category_icons.get(cat_key, ('📄', cat_key, 'dim'))
            table.add_row("", f"[bold {color}]--- {icon} {cat_name} ({len(models)}) ---[/bold {color}]", "", "", "")

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
            questionary.Choice(
                title="📂 Avaa kansio          Open folder",
                value="folder"
            ),
            questionary.Choice(
                title="🏷️  Muokkaa tageja       Edit tags",
                value="tags"
            ),
        ]

        # Add conversion option for non-GGUF models
        if model.format != 'gguf':
            choices.append(questionary.Choice(
                title="🔄 Muunna GGUF:ksi      Convert to GGUF",
                value="convert"
            ))

        # Add Ollama option for GGUF models
        if model.format == 'gguf' and model.category != 'ollama':
            choices.append(questionary.Choice(
                title="🤖 Luo Ollama-malli     Create Ollama model",
                value="ollama"
            ))

        choices.extend([
            questionary.Separator(),
            questionary.Choice(
                title="🗑️  Poista              Remove",
                value="remove"
            ),
            questionary.Choice(
                title="⬅️  Palaa               Back",
                value="back"
            ),
        ])

        action = questionary.select(
            "Toiminto:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
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

        console.print(Panel(
            panel_content,
            title="[bold]Mallin tiedot[/bold]",
            border_style="cyan",
            padding=(1, 2)
        ))

        choices = [
            questionary.Choice(title="Convert to GGUF", value="convert"),
            questionary.Choice(title="Edit Tags", value="tags"),
            questionary.Choice(title="Open Folder", value="folder"),
            questionary.Choice(title="Remove", value="remove"),
            questionary.Separator(),
            questionary.Choice(title="Back", value="back"),
        ]

        action = questionary.select(
            "Toiminnot:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
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
        console.print("  [dim]🏠 = Base model  |  🔧 = LoRA adapter  |  🔀 = Merged  |  🤖 = Ollama[/dim]")
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
        """Search for models on HuggingFace Hub."""
        query = questionary.text(
            "Hakusana (esim. 'llama', 'mistral', 'qwen'):",
            style=custom_style,
        ).ask()

        if not query:
            return

        task_choices = [
            questionary.Choice(title="Kaikki tehtavat", value=None),
            questionary.Choice(title="Tekstin generointi (LLM)", value="text-generation"),
            questionary.Choice(title="Teksti-tekstiksi", value="text2text-generation"),
            questionary.Choice(title="Piirteiden poiminta", value="feature-extraction"),
        ]

        task = questionary.select(
            "Suodata tehtavan mukaan:",
            choices=task_choices,
            style=custom_style,
        ).ask()

        console.print(f"\n[cyan]Haetaan '{query}'...[/cyan]\n")

        results = self.downloader.search_models(query, limit=15, filter_task=task)

        if not results:
            print_warning("Malleja ei loytynyt")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        self._select_from_search_results(results)

    def _browse_popular_models(self):
        """Browse popular/trending models."""
        console.print("\n[cyan]Haetaan suosittuja malleja...[/cyan]\n")

        results = self.downloader.search_models("", limit=20, sort="downloads")

        if results:
            self._select_from_search_results(results)
        else:
            print_warning("Suosittujen mallien hakeminen epäonnistui")
            questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_from_search_results(self, results):
        """Allow user to select from search results using table."""
        while True:
            print_branded_header("Hakutulokset", f"{len(results)} mallia")

            # Create table for search results
            table = Table(
                title=f"[bold orange1]HuggingFace-mallit[/bold orange1]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                border_style="orange3",
                padding=(0, 1),
            )
            table.add_column("#", style="bold yellow", width=4, justify="right")
            table.add_column("Malli", style="white", min_width=40)
            table.add_column("Latauksia", style="cyan", width=12, justify="right")

            for i, result in enumerate(results, 1):
                downloads = f"{result.downloads:,}" if result.downloads else "0"
                model_name = result.model_id[:45] if len(result.model_id) <= 45 else result.model_id[:42] + "..."
                table.add_row(str(i), f"🤗 {model_name}", downloads)

            console.print(table)
            console.print()

            # Get selection by number
            answer = questionary.text(
                f"Valitse numero [1-{len(results)}] (0 = palaa)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                break

            try:
                idx = int(answer.strip())
                if 1 <= idx <= len(results):
                    self._show_download_details(results[idx - 1].model_id)
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(results)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

    def _show_download_details(self, model_id: str):
        """Show model details in preview panel and offer download."""
        console.print(f"\n[cyan]Haetaan tietoja: {model_id}...[/cyan]\n")

        details = self.downloader.get_model_details(model_id)
        if not details:
            return

        existing = self.downloader.check_exists(model_id)

        # Calculate file types
        safetensors_files = [f for f in details.files if f['name'].endswith('.safetensors')]
        bin_files = [f for f in details.files if f['name'].endswith('.bin') and 'pytorch' in f['name'].lower()]
        gguf_files = [f for f in details.files if f['name'].endswith('.gguf')]
        config_files = [f for f in details.files if f['name'].endswith('.json')]

        safetensors_size = sum(f['size'] for f in safetensors_files)
        total_size = details.total_size

        # Determine model type
        if gguf_files:
            model_type = "GGUF (valmis kaytettavaksi)"
        elif safetensors_files:
            model_type = "SafeTensors (muunnettavissa GGUF:ksi)"
        elif bin_files:
            model_type = "PyTorch (muunnettavissa GGUF:ksi)"
        else:
            model_type = "Tuntematon"

        # Preview panel
        preview = f"""[bold white]{details.model_id}[/bold white]
{'[yellow]Jo ladattu[/yellow]' if existing else ''}

[cyan]Author:[/cyan]       {details.author}
[cyan]Downloads:[/cyan]    {details.downloads:,}
[cyan]Likes:[/cyan]        {details.likes:,}
[cyan]Task:[/cyan]         {details.pipeline_tag or 'N/A'}

[bold]--- Tiedostot ---[/bold]
[cyan]Kokonaiskoko:[/cyan]  {format_size(total_size)}
[cyan]SafeTensors:[/cyan]   {len(safetensors_files)} kpl ({format_size(safetensors_size)})
[cyan]GGUF:[/cyan]          {len(gguf_files)} kpl
[cyan]Config:[/cyan]        {len(config_files)} kpl

[bold]--- Tyyppi ---[/bold]
{model_type}"""

        if details.tags:
            tags_display = ", ".join(details.tags[:6])
            if len(details.tags) > 6:
                tags_display += f" (+{len(details.tags) - 6})"
            preview += f"\n\n[cyan]Tags:[/cyan] [dim]{tags_display}[/dim]"

        console.print(Panel(
            preview,
            title="[bold]Model Preview[/bold]",
            border_style="cyan",
            padding=(1, 2)
        ))

        # Show largest files
        if details.files:
            console.print("\n[bold]Suurimmat tiedostot:[/bold]")
            sorted_files = sorted(details.files, key=lambda x: x['size'], reverse=True)[:5]
            for f in sorted_files:
                size_str = format_size(f['size'])
                name = f['name']
                if len(name) > 50:
                    name = "..." + name[-47:]
                console.print(f"  [dim]{size_str:>10}[/dim]  {name}")

        # Download options
        console.print()

        choices = []

        if gguf_files:
            choices.append(questionary.Choice(
                title=f"Download GGUF files only ({len(gguf_files)} files)",
                value="gguf"
            ))
        if safetensors_files:
            est_size = format_size(safetensors_size + sum(f['size'] for f in config_files))
            choices.append(questionary.Choice(
                title=f"Download SafeTensors ({est_size}) - suositeltu",
                value="safetensors"
            ))

        choices.extend([
            questionary.Choice(title=f"Download All ({format_size(total_size)})", value="full"),
            questionary.Choice(title="Download Config only", value="config"),
        ])

        if existing:
            choices.insert(0, questionary.Choice(title="Re-download (overwrite)", value="force"))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="Back", value="back"))

        action = questionary.select(
            "Download options:",
            choices=choices,
            style=custom_style,
        ).ask()

        if action == "back" or action is None:
            return

        include_patterns = None
        exclude_patterns = None
        force = False

        if action == "force":
            force = True
        elif action == "gguf":
            include_patterns = ["*.gguf", "*.json"]
            exclude_patterns = ["*.safetensors", "*.bin", "*.md"]
        elif action == "safetensors":
            include_patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"]
            exclude_patterns = ["*.bin", "*.md", "*.gguf"]
        elif action == "config":
            include_patterns = ["*.json", "*.txt", "tokenizer*"]
            exclude_patterns = ["*.safetensors", "*.bin", "*.md", "*.gguf"]

        # Download
        downloaded_path = self.downloader.download_model(
            model_id,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            force=force,
        )

        # Automatic library addition
        if downloaded_path:
            try:
                existing_in_lib = self.library.search_models(model_id)
                if not existing_in_lib:
                    entry = self.library.add_model(
                        path=str(downloaded_path),
                        source="huggingface",
                        source_id=model_id,
                    )
                    print_success(f"Lisatty automaattisesti Model Libraryyn: {entry.name}")
                else:
                    print_info("Malli on jo Model Libraryssa")
            except Exception as e:
                print_warning(f"Automaattinen kirjastolisays epäonnistui: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

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
                title="Enter LoRA ID              Syota HuggingFace ID",
                value="direct"
            ),
            questionary.Choice(
                title="Search LoRAs               Hae LoRA-adaptereita",
                value="search"
            ),
            questionary.Choice(
                title="View Downloaded            Nayta ladatut LoRAt",
                value="view"
            ),
            questionary.Separator(),
            questionary.Choice(
                title="Back                       Palaa",
                value="back"
            ),
        ]

        choice = questionary.select(
            "Download LoRA:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
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

            if questionary.confirm(
                "Lataa tama LoRA?",
                style=custom_style,
                default=True
            ).ask():
                result = self.downloader.download_lora(model_id)
                if result:
                    print_success(f"LoRA ladattu: {result}")
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
            name = result.model_id[:45] if len(result.model_id) <= 45 else result.model_id[:42] + "..."
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

            if questionary.confirm(
                "Lataa tama LoRA?",
                style=custom_style,
                default=True
            ).ask():
                result = self.downloader.download_lora(selected)
                if result:
                    print_success(f"LoRA ladattu: {result}")

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
            name = lora['name'][:35] if len(lora['name']) <= 35 else lora['name'][:32] + "..."
            size = format_size(lora.get('size', 0)) if lora.get('size') else "-"
            table.add_row(str(i), f"🔧 {name}", size)

        console.print(table)
        console.print()

        if questionary.confirm(
            "Haluatko poistaa jonkin LoRAn?",
            style=custom_style,
            default=False
        ).ask():
            answer = questionary.text(
                f"Poistettavan numero [1-{len(loras)}] (0 = peruuta)",
                style=custom_style,
            ).ask()

            if answer and answer.strip() not in ("", "0", "q"):
                try:
                    idx = int(answer.strip())
                    if 1 <= idx <= len(loras):
                        to_delete = loras[idx - 1]['model_id']
                        if questionary.confirm(
                            f"Vahvista poisto: {loras[idx - 1]['name']}?",
                            style=custom_style,
                            default=False
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
                    model['name'][:40],
                    model['format'].upper(),
                    format_size(model['size'])
                )

            console.print(table)

            if questionary.confirm(
                "Lisaa kaikki loydetyt mallit kirjastoon?",
                style=custom_style,
                default=True
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
        console.print(f"[cyan]Nykyiset tagit:[/cyan] {', '.join(model.tags) if model.tags else 'Ei'}")
        new_tags = questionary.text(
            "Syota tagit (pilkuilla erotettuna):",
            style=custom_style,
        ).ask()
        if new_tags:
            tags = [t.strip() for t in new_tags.split(',')]
            self.library.update_model(model.id, tags=tags)
            print_success("Tagit paivitetty")

    def _open_model_folder(self, model):
        """Open model folder in file manager."""
        import os
        import sys
        import subprocess

        path = Path(model.path)
        folder = path.parent if path.is_file() else path

        if sys.platform == 'win32':
            os.startfile(str(folder))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(folder)])
        else:
            subprocess.run(['xdg-open', str(folder)])

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
            console.print(f"\n[yellow]Varoitus: Talla mallilla on {len(children)} lapsimallia:[/yellow]")
            for child in children:
                category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}
                icon = category_icons.get(child.category, "📄")
                console.print(f"  {icon} {child.name} ({child.category})")

            if not questionary.confirm(
                "Poista silti? Lapsimallit menettavat viittauksen tahan malliin.",
                style=custom_style,
                default=False
            ).ask():
                return

        if questionary.confirm(
            f"Poista '{model.name}' kirjastosta?",
            style=custom_style,
            default=False
        ).ask():
            delete_files = questionary.confirm(
                "Poista myos tiedostot levylta?",
                style=custom_style,
                default=False
            ).ask()

            success = self.library.remove_model(model.id, delete_files=delete_files)
            if success:
                print_success("Malli poistettu")
            else:
                print_error("Mallin poisto epäonnistui")

    def _remove_ollama_model(self, model):
        """Remove Ollama model with full cleanup including ollama rm."""
        from ..integrations.ollama import OllamaManager

        ollama_name = model.ollama_info.get('ollama_name') if model.ollama_info else None
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
            siblings = [c for c in self.library.get_children(parent.id)
                        if c.id != model.id and c.category == "ollama"]

            if siblings:
                console.print(f"\n[yellow]Huom: Muut Ollama-mallit kayttavat samaa GGUF:ia:[/yellow]")
                for sib in siblings:
                    console.print(f"  - {sib.name}")
            else:
                cascade_option = True

        console.print()

        # Build deletion options
        choices = [
            questionary.Choice(
                title="Poista vain Ollama-malli (sailyta GGUF)",
                value="ollama_only"
            ),
        ]

        if cascade_option and parent:
            choices.append(questionary.Choice(
                title=f"Poista myos lahde-GGUF ({format_size(parent.size_bytes)} vapautuu)",
                value="cascade"
            ))

        choices.append(questionary.Choice(title="Peruuta", value="cancel"))

        action = questionary.select(
            "Poistotapa:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
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
        if questionary.confirm(
            "Suorita siivous?",
            default=True,
            style=custom_style
        ).ask():
            results = self.library.cleanup_library()

            console.print()
            if results['missing'] > 0 or results['duplicates'] > 0:
                print_success(f"Siivottu: {results['missing']} puuttuvaa, {results['duplicates']} duplikaattia")
            else:
                console.print("  [green]✓[/green] Kirjasto on jo puhdas!")

            # Show stats after
            stats_after = self.library.get_stats()
            if stats_before['total_models'] != stats_after['total_models']:
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
            if issues['broken_parents']:
                console.print(f"   - Rikkinaiset vanhemmat: {len(issues['broken_parents'])}")
            if issues['broken_children']:
                console.print(f"   - Rikkinaiset lapset: {len(issues['broken_children'])}")
            if issues['orphaned_children']:
                console.print(f"   - Orvot lapset: {len(issues['orphaned_children'])}")

            if questionary.confirm(
                "Korjaa ongelmat automaattisesti?",
                default=True,
                style=custom_style
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
                "Poista puuttuvat kirjastosta?",
                default=True,
                style=custom_style
            ).ask():
                for m in missing_files:
                    self.library.remove_model(m.id, delete_files=False)
                print_success(f"   Poistettu {len(missing_files)} merkintaa")
        else:
            console.print("[green]   ✓ Kaikki tiedostot loytyvat[/green]")

        # 3. Orphan check
        console.print("\n[bold]3. Orpo-tiedostot[/bold]")
        orphan_stats = self.library.get_orphan_stats()

        if orphan_stats['total_count'] > 0:
            console.print(
                f"[yellow]   {orphan_stats['total_count']} orpo-tiedostoa "
                f"({format_size(orphan_stats['total_size_bytes'])})[/yellow]"
            )

            # Show by category
            for cat, stats in orphan_stats['by_category'].items():
                if stats['count'] > 0:
                    console.print(f"   - {cat}: {stats['count']} kpl ({format_size(stats['size_bytes'])})")

            if questionary.confirm(
                "Poista orvot?",
                default=False,
                style=custom_style
            ).ask():
                deleted = self.library.cleanup_orphans(dry_run=False)
                print_success(f"   Poistettu {len(deleted)} orpo-tiedostoa")
        else:
            console.print("[green]   ✓ Ei orpo-tiedostoja[/green]")

        # 4. Category stats
        console.print("\n[bold]4. Levytilan jakautuminen[/bold]")
        category_stats = self.library.get_category_stats()
        stats = self.library.get_stats()

        categories_display = [
            ('base', '🏠 Base', 'green'),
            ('adapter', '🔧 Adapter', 'blue'),
            ('merged', '🔀 Merged', 'yellow'),
            ('ollama', '🤖 Ollama', 'cyan'),
        ]

        total_size = stats['total_size_bytes']

        for cat_key, cat_name, color in categories_display:
            cat_stats = category_stats.get(cat_key, {'count': 0, 'size_bytes': 0})
            size = cat_stats['size_bytes']
            count = cat_stats['count']
            pct = (size / total_size * 100) if total_size > 0 else 0

            console.print(
                f"   [{color}]{cat_name:12}[/{color}] "
                f"{count:3} kpl  {format_size(size):>10}  ({pct:5.1f}%)"
            )

        console.print(f"\n   [bold]Yhteensa: {stats['total_models']} mallia, {format_size(total_size)}[/bold]")

        console.print()
        questionary.press_any_key_to_continue(style=custom_style).ask()
