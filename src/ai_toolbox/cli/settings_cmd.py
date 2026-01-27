"""
AI TOOLBOX - Settings Commands
==============================

CLI commands for settings, configuration, and system info.
"""

from pathlib import Path
from typing import Optional

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
    MENU_STYLE,
)
from ..core.paths import get_paths
from ..models.downloader import ModelDownloader
from ..models.library import ModelLibrary

# Use unified menu style
custom_style = MENU_STYLE


class SettingsCommands:
    """CLI commands for settings and configuration."""

    def __init__(
        self,
        downloader: Optional[ModelDownloader] = None,
        library: Optional[ModelLibrary] = None,
    ):
        """
        Initialize settings commands.

        Args:
            downloader: Optional model downloader instance for HF token status
            library: Optional model library instance for orphan cleanup
        """
        self.downloader = downloader
        self.library = library

    def settings_menu(self):
        """Settings sub-menu."""
        while True:
            print_branded_header("Asetukset", "Polut, tokenit ja järjestelmä")

            choices = [
                questionary.Separator("--- Konfiguraatio ----------------------------"),
                questionary.Choice(
                    title=format_menu_item("Show All Paths", "Näytä kaikki polut"),
                    value="show_paths"
                ),
                questionary.Choice(
                    title=format_menu_item("HuggingFace Token", "HF-token asetukset"),
                    value="hf_token"
                ),
                questionary.Choice(
                    title=format_menu_item("Clear Cache", "Tyhjennä välimuisti"),
                    value="clear_cache"
                ),
                questionary.Separator("--- Kirjasto ---------------------------------"),
                questionary.Choice(
                    title=format_menu_item("Library Cleanup", "Siivoa duplikaatit"),
                    value="library_cleanup"
                ),
                questionary.Choice(
                    title=format_menu_item("Orphan Cleanup", "Orpo-tiedostojen siivous"),
                    value="orphan_cleanup"
                ),
                questionary.Choice(
                    title=format_menu_item("Disk Analysis", "Levytilan analyysi"),
                    value="disk_analysis"
                ),
                questionary.Separator("--- Järjestelmä ------------------------------"),
                questionary.Choice(
                    title=format_menu_item("System Info", "Järjestelmätiedot"),
                    value="sysinfo"
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
            elif choice == "show_paths":
                self._show_paths()
            elif choice == "hf_token":
                self._hf_token_settings()
            elif choice == "clear_cache":
                self._clear_cache()
            elif choice == "library_cleanup":
                self._library_cleanup()
            elif choice == "orphan_cleanup":
                self._orphan_cleanup()
            elif choice == "disk_analysis":
                self._disk_analysis()
            elif choice == "sysinfo":
                self._show_system_info()

    def _show_paths(self):
        """Show all configured paths."""
        paths = get_paths()
        console.print(Panel(
            f"""[bold white]AI Toolbox - Portable Paths[/bold white]

[cyan]Root:[/cyan]           {paths.root}
[cyan]Models:[/cyan]         {paths.models_dir}
[cyan]Downloads:[/cyan]      {paths.downloads_dir}
[cyan]GGUF:[/cyan]           {paths.gguf_dir}
[cyan]Library file:[/cyan]   {paths.library_file}
[cyan]llama.cpp:[/cyan]      {paths.llama_cpp_dir}
[cyan]Config:[/cyan]         {paths.config_dir}
[cyan]Datasets:[/cyan]       {paths.datasets_dir}
[cyan]LoRAs:[/cyan]          {paths.loras_dir}
[cyan]Benchmarks:[/cyan]     {paths.benchmarks_dir}

[dim]Kaikki polut ovat suhteessa AI Toolboxin asennuskansioon.
Voit siirtaa koko kansion (esim. USB-tikulle) ja kaikki toimii.[/dim]""",
            title="[bold]Path Configuration[/bold]",
            border_style="cyan"
        ))
        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _hf_token_settings(self):
        """HuggingFace token settings."""
        print_mini_banner("HuggingFace Token")

        if self.downloader and self.downloader.token:
            console.print("[green]HF-token on asetettu[/green]")
            token_preview = self.downloader.token[:8] + "..." if len(self.downloader.token) > 8 else "***"
            console.print(f"[dim]Token: {token_preview}[/dim]\n")
        else:
            console.print("[yellow]HF-tokenia ei ole asetettu[/yellow]")
            console.print("[dim]Aseta HF_TOKEN-ymparistomuuttuja yksityisille malleille[/dim]\n")

        console.print("[bold white]Kuinka asettaa HuggingFace token:[/bold white]")
        console.print("")
        console.print("[cyan]Windows (PowerShell):[/cyan]")
        console.print('  $env:HF_TOKEN = "hf_xxxxxxxxxxxx"')
        console.print("")
        console.print("[cyan]Windows (CMD):[/cyan]")
        console.print('  set HF_TOKEN=hf_xxxxxxxxxxxx')
        console.print("")
        console.print("[cyan]Linux/Mac:[/cyan]")
        console.print('  export HF_TOKEN="hf_xxxxxxxxxxxx"')
        console.print("")
        console.print("[dim]Tokenin saat osoitteesta: https://huggingface.co/settings/tokens[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _clear_cache(self):
        """Clear cache files."""
        print_mini_banner("Clear Cache")

        paths = get_paths()
        temp_dir = paths.root / "temp"
        cache_dirs = [
            temp_dir,
            paths.root / ".cache",
        ]

        # Calculate cache size
        total_size = 0
        file_count = 0
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
                        file_count += 1

        if total_size == 0:
            console.print("[dim]Valimuisti on tyhja.[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        size_str = format_size(total_size)
        console.print(f"[white]Valimuistin koko:[/white] {size_str}")
        console.print(f"[white]Tiedostoja:[/white] {file_count}")
        console.print(f"[dim]Sijainti: {temp_dir}[/dim]\n")

        if questionary.confirm(
            "Tyhjenna kaikki valimuistitetut tiedot?",
            default=False,
            style=custom_style
        ).ask():
            import shutil
            cleared = 0
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    try:
                        shutil.rmtree(cache_dir)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cleared += 1
                    except Exception as e:
                        print_warning(f"Virhe tyhjennyksessa: {e}")

            if cleared > 0:
                print_success(f"Valimuisti tyhjennetty ({size_str} vapautettu)")
            else:
                print_warning("Tyhjennyksessa tapahtui virheita")

            questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_system_info(self):
        """Show system information."""
        print_mini_banner("System Info")

        import platform
        import psutil

        # CPU info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)

        # Memory info
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
        used_ram_gb = mem.used / (1024**3)
        ram_percent = mem.percent

        # Disk info
        paths = get_paths()
        try:
            disk = psutil.disk_usage(str(paths.root))
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_percent = disk.percent
        except Exception:
            disk_total_gb = 0
            disk_free_gb = 0
            disk_percent = 0

        # GPU info
        gpu_info = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_info = f"  [green]Saatavilla:[/green] Kylla\n  Nimi: {gpu_name}\n  Muisti: {gpu_mem:.1f} GB"
            else:
                gpu_info = "  Saatavilla: Ei (CUDA ei loydy)"
        except ImportError:
            gpu_info = "  Saatavilla: ? (torch ei asennettu)"

        console.print(Panel(
            f"[bold white]Kayttojarjestelma[/bold white]\n"
            f"  OS: {platform.system()} {platform.release()}\n"
            f"  Python: {platform.python_version()}\n\n"
            f"[bold white]CPU[/bold white]\n"
            f"  Ytimet (looginen): {cpu_count}\n"
            f"  Ytimet (fyysinen): {cpu_count_physical}\n\n"
            f"[bold white]RAM[/bold white]\n"
            f"  Yhteensa: {total_ram_gb:.1f} GB\n"
            f"  Kaytettavissa: {available_ram_gb:.1f} GB\n"
            f"  Kaytossa: {used_ram_gb:.1f} GB ({ram_percent:.0f}%)\n\n"
            f"[bold white]Levy ({paths.root.drive or 'root'})[/bold white]\n"
            f"  Yhteensa: {disk_total_gb:.1f} GB\n"
            f"  Vapaana: {disk_free_gb:.1f} GB ({100-disk_percent:.0f}%)\n\n"
            f"[bold white]GPU[/bold white]\n"
            f"{gpu_info}",
            title="[bold]System Information[/bold]",
            border_style="cyan"
        ))

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _check_llama_cpp_status(self) -> dict:
        """Check llama.cpp installation status."""
        paths = get_paths()
        llama_cpp_path = paths.llama_cpp_dir

        status = {
            "installed": False,
            "path": str(llama_cpp_path),
            "convert_script": None,
            "quantize_binary": None,
        }

        if not llama_cpp_path.exists():
            return status

        # Check for convert script
        convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
        if convert_script.exists():
            status["convert_script"] = str(convert_script)

        # Check for quantize binary
        import sys
        if sys.platform == "win32":
            quantize_names = ["llama-quantize.exe", "quantize.exe"]
        else:
            quantize_names = ["llama-quantize", "quantize"]

        for name in quantize_names:
            quantize_path = llama_cpp_path / name
            if quantize_path.exists():
                status["quantize_binary"] = str(quantize_path)
                break

            # Also check in build directory
            build_path = llama_cpp_path / "build" / "bin" / name
            if build_path.exists():
                status["quantize_binary"] = str(build_path)
                break

        status["installed"] = status["convert_script"] is not None or status["quantize_binary"] is not None

        return status

    def _library_cleanup(self):
        """Clean up library: remove duplicates and missing files."""
        print_mini_banner("Library Cleanup")

        if not self.library:
            print_warning("Kirjasto ei ole käytettävissä")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        console.print("[cyan]Analysoidaan kirjastoa...[/cyan]\n")

        # Check for issues
        missing = self.library.find_missing_files()

        # Check for duplicates manually
        seen_paths = {}
        duplicates = []
        for model in self.library._models.values():
            try:
                abs_path = str(Path(model.path).absolute())
                if abs_path in seen_paths:
                    duplicates.append(model)
                else:
                    seen_paths[abs_path] = model
            except Exception:
                pass

        if not missing and not duplicates:
            console.print("[green]Kirjasto on kunnossa! Ei ongelmia.[/green]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Show issues
        console.print(Panel(
            f"[yellow]Puuttuvat tiedostot:[/yellow]  {len(missing)}\n"
            f"[yellow]Duplikaatit:[/yellow]          {len(duplicates)}",
            title="[bold]Löydetyt ongelmat[/bold]",
            border_style="yellow"
        ))
        console.print()

        if missing:
            console.print("[bold]Puuttuvat tiedostot:[/bold]")
            for m in missing[:5]:
                console.print(f"  [dim]• {m.name}[/dim]")
            if len(missing) > 5:
                console.print(f"  [dim]  ... ja {len(missing) - 5} muuta[/dim]")
            console.print()

        if duplicates:
            console.print("[bold]Duplikaatit:[/bold]")
            for m in duplicates[:5]:
                console.print(f"  [dim]• {m.name}[/dim]")
            if len(duplicates) > 5:
                console.print(f"  [dim]  ... ja {len(duplicates) - 5} muuta[/dim]")
            console.print()

        if questionary.confirm(
            "Siivoa ongelmat?",
            default=True,
            style=custom_style
        ).ask():
            results = self.library.cleanup_library() or {}
            print_success(f"Poistettu {results.get('missing', 0)} puuttuvaa ja {results.get('duplicates', 0)} duplikaattia")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _orphan_cleanup(self):
        """Find and clean up orphaned model files."""
        print_mini_banner("Orphan Cleanup")

        if not self.library:
            print_warning("Kirjasto ei ole kaytettavissa")
            console.print("[dim]Vihje: Kaynnista sovellus Model Hubin kautta[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        console.print("[cyan]Etsitaan orpo-tiedostoja...[/cyan]\n")

        orphan_stats = self.library.get_orphan_stats() or {}
        orphans = self.library.find_orphaned_files() or {}

        if orphan_stats.get('total_count', 0) == 0:
            console.print("[green]Ei orpo-tiedostoja! Kirjasto on siisti.[/green]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Show table of orphans by category
        table = Table(title="Loydetyt orvot", box=box.ROUNDED)
        table.add_column("Kategoria", style="cyan")
        table.add_column("Maara", justify="right")
        table.add_column("Koko", justify="right", style="yellow")

        for category, cat_stats in orphan_stats.get('by_category', {}).items():
            if cat_stats.get('count', 0) > 0:
                table.add_row(
                    category.upper(),
                    str(cat_stats.get('count', 0)),
                    format_size(cat_stats.get('size_bytes', 0))
                )

        table.add_row(
            "[bold]YHTEENSA[/bold]",
            f"[bold]{orphan_stats.get('total_count', 0)}[/bold]",
            f"[bold]{format_size(orphan_stats.get('total_size_bytes', 0))}[/bold]"
        )

        console.print(table)
        console.print()

        # Show detailed list if requested
        if questionary.confirm(
            "Nayta yksityiskohtainen lista?",
            default=True,
            style=custom_style
        ).ask():
            for category, files in orphans.items():
                if files:
                    console.print(f"\n[bold cyan]{category.upper()}:[/bold cyan]")
                    for f in files[:10]:
                        is_dir = f.get('is_directory', False)
                        icon = "[dim]folder[/dim]" if is_dir else "[dim]file[/dim]"
                        name = f.get('name', 'tuntematon')
                        size = f.get('size_bytes', 0)
                        console.print(f"  {icon}  {name} ({format_size(size)})")
                    if len(files) > 10:
                        console.print(f"  [dim]... ja {len(files) - 10} muuta[/dim]")

        console.print()

        # Cleanup options
        orphan_total_size = orphan_stats.get('total_size_bytes', 0)
        if questionary.confirm(
            f"Poista kaikki orvot? (vapautuu {format_size(orphan_total_size)})",
            default=False,
            style=custom_style
        ).ask():
            deleted = self.library.cleanup_orphans(dry_run=False) or []
            print_success(f"Poistettu {len(deleted)} orpo-tiedostoa")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _disk_analysis(self):
        """Show detailed disk usage analysis."""
        print_mini_banner("Disk Analysis")

        if not self.library:
            print_warning("Kirjasto ei ole kaytettavissa")
            console.print("[dim]Vihje: Kaynnista sovellus Model Hubin kautta[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        stats = self.library.get_stats() or {}
        category_stats = self.library.get_category_stats() or {}
        orphan_stats = self.library.get_orphan_stats() or {}

        console.print("[bold]Levytilan jakautuminen kategorioittain:[/bold]\n")

        total = stats.get('total_size_bytes', 0) + orphan_stats.get('total_size_bytes', 0)
        max_bar_width = 35

        categories = [
            ('base', '🏠 Base Models    ', 'green'),
            ('adapter', '🔧 LoRA Adapters   ', 'blue'),
            ('merged', '🔀 Merged Models  ', 'yellow'),
            ('ollama', '🤖 Ollama Models  ', 'cyan'),
        ]

        for cat_key, cat_name, color in categories:
            cat_stat = category_stats.get(cat_key, {'count': 0, 'size_bytes': 0})
            size = cat_stat['size_bytes']
            count = cat_stat['count']

            if total > 0:
                bar_width = int((size / total) * max_bar_width)
            else:
                bar_width = 0

            bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
            pct = (size / total * 100) if total > 0 else 0

            console.print(f"{cat_name}")
            console.print(f"  [{color}]{bar}[/{color}] {format_size(size):>10} ({count} kpl) {pct:5.1f}%")
            console.print()

        # Orphan space
        orphan_size = orphan_stats.get('total_size_bytes', 0)
        orphan_count = orphan_stats.get('total_count', 0)
        if orphan_size > 0:
            bar_width = int((orphan_size / total) * max_bar_width) if total > 0 else 0
            bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
            pct = (orphan_size / total * 100) if total > 0 else 0

            console.print("⚠️  Orvot tiedostot")
            console.print(f"  [red]{bar}[/red] {format_size(orphan_size):>10} ({orphan_count} kpl) {pct:5.1f}%")
            console.print()

        console.print(f"[bold]Yhteensa: {format_size(total)}[/bold]")
        console.print(f"[dim]Malleja kirjastossa: {stats.get('total_models', 0)}[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()
