"""
AI TOOLBOX - GGUF Tools Commands
================================

Unified CLI for GGUF conversion, quantization, and VRAM calculation.
Combines the functionality of converter_cmd.py.
"""

import os
import sys
import re
import subprocess
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
    menu_separator,
    create_model_preview_card,
    MENU_STYLE,
)
from ..core.paths import get_paths, get_gguf_dir
from ..models.library import ModelLibrary
from ..conversion.converter import GGUFConverter
from .selection import build_conversion_choices

# Use unified menu style
custom_style = MENU_STYLE


class GGUFToolsCommands:
    """Unified CLI commands for GGUF conversion and quantization."""

    def __init__(self, library: ModelLibrary, converter: GGUFConverter, downloader=None):
        """
        Initialize GGUF Tools commands.

        Args:
            library: Model library instance
            converter: GGUF converter instance
            downloader: Optional model downloader instance
        """
        self.library = library
        self.converter = converter
        self.downloader = downloader

    def gguf_tools_menu(self):
        """GGUF Tools main menu."""
        while True:
            print_branded_header("GGUF Tools", "Muunna, kvantisoi ja hallitse GGUF-malleja")

            # Show GGUF stats
            gguf_dir = get_gguf_dir()
            gguf_files = list(gguf_dir.glob("*.gguf"))
            total_size = sum(f.stat().st_size for f in gguf_files)
            console.print(
                f"[dim]GGUF-malleja: {len(gguf_files)} ({format_size(total_size)})[/dim]\n"
            )

            choices = [
                menu_separator("Konvertointi"),
                questionary.Choice(
                    title=format_menu_item("HuggingFace -> GGUF", "Lataa ja muunna"), value="hf"
                ),
                questionary.Choice(
                    title=format_menu_item("Paikallinen -> GGUF", "Muunna levyltä"), value="local"
                ),
                questionary.Choice(
                    title=format_menu_item("Kirjastosta -> GGUF", "Muunna kirjastosta"),
                    value="library",
                ),
                menu_separator("Kvantisointi"),
                questionary.Choice(
                    title=format_menu_item("Kvantisoi GGUF", "Pienennä GGUF-mallia"),
                    value="quantize",
                ),
                questionary.Choice(
                    title=format_menu_item("Konvertoi & Kvantisoi", "Yhdistetty prosessi"),
                    value="convert_quantize",
                ),
                menu_separator("Työkalut"),
                questionary.Choice(
                    title=format_menu_item("VRAM-laskuri", "Laske muistivaatimukset"), value="vram"
                ),
                questionary.Choice(
                    title=format_menu_item("Kvantisointityypit", "Vertaile Q-tasoja"),
                    value="quants",
                ),
                menu_separator(),
                questionary.Choice(title=format_menu_item("<- Palaa", ""), value="back"),
            ]

            choice = questionary.select(
                "Valitse toiminto:", choices=choices, style=custom_style, qmark="#", pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "hf":
                self._convert_from_huggingface()
            elif choice == "local":
                self._convert_local_model()
            elif choice == "library":
                self._convert_from_library()
            elif choice == "quantize":
                self._quantize_tool()
            elif choice == "convert_quantize":
                self._convert_and_quantize()
            elif choice == "vram":
                self._vram_calculator()
            elif choice == "quants":
                self._show_quantization_types()

    # ==================== CONVERSION ====================

    def _convert_from_huggingface(self):
        """Download model from HuggingFace and convert to GGUF."""
        print_mini_banner("HuggingFace -> GGUF")

        model_id = questionary.text(
            "HuggingFace Model ID (esim. Qwen/Qwen2.5-0.5B):",
            style=custom_style,
        ).ask()

        if not model_id:
            return

        if not self.downloader:
            print_error("Downloader ei ole kaytettavissa")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Check if already downloaded
        existing = self.downloader.check_exists(model_id)
        if existing:
            console.print(f"[green]Malli loytyi jo:[/green] {existing}")
            if not questionary.confirm(
                "Kayta olemassa olevaa?", default=True, style=custom_style
            ).ask():
                return
            model_path = existing
        else:
            # Download first
            console.print(f"\n[cyan]Ladataan mallia: {model_id}...[/cyan]\n")
            model_path = self.downloader.download_model(
                model_id,
                include_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"],
                exclude_patterns=["*.bin", "*.md"],
            )
            if not model_path:
                print_error("Lataus epäonnistui")
                questionary.press_any_key_to_continue(style=custom_style).ask()
                return

            # Add to library
            try:
                entry = self.library.add_model(
                    str(model_path), source="huggingface", source_id=model_id
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")

        # Select quantization and convert
        self._run_conversion(str(model_path), model_id.replace("/", "_"))

    def _convert_local_model(self):
        """Convert local model to GGUF."""
        print_mini_banner("Local Model -> GGUF")

        model_path = self._select_model_for_conversion("Valitse muunnettava malli:")
        if not model_path:
            return

        if not (model_path / "config.json").exists():
            print_error("Kansiosta ei loydy config.json - ei ole HuggingFace-malli")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        self._run_conversion(str(model_path), model_path.name)

    def _convert_from_library(self):
        """Convert model from library to GGUF using table-based selection."""
        print_mini_banner("Library -> GGUF Konvertointi")

        # Get convertible models
        convertible = self.library.get_convertible_models()
        merged = self.library.get_all_merged()
        merged_convertible = [m for m in merged if m.format in ["safetensors", "pytorch"]]

        if not convertible and not merged_convertible:
            print_warning("Kirjastossa ei ole muunnettavia malleja (SafeTensors/PyTorch)")
            console.print("[dim]Lataa ensin malli Model Hub -valikosta[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Count total
        total = len(convertible) + len(merged_convertible)

        # Build unified list for table selection
        all_models = []
        idx = 1

        # Create table
        table = Table(
            title=f"[bold orange1]Muunnettavat mallit ({total})[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("Malli", style="white", min_width=35)
        table.add_column("Tyyppi", style="green", width=12, justify="center")
        table.add_column("Koko", style="cyan", width=10, justify="right")

        # Add convertible models (SafeTensors)
        if convertible:
            table.add_row("", "[bold green]--- 🏠 SafeTensors ---[/bold green]", "", "")
            for m in convertible[:12]:
                size = format_size(m.size_bytes) if m.size_bytes else "-"
                name = m.name[:35] if len(m.name) <= 35 else m.name[:32] + "..."
                table.add_row(str(idx), f"🏠 {name}", "SafeTensors", size)
                all_models.append(m)
                idx += 1

        # Add merged models
        if merged_convertible:
            table.add_row("", "[bold magenta]--- 🔀 Merged ---[/bold magenta]", "", "")
            for m in merged_convertible[:8]:
                size = format_size(m.size_bytes) if m.size_bytes else "-"
                name = m.name[:35] if len(m.name) <= 35 else m.name[:32] + "..."
                table.add_row(str(idx), f"🔀 {name}", "Merged", size)
                all_models.append(m)
                idx += 1

        console.print(table)
        console.print()

        # Get selection by number
        while True:
            answer = questionary.text(
                f"Valitse numero [1-{len(all_models)}] (0 = peruuta)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                return

            try:
                sel_idx = int(answer.strip())
                if 1 <= sel_idx <= len(all_models):
                    selected = all_models[sel_idx - 1]
                    break
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(all_models)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

        # Show preview card before conversion
        console.print()
        console.print(create_model_preview_card(selected, show_path=True))
        console.print()

        if not questionary.confirm(
            "Muunna tämä malli GGUF-muotoon?", default=True, style=custom_style
        ).ask():
            return

        self._run_conversion(selected.path, selected.name)

    def _convert_and_quantize(self):
        """Combined convert and quantize."""
        print_mini_banner("Konvertoi & Kvantisoi")

        console.print("[cyan]Muunna HuggingFace-malli suoraan kvantisoituun GGUF:iin.[/cyan]\n")

        # Select model
        model_path = self._select_model_for_conversion("Valitse muunnettava malli:")
        if not model_path:
            return

        self._run_conversion(str(model_path), model_path.name)

    def _run_conversion(self, model_path: str, model_name: str):
        """Run GGUF conversion with selected quantization."""

        # Select quantization
        quant_choices = [
            questionary.Choice(title="Q4_K_M  - Paras tasapaino (suositeltu)", value="q4_k_m"),
            questionary.Choice(title="Q5_K_M  - Korkeampi laatu", value="q5_k_m"),
            questionary.Choice(title="Q8_0    - Lahes häviötön", value="q8_0"),
            questionary.Choice(title="Q6_K    - Korkea laatu", value="q6_k"),
            questionary.Choice(title="Q3_K_M  - Pienempi koko", value="q3_k_m"),
            questionary.Choice(title="Q2_K    - Erittain pieni", value="q2_k"),
            questionary.Separator(),
            questionary.Choice(title="F16     - Ei kvantisointia (iso)", value="f16"),
            questionary.Choice(title="Back", value=None),
        ]

        quantization = questionary.select(
            "Valitse quantization:",
            choices=quant_choices,
            style=custom_style,
        ).ask()

        if not quantization:
            return

        # Show estimate
        console.print("\n[dim]Lasketaan mallin kokoa...[/dim]")
        estimates = self.converter.estimate_model_size(Path(model_path), quantization)
        if estimates.get("estimated_size_gb"):
            console.print(
                Panel(
                    f"[white]Malli:[/white] {model_name}\n"
                    f"[white]Quantization:[/white] {quantization.upper()}\n"
                    f"[white]Parametreja:[/white] {estimates.get('total_params_billions', '?')}B\n"
                    f"[white]Arvioitu koko:[/white] {estimates.get('estimated_size_gb', '?')} GB\n"
                    f"[white]Pakkaussuhde:[/white] {estimates.get('compression_ratio', '?')}x",
                    title="[bold]Conversion Preview[/bold]",
                    border_style="yellow",
                )
            )

        if not questionary.confirm("Aloita muunnos?", default=True, style=custom_style).ask():
            return

        # Check llama.cpp
        console.print("\n[bold cyan][ 1/4 ] Tarkistetaan llama.cpp...[/bold cyan]")
        status = self.converter.check_llama_cpp()

        if not status["installed"]:
            console.print("[yellow]llama.cpp ei ole asennettu. Asennetaan...[/yellow]")
            console.print("[dim]Tama voi kestaa muutaman minuutin...[/dim]\n")

            if not self.converter.setup_llama_cpp():
                print_error("llama.cpp asennus epäonnistui")
                questionary.press_any_key_to_continue(style=custom_style).ask()
                return

            console.print("[green]llama.cpp asennettu![/green]\n")
        else:
            console.print(f"[green]OK[/green] - {status.get('path', 'loytyi')}\n")

        # Run conversion
        console.print(f"[bold cyan][ 2/4 ] Aloitetaan GGUF-muunnos...[/bold cyan]")
        console.print(f"[dim]Lahde: {model_path}[/dim]")
        console.print(f"[dim]Kohde: {self.converter.output_dir}[/dim]\n")

        if quantization == "f16":
            console.print("[bold]Muunnetaan F16-muotoon (ei kvantisointia)...[/bold]\n")
            result = self.converter.convert_to_gguf(
                model_path=model_path,
                output_type="f16",
            )
        else:
            console.print(f"[bold]Vaihe 1: HuggingFace -> GGUF (F16)[/bold]")
            console.print(f"[bold]Vaihe 2: F16 -> {quantization.upper()}[/bold]\n")
            result = self.converter.convert_and_quantize(
                model_path=model_path,
                quantization=quantization,
                keep_f16=False,
            )

        if result.get("success"):
            output_path = result.get("output_path")
            file_size = result.get("file_size_gb", 0)

            console.print(
                Panel(
                    f"[green]Muunnos valmis![/green]\n\n"
                    f"[white]Tiedosto:[/white] {output_path}\n"
                    f"[white]Koko:[/white] {file_size:.2f} GB\n"
                    f"[white]Quantization:[/white] {quantization.upper()}",
                    title="[bold green]Success[/bold green]",
                    border_style="green",
                )
            )

            # Add automatically to library
            try:
                entry = self.library.add_model(
                    path=output_path,
                    source="converted",
                    source_id=model_name,
                )
                print_success(f"Lisatty automaattisesti Model Libraryyn: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            error = result.get("error", "Tuntematon virhe")
            console.print(
                Panel(
                    f"[red]Muunnos epäonnistui[/red]\n\n{error}",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_model_for_conversion(self, prompt: str) -> Optional[Path]:
        """Select a convertible model from library + downloads (table, numbered).

        Shows the library's convertible (SafeTensors/PyTorch) models and any
        downloaded HF models. No manual path entry - add models to the library
        or downloads first.
        """
        # Convertible library models (SafeTensors/PyTorch), incl. merged models
        convertible = list(self.library.get_convertible_models())
        merged = self.library.get_all_merged()
        seen = {str(m.path) for m in convertible}
        for m in merged:
            if m.format in ("safetensors", "pytorch") and str(m.path) not in seen:
                convertible.append(m)
                seen.add(str(m.path))

        downloaded = self.downloader.list_downloaded() if self.downloader else []

        items = build_conversion_choices(convertible, downloaded)

        if not items:
            print_warning(
                "Kirjastossa ei ole muunnettavia malleja (SafeTensors/PyTorch) eika latauksia"
            )
            console.print("[dim]Lataa ensin malli Model Hub -valikosta[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return None

        print_branded_header("GGUF Konvertointi", prompt)
        table = Table(
            title="[bold orange1]Muunnettavat mallit[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("Malli", style="white", min_width=35)
        table.add_column("Koko", style="cyan", width=10, justify="right")
        table.add_column("Lahde", style="dim", width=10)

        last_source = None
        for i, it in enumerate(items, 1):
            if it["source"] != last_source:
                header = (
                    "[bold green]--- Kirjasto ---[/bold green]"
                    if it["source"] == "library"
                    else "[bold cyan]--- HuggingFace ---[/bold cyan]"
                )
                table.add_row("", header, "", "")
                last_source = it["source"]
            size = format_size(it["size_bytes"]) if it["size_bytes"] else "-"
            name = it["name"][:35] if len(it["name"]) <= 35 else it["name"][:32] + "..."
            icon = "🏠" if it["source"] == "library" else "🤗"
            tag = "Local" if it["source"] == "library" else "HF"
            table.add_row(str(i), f"{icon} {name}", size, tag)

        console.print(table)
        console.print()

        # Get selection by number
        while True:
            answer = questionary.text(
                f"Valitse numero [1-{len(items)}] (0 = peruuta)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                return None

            try:
                sel_idx = int(answer.strip())
                if 1 <= sel_idx <= len(items):
                    return items[sel_idx - 1]["path"]
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(items)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

    # ==================== QUANTIZATION ====================

    def _quantize_tool(self):
        """Quantize GGUF models using table-based selection."""
        print_branded_header("Quantize Tool", "Kvantisoi GGUF-malleja")

        # Get GGUF models from library using new methods
        quantizable = self.library.get_quantizable_models()  # F16/F32 - recommended
        already_quantized = self.library.get_already_quantized_models()  # Already quantized

        # Also scan GGUF directory for files not in library
        gguf_dir = get_gguf_dir()
        library_paths = {Path(m.path).resolve() for m in quantizable + already_quantized}

        # Find orphan GGUF files not in library
        orphan_files = []
        for f in gguf_dir.glob("*.gguf"):
            if f.resolve() not in library_paths:
                orphan_files.append(f)

        if not quantizable and not already_quantized and not orphan_files:
            print_warning("GGUF-malleja ei löytynyt.")
            console.print(f"[dim]Kansio: {gguf_dir}[/dim]")
            console.print("[dim]Muunna ensin malli GGUF-muotoon.[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Build unified list with index for table selection
        all_models = []  # List of (model_or_path, category, index)
        idx = 1

        # Create table
        table = Table(
            title=f"[bold orange1]GGUF-mallit kvantisoitavaksi[/bold orange1]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="orange3",
            padding=(0, 1),
        )
        table.add_column("#", style="bold yellow", width=4, justify="right")
        table.add_column("Malli", style="white", min_width=30)
        table.add_column("Quant", style="yellow", width=8, justify="center")
        table.add_column("Koko", style="cyan", width=10, justify="right")
        table.add_column("Tyyppi", style="dim", width=12)

        # Add F16/F32 models (recommended for quantization)
        if quantizable:
            table.add_row("", "[bold green]--- Suositeltu (F16/F32) ---[/bold green]", "", "", "")
            for m in quantizable[:10]:
                size = format_size(m.size_bytes) if m.size_bytes else "-"
                quant = m.quantization or "F16"
                name = m.name[:35] if len(m.name) <= 35 else m.name[:32] + "..."
                table.add_row(str(idx), f"📦 {name}", quant, size, "Kirjasto")
                all_models.append((m, "quantizable"))
                idx += 1

        # Add already quantized models
        if already_quantized:
            table.add_row("", "[bold yellow]--- Jo kvantisoidut ---[/bold yellow]", "", "", "")
            for m in already_quantized[:10]:
                size = format_size(m.size_bytes) if m.size_bytes else "-"
                quant = m.quantization or "-"
                name = m.name[:35] if len(m.name) <= 35 else m.name[:32] + "..."
                table.add_row(str(idx), f"⚡ {name}", quant, size, "Kirjasto")
                all_models.append((m, "quantized"))
                idx += 1

        # Add orphan files
        if orphan_files:
            table.add_row("", "[bold dim]--- Ei kirjastossa ---[/bold dim]", "", "", "")
            for f in sorted(orphan_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                size = format_size(f.stat().st_size)
                name = f.stem[:35] if len(f.stem) <= 35 else f.stem[:32] + "..."
                table.add_row(str(idx), f"📄 {name}", "-", size, "Tiedosto")
                all_models.append((f, "orphan"))
                idx += 1

        console.print(table)
        console.print()

        # Get selection by number
        while True:
            answer = questionary.text(
                f"Valitse numero [1-{len(all_models)}] (0 = peruuta)",
                style=custom_style,
            ).ask()

            if answer is None or answer.strip() in ("", "0", "q"):
                return

            try:
                sel_idx = int(answer.strip())
                if 1 <= sel_idx <= len(all_models):
                    selected_item, category = all_models[sel_idx - 1]
                    break
                else:
                    print_warning(f"Valitse numero väliltä 1-{len(all_models)}")
            except ValueError:
                print_warning("Anna kelvollinen numero")

        # Convert to Path
        if hasattr(selected_item, "path"):
            # It's a ModelEntry - show preview
            console.print()
            console.print(create_model_preview_card(selected_item, show_path=True))
            console.print()
            selected = Path(selected_item.path)
        else:
            # It's already a Path (orphan file)
            console.print(f"\n[dim]Tiedosto: {selected_item}[/dim]\n")
            selected = selected_item

        # Select target quantization
        quant_choices = [
            questionary.Choice(title="Q4_K_M  - Paras tasapaino (suositeltu)", value="Q4_K_M"),
            questionary.Choice(title="Q5_K_M  - Korkeampi laatu", value="Q5_K_M"),
            questionary.Choice(title="Q6_K    - Korkea laatu", value="Q6_K"),
            questionary.Choice(title="Q8_0    - Lahes häviötön", value="Q8_0"),
            questionary.Separator(),
            questionary.Choice(title="Q4_K_S  - 4-bit small", value="Q4_K_S"),
            questionary.Choice(title="Q3_K_M  - 3-bit medium", value="Q3_K_M"),
            questionary.Choice(title="Q3_K_S  - 3-bit small", value="Q3_K_S"),
            questionary.Choice(title="Q2_K    - 2-bit (pienin)", value="Q2_K"),
            questionary.Separator(),
            questionary.Choice(title="IQ4_XS  - 4-bit importance-weighted", value="IQ4_XS"),
            questionary.Choice(title="IQ3_M   - 3-bit importance-weighted", value="IQ3_M"),
            questionary.Choice(title="Back", value=None),
        ]

        target_quant = questionary.select(
            "Valitse kohde-kvantisointi:",
            choices=quant_choices,
            style=custom_style,
        ).ask()

        if not target_quant:
            return

        # Check llama-quantize
        quantize_exe = self.converter._llama_cpp.find_quantize_binary()

        if not quantize_exe:
            print_error("llama-quantize ei loydy!")
            console.print(f"[dim]Odotettu sijainti: {self.converter.llama_cpp_path}[/dim]")

            if questionary.confirm(
                "Haluatko ladata llama.cpp binaarit automaattisesti?",
                default=True,
                style=custom_style,
            ).ask():
                cuda = questionary.confirm(
                    "Lataa CUDA-versio (GPU-kiihdytys)?", default=True, style=custom_style
                ).ask()

                if self.converter.download_llama_cpp_binaries(cuda=cuda):
                    quantize_exe = self.converter._llama_cpp.find_quantize_binary()
                    if not quantize_exe:
                        print_error("Lataus onnistui, mutta llama-quantize ei loydy.")
                        questionary.press_any_key_to_continue(style=custom_style).ask()
                        return
                    console.print(f"[green]llama-quantize loytyi: {quantize_exe}[/green]")
                else:
                    console.print("\n[yellow]Lataa manuaalisesti:[/yellow]")
                    console.print("[cyan]https://github.com/ggml-org/llama.cpp/releases[/cyan]")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                    return
            else:
                console.print("\n[yellow]Lataa manuaalisesti:[/yellow]")
                console.print("[cyan]https://github.com/ggml-org/llama.cpp/releases[/cyan]")
                questionary.press_any_key_to_continue(style=custom_style).ask()
                return

        # Helper to detect current quantization from filename
        def detect_quant(filename: str) -> str:
            name_lower = filename.lower()
            quants = [
                "q8_0",
                "q6_k",
                "q5_k_m",
                "q5_k_s",
                "q4_k_m",
                "q4_k_s",
                "q4_0",
                "q4_1",
                "q3_k_l",
                "q3_k_m",
                "q3_k_s",
                "q2_k",
                "iq4_xs",
                "iq3_m",
                "iq3_s",
                "iq2_xs",
                "f16",
                "f32",
                "bf16",
            ]
            for q in quants:
                if q in name_lower or q.replace("_", "-") in name_lower:
                    return q.upper()
            return "F16/F32"

        # Define filenames
        input_path = selected
        stem = input_path.stem
        # Remove old quantization from name if present
        for q in [
            "f16",
            "f32",
            "bf16",
            "q8_0",
            "q6_k",
            "q5_k_m",
            "q5_k_s",
            "q4_k_m",
            "q4_k_s",
            "q4_0",
            "q4_1",
            "q3_k_l",
            "q3_k_m",
            "q3_k_s",
            "q2_k",
            "iq4_xs",
            "iq3_m",
            "iq2_xs",
        ]:
            stem = re.sub(rf"[-_]{q}$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(rf"[-_]{q}[-_]", "-", stem, flags=re.IGNORECASE)
        output_path = input_path.parent / f"{stem}-{target_quant.lower()}.gguf"

        # Show summary
        input_size = format_size(input_path.stat().st_size)
        current_quant = detect_quant(input_path.name)
        console.print(
            Panel(
                f"[white]Lahde:[/white]      {input_path.name}\n"
                f"[white]Koko:[/white]       {input_size}\n"
                f"[white]Nykyinen:[/white]   {current_quant}\n"
                f"[white]Kohde:[/white]      {target_quant}\n"
                f"[white]Tuloste:[/white]    {output_path.name}",
                title="[bold]Kvantisoinnin yhteenveto[/bold]",
                border_style="yellow",
            )
        )

        if not questionary.confirm("Aloita kvantisointi?", default=True, style=custom_style).ask():
            return

        # Run quantization
        console.print(f"\n[bold cyan]Kvantisoidaan {target_quant}...[/bold cyan]")
        console.print("[dim]Tama voi kestaa muutaman minuutin...[/dim]\n")

        try:
            result = subprocess.run(
                [str(quantize_exe), str(input_path), str(output_path), target_quant],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=7200,  # 2 tunnin timeout - suuret mallit voivat kestää kauan
            )

            if output_path.exists() and output_path.stat().st_size > 0:
                output_size = format_size(output_path.stat().st_size)
                console.print(
                    Panel(
                        f"[green]Kvantisointi valmis![/green]\n\n"
                        f"[white]Tiedosto:[/white]  {output_path.name}\n"
                        f"[white]Koko:[/white]      {output_size}\n"
                        f"[white]Sijainti:[/white]  {output_path}",
                        title="[bold green]Success[/bold green]",
                        border_style="green",
                    )
                )

                # Add to library
                try:
                    entry = self.library.add_model(
                        path=str(output_path),
                        source="quantized",
                        source_id=input_path.stem,
                    )
                    print_success(f"Lisatty kirjastoon: {entry.name}")
                except Exception as e:
                    print_warning(f"Kirjastolisays epäonnistui: {e}")
            else:
                error_msg = result.stderr or result.stdout or "Tuntematon virhe"
                print_error(f"Kvantisointi epäonnistui:\n{error_msg[:500]}")

        except Exception as e:
            print_error(f"Virhe kvantisoinnissa: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    # ==================== VRAM CALCULATOR ====================

    def _vram_calculator(self):
        """VRAM calculator tool."""
        print_mini_banner("VRAM-laskuri")

        params = questionary.text(
            "Mallin parametrit (miljardeina, esim. 7, 13, 70):", style=custom_style, default="7"
        ).ask()

        if not params:
            return

        try:
            params_b = float(params)
        except ValueError:
            print_error("Virheellinen numero")
            return

        calculations = [
            ("F16", 16, "Taysi tarkkuus"),
            ("Q8_0", 8, "8-bittinen kvantisointi"),
            ("Q6_K", 6.5, "6-bittinen K-quant"),
            ("Q5_K_M", 5.5, "5-bittinen K-quant"),
            ("Q4_K_M", 4.5, "4-bittinen K-quant (suositeltu)"),
            ("Q4_0", 4.0, "4-bittinen perus"),
            ("Q3_K_M", 3.5, "3-bittinen K-quant"),
            ("Q2_K", 2.5, "2-bittinen K-quant"),
        ]

        console.print(f"\n[bold]VRAM-vaatimukset {params_b}B parametrin mallille[/bold]\n")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Kvantisointi", style="white")
        table.add_column("Mallin koko", justify="right", style="yellow")
        table.add_column("+ Konteksti (4K)", justify="right", style="cyan")
        table.add_column("+ Konteksti (8K)", justify="right", style="cyan")
        table.add_column("Kuvaus", style="dim")

        for name, bits, desc in calculations:
            model_gb = (params_b * 1e9 * bits / 8) / (1024**3)
            ctx_4k = model_gb + 0.5
            ctx_8k = model_gb + 1.0

            table.add_row(name, f"{model_gb:.1f} GB", f"{ctx_4k:.1f} GB", f"{ctx_8k:.1f} GB", desc)

        console.print(table)
        console.print(
            "\n[dim]Huom: Todellinen VRAM-kaytto voi vaihdella toteutuksen ja kontekstipituuden mukaan.[/dim]"
        )
        questionary.press_any_key_to_continue(style=custom_style).ask()

    # ==================== QUANTIZATION TYPES ====================

    def _show_quantization_types(self):
        """Show available quantization types."""
        print_mini_banner("Kvantisointityypit")

        table = Table(
            title="Saatavilla olevat kvantisointityypit",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Tyyppi", style="white", width=10)
        table.add_column("Bitit", justify="right", style="yellow", width=6)
        table.add_column("Laatu", style="green", width=12)
        table.add_column("Kuvaus", style="dim")

        quants = [
            ("F16", "16.0", "Korkein", "Taysi 16-bittinen tarkkuus"),
            ("Q8_0", "8.0", "Erittain korkea", "8-bittinen kvantisointi"),
            ("Q6_K", "6.5", "Korkea", "6-bittinen K-quant"),
            ("Q5_K_M", "5.5", "Korkea", "5-bittinen K-quant medium"),
            ("Q5_K_S", "5.5", "Korkea", "5-bittinen K-quant small"),
            ("Q4_K_M", "4.5", "Keskikorkea", "4-bittinen K-quant medium (suositeltu)"),
            ("Q4_K_S", "4.5", "Keskitaso", "4-bittinen K-quant small"),
            ("Q4_0", "4.0", "Keskitaso", "4-bittinen peruskvantisointi"),
            ("Q3_K_M", "3.5", "Matala", "3-bittinen K-quant medium"),
            ("Q3_K_S", "3.5", "Matala", "3-bittinen K-quant small"),
            ("Q2_K", "2.5", "Erittain matala", "2-bittinen K-quant (äärimmäinen)"),
            ("IQ4_XS", "4.0", "Keskitaso", "4-bittinen tärkeyspainotettu"),
            ("IQ3_M", "3.4", "Matala", "3-bittinen tärkeyspainotettu"),
            ("IQ2_XS", "2.3", "Erittain matala", "2-bittinen tärkeyspainotettu"),
        ]

        for name, bits, quality, desc in quants:
            table.add_row(name, bits, quality, desc)

        console.print(table)
        questionary.press_any_key_to_continue(style=custom_style).ask()
