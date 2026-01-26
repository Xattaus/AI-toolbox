"""
AI TOOLBOX - Benchmark Commands
===============================

CLI commands for model benchmarking, comparison, and profiling.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ..core.ui import (
    console,
    print_mini_banner,
    print_success,
    print_error,
    print_warning,
    print_info,
    format_size,
)
from ..core.paths import get_paths
from ..inference.benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    ComparisonReport,
)
from ..models.library import ModelLibrary

# Questionary style
custom_style = Style([
    ('qmark', 'fg:#ff9d00 bold'),
    ('question', 'fg:white bold'),
    ('answer', 'fg:#00d7ff bold'),
    ('pointer', 'fg:#ff9d00 bold'),
    ('highlighted', 'fg:#ff9d00 bold'),
    ('selected', 'fg:#00ff00'),
    ('separator', 'fg:#666666'),
    ('instruction', 'fg:#666666'),
])


class BenchmarkCommands:
    """CLI commands for model benchmarking."""

    def __init__(
        self,
        benchmark: BenchmarkRunner,
        library: ModelLibrary,
    ):
        """
        Initialize benchmark commands.

        Args:
            benchmark: BenchmarkRunner instance
            library: Model library instance
        """
        self.benchmark = benchmark
        self.library = library

    def benchmark_menu(self):
        """Benchmark Runner sub-menu."""
        while True:
            print_mini_banner("Benchmark Runner")

            # Show status
            status = self.benchmark.get_status()

            if not status["llama_cpp_available"]:
                console.print("[yellow]llama-cpp-python ei ole asennettu[/yellow]")
                console.print("[dim]Asenna: pip install llama-cpp-python[/dim]\n")
            else:
                console.print(f"[green]llama-cpp-python: Asennettu[/green]")

            console.print(f"[dim]Tuloksia: {status['results_count']}[/dim]")
            console.print(f"[dim]Output: {status['benchmarks_dir']}[/dim]\n")

            choices = [
                questionary.Choice(
                    title="Quick Benchmark       Nopea yhden mallin testi",
                    value="quick"
                ),
                questionary.Choice(
                    title="Compare Models        Vertaile useita malleja",
                    value="compare"
                ),
                questionary.Choice(
                    title="Throughput Test       Mittaa tokens/second eri prompteilla",
                    value="throughput"
                ),
                questionary.Choice(
                    title="Memory Profile        Mittaa mallin muistinkaytto",
                    value="memory"
                ),
                questionary.Separator(),
                questionary.Choice(
                    title="View Results          Nayta aiemmat tulokset",
                    value="view"
                ),
                questionary.Choice(
                    title="Export Results        Vie CSV/JSON",
                    value="export"
                ),
                questionary.Choice(
                    title="System Info           Nayta jarjestelmatiedot",
                    value="sysinfo"
                ),
                questionary.Separator(),
                questionary.Choice(title="Back                  Palaa", value="back"),
            ]

            choice = questionary.select(
                "Benchmark Runner:",
                choices=choices,
                style=custom_style,
                qmark=">>",
                pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "quick":
                self._quick_benchmark_wizard()
            elif choice == "compare":
                self._compare_models_wizard()
            elif choice == "throughput":
                self._throughput_test_wizard()
            elif choice == "memory":
                self._memory_profile_wizard()
            elif choice == "view":
                self._view_benchmark_results()
            elif choice == "export":
                self._export_benchmark_results()
            elif choice == "sysinfo":
                self._show_system_info()

    def _select_gguf_model(self, prompt: str = "Valitse GGUF-malli:") -> Optional[Path]:
        """Helper: Select GGUF model from library."""
        gguf_models = self.library.list_models(format_filter="gguf")

        if not gguf_models:
            print_warning("Ei GGUF-malleja kirjastossa.")
            console.print("[dim]Lisaa malleja Model Library -> Refresh Library tai GGUF Converter[/dim]")
            console.print(f"[dim]GGUF-kansio: {get_paths().gguf_dir}[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return None

        choices = []
        for model in gguf_models[:20]:  # Max 20
            size = format_size(model.size_bytes)
            quant = model.quantization or "-"
            title = f"{model.name[:35]:<35} {quant:<8} {size:>10}"
            choices.append(questionary.Choice(title=title, value=model.path))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Back", value="back"))

        result = questionary.select(
            prompt,
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result == "back":
            return None

        return Path(result)

    def _quick_benchmark_wizard(self):
        """Quick benchmark for a single model."""
        print_mini_banner("Quick Benchmark")

        if not self.benchmark.get_status()["llama_cpp_available"]:
            print_error("llama-cpp-python ei ole asennettu!")
            console.print("[dim]Asenna: pip install llama-cpp-python[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select model
        model_path = self._select_gguf_model("Valitse testattava malli:")
        if not model_path:
            return

        # Select prompt
        prompts = self.benchmark.get_default_prompts()
        prompt_choices = [
            questionary.Choice(title=f"{name:<12} {text[:50]}...", value=name)
            for name, text in prompts.items()
        ]
        prompt_choices.append(questionary.Choice(title="Oma promptti", value="custom"))

        prompt_name = questionary.select(
            "Valitse promptti:",
            choices=prompt_choices,
            style=custom_style,
        ).ask()

        if not prompt_name:
            return

        if prompt_name == "custom":
            prompt_text = questionary.text("Syota promptti:", style=custom_style).ask()
            if not prompt_text:
                return
            prompt_name = "custom"
        else:
            prompt_text = prompts[prompt_name]

        # Configuration
        max_tokens = questionary.text(
            "Max tokeneita:",
            default="128",
            style=custom_style,
        ).ask()

        num_runs = questionary.text(
            "Ajojen maara:",
            default="3",
            style=custom_style,
        ).ask()

        try:
            max_tokens = int(max_tokens)
            num_runs = int(num_runs)
        except ValueError:
            max_tokens = 128
            num_runs = 3

        config = BenchmarkConfig(
            prompt=prompt_text,
            prompt_name=prompt_name,
            max_tokens=max_tokens,
            num_runs=num_runs,
            warmup_runs=1,
        )

        # Run benchmark
        console.print("\n[cyan]Suoritetaan benchmark...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.benchmark.run_benchmark(str(model_path), config, progress_cb)

        if result:
            console.print()
            console.print(self.benchmark.format_result_table(result))

            # Save
            if questionary.confirm("Tallenna tulos?", default=True, style=custom_style).ask():
                self.benchmark.save_result(result)
                print_success("Tulos tallennettu!")

            # Show response
            if result.response_preview:
                console.print(Panel(
                    result.response_preview,
                    title="[bold]Mallin vastaus[/bold]",
                    border_style="dim"
                ))
        else:
            print_error("Benchmark epäonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _compare_models_wizard(self):
        """Compare multiple models."""
        print_mini_banner("Compare Models")

        if not self.benchmark.get_status()["llama_cpp_available"]:
            print_error("llama-cpp-python ei ole asennettu!")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        gguf_models = self.library.list_models(format_filter="gguf")

        if len(gguf_models) < 2:
            print_warning("Tarvitaan vahintaan 2 GGUF-mallia vertailuun.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        console.print("[cyan]Valitse vertailtavat mallit:[/cyan]")
        console.print("[dim]Valilyonti valitsee, Enter vahvistaa[/dim]\n")

        # Multi-select
        choices = []
        for model in gguf_models[:15]:
            size = format_size(model.size_bytes)
            quant = model.quantization or "-"
            title = f"{model.name[:35]:<35} {quant:<8} {size:>10}"
            choices.append(questionary.Choice(title=title, value=model.path))

        selected = questionary.checkbox(
            "Valitse mallit (2-5):",
            choices=choices,
            style=custom_style,
        ).ask()

        if not selected or len(selected) < 2:
            print_warning("Valitse vahintaan 2 mallia.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        if len(selected) > 5:
            selected = selected[:5]
            console.print("[yellow]Rajoitettu 5 malliin.[/yellow]")

        # Prompt
        prompts = self.benchmark.get_default_prompts()
        prompt_name = questionary.select(
            "Valitse promptti:",
            choices=[
                questionary.Choice(title=f"{name:<12} {text[:40]}...", value=name)
                for name, text in list(prompts.items())[:5]
            ],
            style=custom_style,
        ).ask()

        if not prompt_name:
            return

        config = BenchmarkConfig(
            prompt=prompts[prompt_name],
            prompt_name=prompt_name,
            max_tokens=128,
            num_runs=2,
            warmup_runs=1,
        )

        # Run
        console.print("\n[cyan]Vertaillaan malleja...[/cyan]")
        console.print(f"[dim]Tama voi kestaa hetken ({len(selected)} mallia)...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        report = self.benchmark.compare_models(selected, config, progress_cb)

        if report.results:
            console.print()
            console.print(self.benchmark.format_comparison_table(report))
            console.print(f"\n[bold green]Nopein: {report.fastest_model}[/bold green]")

            # Save results
            if questionary.confirm("Tallenna tulokset?", default=True, style=custom_style).ask():
                for result in report.results:
                    self.benchmark.save_result(result)
                print_success(f"Tallennettu {len(report.results)} tulosta!")
        else:
            print_error("Vertailu epäonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _throughput_test_wizard(self):
        """Throughput test with different prompts."""
        print_mini_banner("Throughput Test")

        if not self.benchmark.get_status()["llama_cpp_available"]:
            print_error("llama-cpp-python ei ole asennettu!")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select model
        model_path = self._select_gguf_model("Valitse testattava malli:")
        if not model_path:
            return

        # Select prompt sizes
        size_choices = [
            questionary.Choice(title="Short - Lyhyt kysymys", value="short", checked=True),
            questionary.Choice(title="Medium - Keskipitka", value="medium", checked=True),
            questionary.Choice(title="Long - Pitka selitys", value="long", checked=True),
            questionary.Choice(title="Code - Koodigenerointi", value="code"),
            questionary.Choice(title="Reasoning - Paattelyt", value="reasoning"),
        ]

        sizes = questionary.checkbox(
            "Valitse testattavat promptit:",
            choices=size_choices,
            style=custom_style,
        ).ask()

        if not sizes:
            return

        # Run
        console.print("\n[cyan]Suoritetaan throughput-testi...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        results = self.benchmark.run_throughput_test(str(model_path), sizes, progress_callback=progress_cb)

        if results:
            # Show results
            table = Table(title=f"Throughput: {model_path.name}", box=box.ROUNDED)
            table.add_column("Prompt", style="cyan")
            table.add_column("Tokens/s", style="green", justify="right")
            table.add_column("Time", style="yellow", justify="right")
            table.add_column("Tokens", style="white", justify="right")

            for r in results:
                table.add_row(
                    r.prompt_name,
                    f"{r.tokens_per_second:.1f}",
                    f"{r.total_time_ms:.0f} ms",
                    str(r.completion_tokens),
                )

            console.print(table)

            # Save
            if questionary.confirm("Tallenna tulokset?", default=True, style=custom_style).ask():
                for r in results:
                    self.benchmark.save_result(r)
                print_success(f"Tallennettu {len(results)} tulosta!")
        else:
            print_error("Testi epäonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _memory_profile_wizard(self):
        """Memory profiling."""
        print_mini_banner("Memory Profile")

        if not self.benchmark.get_status()["llama_cpp_available"]:
            print_error("llama-cpp-python ei ole asennettu!")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select model
        model_path = self._select_gguf_model("Valitse profiloitava malli:")
        if not model_path:
            return

        # Run
        console.print("\n[cyan]Mitataan muistinkayttoa...[/cyan]\n")

        result = self.benchmark.measure_memory_usage(str(model_path))

        if "error" in result:
            print_error(f"Profilointi epäonnistui: {result['error']}")
        else:
            console.print(Panel(
                f"[bold white]Malli:[/bold white] {result['model_name']}\n\n"
                f"[white]Muisti ennen:[/white] {result['memory_before_mb']:.0f} MB\n"
                f"[white]Muisti latauksen jalkeen:[/white] {result['memory_after_load_mb']:.0f} MB\n"
                f"[white]Muisti inferenssin jalkeen:[/white] {result['memory_after_inference_mb']:.0f} MB\n"
                f"[white]Muisti vapautuksen jalkeen:[/white] {result['memory_after_unload_mb']:.0f} MB\n\n"
                f"[bold green]Mallin muistinkaytto: {result['model_memory_mb']:.0f} MB[/bold green]\n"
                f"[dim]Inferenssin overhead: {result['inference_overhead_mb']:.0f} MB[/dim]",
                title="[bold]Memory Profile[/bold]",
                border_style="green"
            ))

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _view_benchmark_results(self):
        """View previous results."""
        print_mini_banner("Benchmark Results")

        results = self.benchmark.load_results()

        if not results:
            print_warning("Ei tallennettuja tuloksia.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Show table
        table = Table(title=f"Benchmark Results ({len(results)} kpl)", box=box.ROUNDED)
        table.add_column("Aika", style="dim")
        table.add_column("Malli", style="cyan")
        table.add_column("Tokens/s", style="green", justify="right")
        table.add_column("Time", style="yellow", justify="right")
        table.add_column("Memory", style="magenta", justify="right")
        table.add_column("Prompt", style="dim")

        # Show last 15
        for r in results[-15:]:
            timestamp = r.timestamp.split("T")[0] if r.timestamp else "-"
            table.add_row(
                timestamp,
                r.model_name[:25],
                f"{r.tokens_per_second:.1f}",
                f"{r.total_time_ms:.0f} ms",
                f"{r.memory_used_mb:.0f} MB",
                r.prompt_name,
            )

        console.print(table)

        # Actions
        action = questionary.select(
            "Toiminto:",
            choices=[
                questionary.Choice(title="Vie tulokset", value="export"),
                questionary.Choice(title="Tyhjenna kaikki", value="clear"),
                questionary.Choice(title="<-  Back", value="back"),
            ],
            style=custom_style,
        ).ask()

        if action == "export":
            self._export_benchmark_results()
        elif action == "clear":
            if questionary.confirm("Haluatko varmasti tyhjentaa kaikki tulokset?", default=False, style=custom_style).ask():
                if self.benchmark.clear_results():
                    print_success("Tulokset tyhjennetty!")
                else:
                    print_error("Tyhjennys epäonnistui")
                questionary.press_any_key_to_continue(style=custom_style).ask()

    def _export_benchmark_results(self):
        """Export results."""
        results = self.benchmark.load_results()

        if not results:
            print_warning("Ei tuloksia vietavaksi.")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select format
        fmt = questionary.select(
            "Vientimuoto:",
            choices=[
                questionary.Choice(title="CSV", value="csv"),
                questionary.Choice(title="JSON", value="json"),
            ],
            style=custom_style,
        ).ask()

        if not fmt:
            return

        # Filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"benchmark_results_{timestamp}.{fmt}"

        filename = questionary.text(
            "Tiedostonimi:",
            default=default_name,
            style=custom_style,
        ).ask()

        if not filename:
            return

        output_path = self.benchmark.exports_dir / filename

        if fmt == "csv":
            success = self.benchmark.export_csv(results, output_path)
        else:
            success = self.benchmark.export_json(results, output_path)

        if success:
            print_success(f"Viety: {output_path}")
        else:
            print_error("Vienti epäonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_system_info(self):
        """Show system information."""
        print_mini_banner("System Info")

        info = self.benchmark.get_system_info()

        gpu_info = ""
        if info.get('gpu_available'):
            gpu_info = (
                f"  Saatavilla: Kylla\n"
                f"  Nimi: {info.get('gpu_name', 'N/A')}\n"
                f"  Muisti: {info.get('gpu_memory_gb', 0):.1f} GB\n"
            )
        else:
            gpu_info = "  Saatavilla: Ei\n"

        console.print(Panel(
            f"[bold white]CPU[/bold white]\n"
            f"  Ytimet (looginen): {info['cpu_count']}\n"
            f"  Ytimet (fyysinen): {info['cpu_count_physical']}\n\n"
            f"[bold white]RAM[/bold white]\n"
            f"  Yhteensa: {info['total_ram_gb']:.1f} GB\n"
            f"  Kaytettavissa: {info['available_ram_gb']:.1f} GB\n"
            f"  Kaytossa: {info['used_ram_gb']:.1f} GB ({info['ram_percent']:.0f}%)\n\n"
            f"[bold white]GPU[/bold white]\n"
            f"{gpu_info}",
            title="[bold]System Information[/bold]",
            border_style="cyan"
        ))

        questionary.press_any_key_to_continue(style=custom_style).ask()
