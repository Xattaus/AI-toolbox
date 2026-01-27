"""
AI TOOLBOX - Dataset Commands
=============================

CLI commands for dataset preparation and management.
"""

from pathlib import Path
from typing import Optional, Callable, List

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
from ..training.dataset import (
    DatasetPrep,
    DatasetFormat,
    DatasetStats,
    SplitConfig,
    FilterConfig,
    CleaningOperation,
)

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


class DatasetCommands:
    """CLI commands for dataset preparation."""

    def __init__(self, dataset_prep: DatasetPrep):
        """
        Initialize dataset commands.

        Args:
            dataset_prep: DatasetPrep instance
        """
        self.dataset_prep = dataset_prep

    def dataset_prep_menu(self):
        """Dataset Prep sub-menu."""
        while True:
            print_mini_banner("Dataset Prep")

            # Show status
            status = self.dataset_prep.get_status()
            console.print(f"[dim]Datasetteja: {status['datasets_count']}[/dim]")
            console.print(f"[dim]Output: {status['output_dir']}[/dim]\n")

            choices = [
                questionary.Choice(
                    title="Inspect Dataset       Analysoi datasetin rakenne ja tilastot",
                    value="inspect"
                ),
                questionary.Choice(
                    title="Convert Format        Muunna formaatista toiseen",
                    value="convert"
                ),
                questionary.Choice(
                    title="Split Dataset         Jaa train/test/validation osiin",
                    value="split"
                ),
                questionary.Choice(
                    title="Clean Dataset         Siivoa ja normalisoi",
                    value="clean"
                ),
                questionary.Choice(
                    title="Deduplicate           Poista duplikaatit",
                    value="dedupe"
                ),
                questionary.Choice(
                    title="Filter by Length      Suodata pituuden mukaan",
                    value="filter"
                ),
                questionary.Choice(
                    title="Token Counter         Laske tokenimaarat",
                    value="tokens"
                ),
                questionary.Choice(
                    title="Merge Datasets        Yhdista useita datasetteja",
                    value="merge"
                ),
                questionary.Separator(),
                questionary.Choice(
                    title="Browse Datasets       Selaa ja nayta datasetit",
                    value="browse"
                ),
                questionary.Separator(),
                questionary.Choice(title="Back                  Palaa", value="back"),
            ]

            choice = questionary.select(
                "Dataset Prep:",
                choices=choices,
                style=custom_style,
                qmark=">>",
                pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "inspect":
                self._inspect_dataset_wizard()
            elif choice == "convert":
                self._convert_format_wizard()
            elif choice == "split":
                self._split_dataset_wizard()
            elif choice == "clean":
                self._clean_dataset_wizard()
            elif choice == "dedupe":
                self._dedupe_dataset_wizard()
            elif choice == "filter":
                self._filter_dataset_wizard()
            elif choice == "tokens":
                self._count_tokens_wizard()
            elif choice == "merge":
                self._merge_datasets_wizard()
            elif choice == "browse":
                self._browse_datasets()

    def _select_dataset(self, prompt: str = "Valitse dataset:") -> Optional[Path]:
        """Helper: Select dataset from file list."""
        datasets = self.dataset_prep.list_datasets(include_processed=True)

        if not datasets:
            print_warning("Ei datasetteja. Lisaa .jsonl/.json/.csv tiedostoja datasets/-kansioon.")
            console.print(f"\n[dim]Datasets-kansio: {self.dataset_prep.datasets_dir}[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return None

        choices = []
        for ds in datasets:
            size = format_size(ds["size_bytes"])
            processed_tag = " [processed]" if ds.get("is_processed") else ""
            title = f"{ds['name']:<35} {ds['format']:<10} {size:>10}{processed_tag}"
            choices.append(questionary.Choice(title=title, value=ds["path"]))

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

        return result

    def _browse_datasets(self):
        """Browse datasets."""
        print_mini_banner("Browse Datasets")

        datasets = self.dataset_prep.list_datasets(include_processed=True)

        if not datasets:
            print_warning("Ei datasetteja loytynyt.")
            console.print(f"\n[dim]Lisaa .jsonl, .json tai .csv tiedostoja kansioon:[/dim]")
            console.print(f"[cyan]{self.dataset_prep.datasets_dir}[/cyan]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Create table
        table = Table(title="Datasetit", box=box.ROUNDED)
        table.add_column("Nimi", style="cyan")
        table.add_column("Formaatti", style="yellow")
        table.add_column("Koko", justify="right")
        table.add_column("Tyyppi", style="dim")

        for ds in datasets:
            size = format_size(ds["size_bytes"])
            dtype = "processed" if ds.get("is_processed") else "original"
            table.add_row(ds["name"], ds["format"], size, dtype)

        console.print(table)
        console.print(f"\n[dim]Datasets-kansio: {self.dataset_prep.datasets_dir}[/dim]")
        console.print(f"[dim]Processed-kansio: {self.dataset_prep.output_dir}[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _inspect_dataset_wizard(self):
        """Analyze dataset."""
        print_mini_banner("Inspect Dataset")

        # Select dataset
        dataset_path = self._select_dataset("Valitse analysoitava dataset:")
        if not dataset_path:
            return

        console.print(f"\n[cyan]Analysoidaan: {dataset_path.name}...[/cyan]\n")

        # Analyze
        stats = self.dataset_prep.inspect_dataset(dataset_path)

        # Show results
        console.print(Panel(
            f"[bold white]Tiedosto:[/bold white] {dataset_path.name}\n"
            f"[bold white]Formaatti:[/bold white] {stats.format_detected or 'Tuntematon'}\n"
            f"[bold white]Naytteita:[/bold white] {stats.total_samples:,}",
            title="[bold]Perustiedot[/bold]",
            border_style="cyan"
        ))

        # Schema
        if stats.schema:
            schema_table = Table(title="Skeema", box=box.SIMPLE)
            schema_table.add_column("Kentta", style="cyan")
            schema_table.add_column("Tyyppi", style="yellow")
            schema_table.add_column("Tayttoaste", justify="right")

            for field_name, field_type in stats.schema.items():
                fill_rate = stats.field_fill_rates.get(field_name, 0)
                schema_table.add_row(field_name, field_type, f"{fill_rate:.0f}%")

            console.print(schema_table)

        # Statistics
        console.print(Panel(
            f"[white]Merkkeja yhteensa:[/white] {stats.total_characters:,}\n"
            f"[white]Keskimaarin/nayte:[/white] {stats.avg_chars_per_sample:.0f} merkkia\n"
            f"[white]Min/Max:[/white] {stats.min_chars} - {stats.max_chars} merkkia\n"
            f"[white]Tyhjat rivit:[/white] {stats.empty_rows}\n"
            f"[white]Duplikaatit (arvio):[/white] {stats.duplicate_count}",
            title="[bold]Tilastot[/bold]",
            border_style="green"
        ))

        # Show examples
        if questionary.confirm("Nayta esimerkkeja?", default=True, style=custom_style).ask():
            samples = self.dataset_prep.preview_samples(dataset_path, n=3)
            console.print("\n[bold]Esimerkkeja:[/bold]")
            for i, sample in enumerate(samples, 1):
                console.print(f"\n[cyan]#{i}[/cyan]")
                # Show condensed version
                for key, value in sample.items():
                    if isinstance(value, str):
                        display_val = value[:100] + "..." if len(value) > 100 else value
                    else:
                        display_val = str(value)[:100]
                    console.print(f"  [yellow]{key}:[/yellow] {display_val}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _convert_format_wizard(self):
        """Convert dataset to another format."""
        print_mini_banner("Convert Format")

        # Select source
        source_path = self._select_dataset("Valitse lahde-dataset:")
        if not source_path:
            return

        # Detect current format
        source_format = self.dataset_prep.detect_format(source_path)
        console.print(f"\n[green]Tunnistettu formaatti:[/green] {source_format.value}\n")

        # Select target format
        format_choices = [
            questionary.Choice(title="Alpaca (instruction/input/output)", value="alpaca"),
            questionary.Choice(title="Chat (messages)", value="chat"),
            questionary.Choice(title="ShareGPT (conversations)", value="sharegpt"),
            questionary.Choice(title="Completion (prompt/completion)", value="completion"),
            questionary.Choice(title="JSONL (plain)", value="jsonl"),
        ]

        target = questionary.select(
            "Kohdeformaatti:",
            choices=format_choices,
            style=custom_style,
        ).ask()

        if not target:
            return

        # Define output path
        output_name = f"{source_path.stem}_{target}.jsonl"
        output_path = self.dataset_prep.output_dir / output_name

        # Confirm
        console.print(Panel(
            f"[white]Lahde:[/white] {source_path.name}\n"
            f"[white]Formaatti:[/white] {source_format.value} -> {target}\n"
            f"[white]Output:[/white] {output_path}",
            title="[bold]Muunnos[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita muunnos?", default=True, style=custom_style).ask():
            return

        # Execute
        console.print("\n[cyan]Muunnetaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        target_format_map = {
            "alpaca": DatasetFormat.ALPACA,
            "chat": DatasetFormat.CHAT,
            "sharegpt": DatasetFormat.SHAREGPT,
            "completion": DatasetFormat.COMPLETION,
            "jsonl": DatasetFormat.JSONL,
        }

        result = self.dataset_prep.convert_format(
            input_path=source_path,
            output_path=output_path,
            target_format=target_format_map[target],
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success(f"Muunnos valmis!")
            console.print(f"  [dim]Naytteita: {result['num_samples']:,}[/dim]")
            console.print(f"  [dim]Output: {output_path}[/dim]")
        else:
            print_error("Muunnos epäonnistui")
            for err in result.get("errors", [])[:5]:
                console.print(f"  [red]{err}[/red]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _split_dataset_wizard(self):
        """Split dataset into train/test/val parts."""
        print_mini_banner("Split Dataset")

        # Select dataset
        source_path = self._select_dataset("Valitse jaettava dataset:")
        if not source_path:
            return

        console.print("\n[cyan]Maarita jakosuhteet (yhteensa 100%):[/cyan]\n")

        # Ask for ratios
        train = questionary.text(
            "Train-osuus (%):",
            default="80",
            style=custom_style,
        ).ask()

        test = questionary.text(
            "Test-osuus (%):",
            default="10",
            style=custom_style,
        ).ask()

        val = questionary.text(
            "Validation-osuus (%):",
            default="10",
            style=custom_style,
        ).ask()

        try:
            train_ratio = float(train) / 100
            test_ratio = float(test) / 100
            val_ratio = float(val) / 100

            if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.01:
                print_warning("Suhteiden summa ei ole 100%!")
                questionary.press_any_key_to_continue(style=custom_style).ask()
                return
        except ValueError:
            print_error("Virheelliset arvot")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Shuffle
        shuffle = questionary.confirm("Sekoita data ennen jakoa?", default=True, style=custom_style).ask()

        # Output folder
        output_dir = self.dataset_prep.output_dir / source_path.stem

        config = SplitConfig(
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            validation_ratio=val_ratio,
            shuffle=shuffle,
            seed=42,
        )

        # Confirm
        console.print(Panel(
            f"[white]Lahde:[/white] {source_path.name}\n"
            f"[white]Jako:[/white] {train}% train, {test}% test, {val}% val\n"
            f"[white]Shuffle:[/white] {'Kylla' if shuffle else 'Ei'}\n"
            f"[white]Output:[/white] {output_dir}/",
            title="[bold]Split[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita jako?", default=True, style=custom_style).ask():
            return

        # Execute
        console.print("\n[cyan]Jaetaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.dataset_prep.split_dataset(
            input_path=source_path,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Jako valmis!")
            console.print(f"  [dim]Yhteensa: {result['total_samples']:,} naytetta[/dim]")
            for split_name, split_info in result["splits"].items():
                console.print(f"  [dim]{split_name}: {split_info['count']:,} naytetta[/dim]")
        else:
            print_error(f"Jako epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _clean_dataset_wizard(self):
        """Clean dataset."""
        print_mini_banner("Clean Dataset")

        # Select dataset
        source_path = self._select_dataset("Valitse siivottava dataset:")
        if not source_path:
            return

        # Select operations
        console.print("\n[cyan]Valitse siivoustoimenpiteet:[/cyan]\n")

        operation_choices = [
            questionary.Choice(title="Poista tyhjat rivit", value=CleaningOperation.REMOVE_EMPTY, checked=True),
            questionary.Choice(title="Korjaa merkistoongelmat", value=CleaningOperation.FIX_ENCODING, checked=True),
            questionary.Choice(title="Normalisoi valilyonnit", value=CleaningOperation.NORMALIZE_WHITESPACE, checked=True),
            questionary.Choice(title="Trimmaa tekstit", value=CleaningOperation.TRIM_TEXT, checked=False),
            questionary.Choice(title="Poista HTML-tagit", value=CleaningOperation.REMOVE_HTML, checked=False),
        ]

        operations = questionary.checkbox(
            "Valitse operaatiot:",
            choices=operation_choices,
            style=custom_style,
        ).ask()

        if not operations:
            return

        # Output
        output_path = self.dataset_prep.output_dir / f"{source_path.stem}_cleaned.jsonl"

        # Confirm
        op_names = [op.value for op in operations]
        console.print(Panel(
            f"[white]Lahde:[/white] {source_path.name}\n"
            f"[white]Operaatiot:[/white] {', '.join(op_names)}\n"
            f"[white]Output:[/white] {output_path}",
            title="[bold]Clean[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita siivous?", default=True, style=custom_style).ask():
            return

        # Execute
        console.print("\n[cyan]Siivoaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.dataset_prep.clean_dataset(
            input_path=source_path,
            output_path=output_path,
            operations=operations,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Siivous valmis!")
            console.print(f"  [dim]Alkuperainen: {result['original_count']:,} naytetta[/dim]")
            console.print(f"  [dim]Lopullinen: {result['final_count']:,} naytetta[/dim]")
            console.print(f"  [dim]Poistettu: {result['removed_count']:,} naytetta[/dim]")
        else:
            print_error(f"Siivous epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _dedupe_dataset_wizard(self):
        """Remove duplicates."""
        print_mini_banner("Deduplicate")

        # Select dataset
        source_path = self._select_dataset("Valitse dataset:")
        if not source_path:
            return

        # Select method
        method_choices = [
            questionary.Choice(title="Tarkka (hash)", value="exact"),
        ]

        # Add fuzzy if rapidfuzz available
        if self.dataset_prep._deps["rapidfuzz"]:
            method_choices.append(questionary.Choice(title="Sumea (fuzzy matching)", value="fuzzy"))
        else:
            console.print("[dim]Sumea dedupe ei saatavilla (asenna: pip install rapidfuzz)[/dim]\n")

        method = questionary.select(
            "Dedupe-metodi:",
            choices=method_choices,
            style=custom_style,
        ).ask()

        if not method:
            return

        threshold = 0.9
        if method == "fuzzy":
            threshold_str = questionary.text(
                "Samankaltaisuusraja (0.0-1.0):",
                default="0.9",
                style=custom_style,
            ).ask()
            try:
                threshold = float(threshold_str)
            except ValueError:
                threshold = 0.9

        # Output
        output_path = self.dataset_prep.output_dir / f"{source_path.stem}_deduped.jsonl"

        # Execute
        console.print("\n[cyan]Poistetaan duplikaatit...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.dataset_prep.deduplicate(
            input_path=source_path,
            output_path=output_path,
            method=method,
            threshold=threshold,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Dedupe valmis!")
            console.print(f"  [dim]Alkuperainen: {result['original_count']:,}[/dim]")
            console.print(f"  [dim]Lopullinen: {result['final_count']:,}[/dim]")
            console.print(f"  [dim]Duplikaatteja poistettu: {result['duplicates_removed']:,}[/dim]")
        else:
            print_error(f"Dedupe epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _filter_dataset_wizard(self):
        """Filter dataset by length."""
        print_mini_banner("Filter by Length")

        # Select dataset
        source_path = self._select_dataset("Valitse dataset:")
        if not source_path:
            return

        console.print("\n[cyan]Maarita suodatuskriteerit:[/cyan]")
        console.print("[dim]Jata tyhjäksi ohittaaksesi[/dim]\n")

        # Ask for criteria
        min_chars_str = questionary.text("Min merkkimaara:", default="", style=custom_style).ask()
        max_chars_str = questionary.text("Max merkkimaara:", default="", style=custom_style).ask()

        # Parse with error handling
        try:
            min_chars_val = int(min_chars_str) if min_chars_str else None
        except ValueError:
            print_warning(f"Virheellinen min merkkimaara '{min_chars_str}', ohitetaan")
            min_chars_val = None

        try:
            max_chars_val = int(max_chars_str) if max_chars_str else None
        except ValueError:
            print_warning(f"Virheellinen max merkkimaara '{max_chars_str}', ohitetaan")
            max_chars_val = None

        config = FilterConfig(
            min_chars=min_chars_val,
            max_chars=max_chars_val,
        )

        # Token-based filtering
        if self.dataset_prep._deps["transformers"]:
            use_tokens = questionary.confirm("Suodata myos tokenimaaran mukaan?", default=False, style=custom_style).ask()
            if use_tokens:
                min_tokens_str = questionary.text("Min tokeneita:", default="", style=custom_style).ask()
                max_tokens_str = questionary.text("Max tokeneita:", default="", style=custom_style).ask()
                try:
                    config.min_tokens = int(min_tokens_str) if min_tokens_str else None
                except ValueError:
                    print_warning(f"Virheellinen min tokeneita '{min_tokens_str}', ohitetaan")
                    config.min_tokens = None
                try:
                    config.max_tokens = int(max_tokens_str) if max_tokens_str else None
                except ValueError:
                    print_warning(f"Virheellinen max tokeneita '{max_tokens_str}', ohitetaan")
                    config.max_tokens = None

        # Output
        output_path = self.dataset_prep.output_dir / f"{source_path.stem}_filtered.jsonl"

        # Execute
        console.print("\n[cyan]Suodatetaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        tokenizer_name = "gpt2" if config.min_tokens or config.max_tokens else None

        result = self.dataset_prep.filter_dataset(
            input_path=source_path,
            output_path=output_path,
            config=config,
            tokenizer_name=tokenizer_name,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Suodatus valmis!")
            console.print(f"  [dim]Alkuperainen: {result['original_count']:,}[/dim]")
            console.print(f"  [dim]Hyvaksytty: {result['filtered_count']:,}[/dim]")
            console.print(f"  [dim]Liian lyhyita: {result['removed_too_short']:,}[/dim]")
            console.print(f"  [dim]Liian pitkia: {result['removed_too_long']:,}[/dim]")
        else:
            print_error(f"Suodatus epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _count_tokens_wizard(self):
        """Count tokens."""
        print_mini_banner("Token Counter")

        if not self.dataset_prep._deps["transformers"]:
            print_error("Token counting vaatii transformers-kirjaston")
            console.print("[dim]Asenna: pip install transformers[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select dataset
        source_path = self._select_dataset("Valitse dataset:")
        if not source_path:
            return

        # Select tokenizer
        tokenizer_name = questionary.text(
            "Tokenizer (HuggingFace model ID):",
            default="gpt2",
            style=custom_style,
        ).ask()

        if not tokenizer_name:
            return

        # Sample size
        sample_size = questionary.text(
            "Naytekoko (tyhja = kaikki):",
            default="1000",
            style=custom_style,
        ).ask()

        try:
            sample_limit = int(sample_size) if sample_size else None
        except ValueError:
            print_warning(f"Virheellinen näytekoko '{sample_size}', käytetään oletusta 1000")
            sample_limit = 1000

        # Execute
        console.print("\n[cyan]Lasketaan tokeneita...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.dataset_prep.count_tokens(
            file_path=source_path,
            tokenizer_name=tokenizer_name,
            sample_size=sample_limit,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Laskenta valmis!")
            console.print(Panel(
                f"[white]Tokenizer:[/white] {tokenizer_name}\n"
                f"[white]Naytteita:[/white] {result['samples_counted']:,}\n"
                f"[white]Tokeneita yhteensa:[/white] {result['total_tokens']:,}\n"
                f"[white]Keskimaarin/nayte:[/white] {result['avg_tokens_per_sample']:.0f}\n"
                f"[white]Min/Max:[/white] {result['min_tokens']} - {result['max_tokens']}",
                title="[bold]Token Statistics[/bold]",
                border_style="green"
            ))
        else:
            print_error(f"Laskenta epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _merge_datasets_wizard(self):
        """Merge multiple datasets."""
        print_mini_banner("Merge Datasets")

        console.print("[cyan]Valitse yhdistettavat datasetit:[/cyan]")
        console.print("[dim]Voit valita useita painamalla valilyontia[/dim]\n")

        datasets = self.dataset_prep.list_datasets(include_processed=True)

        if len(datasets) < 2:
            print_warning("Tarvitaan vahintaan 2 datasettia yhdistamiseen")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Multi-select
        choices = []
        for ds in datasets:
            size = format_size(ds["size_bytes"])
            title = f"{ds['name']:<35} {ds['format']:<10} {size:>10}"
            choices.append(questionary.Choice(title=title, value=ds["path"]))

        selected = questionary.checkbox(
            "Valitse datasetit:",
            choices=choices,
            style=custom_style,
        ).ask()

        if not selected or len(selected) < 2:
            print_warning("Valitse vahintaan 2 datasettia")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Options
        dedupe = questionary.confirm("Poista duplikaatit yhdistamisen jalkeen?", default=True, style=custom_style).ask()
        shuffle = questionary.confirm("Sekoita yhdistetty data?", default=True, style=custom_style).ask()

        # Output name
        output_name = questionary.text(
            "Yhdistetyn datasetin nimi:",
            default="merged_dataset.jsonl",
            style=custom_style,
        ).ask()

        if not output_name:
            return

        output_path = self.dataset_prep.output_dir / output_name

        # Execute
        console.print("\n[cyan]Yhdistetaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.dataset_prep.merge_datasets(
            input_paths=selected,
            output_path=output_path,
            deduplicate_result=dedupe,
            shuffle=shuffle,
            progress_callback=progress_cb,
        )

        if result["success"]:
            print_success("Yhdistaminen valmis!")
            console.print(f"  [dim]Tiedostoja: {result['files_merged']}[/dim]")
            console.print(f"  [dim]Yhteensa: {result['total_samples']:,} naytetta[/dim]")
            if result['duplicates_removed'] > 0:
                console.print(f"  [dim]Duplikaatteja poistettu: {result['duplicates_removed']:,}[/dim]")
            console.print(f"  [dim]Lopullinen: {result['final_count']:,} naytetta[/dim]")
        else:
            print_error(f"Yhdistaminen epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()
