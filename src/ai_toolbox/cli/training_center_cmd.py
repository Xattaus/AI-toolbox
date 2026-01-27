"""
AI TOOLBOX - Training Center Commands
=====================================

Unified CLI for LoRA training, dataset preparation, model merging, and abliteration.
Combines the functionality of training_cmd.py, dataset_cmd.py, merger_cmd.py, and abliteration_cmd.py.
"""

import questionary
from rich.panel import Panel

from ..core.ui import (
    console,
    print_mini_banner,
    print_branded_header,
    print_warning,
    format_menu_item,
    MENU_STYLE,
)
from ..training.lora import LoRATrainer
from ..training.dataset import DatasetPrep
from ..merging.merger import ModelMerger
from ..abliteration.abliterator import Abliterator
from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader

# Import the original command classes to reuse their functionality
from .training_cmd import TrainingCommands
from .dataset_cmd import DatasetCommands
from .merger_cmd import MergerCommands
from .abliteration_cmd import AbliterationCommands

# Use unified menu style
custom_style = MENU_STYLE


class TrainingCenterCommands:
    """Unified CLI commands for training, datasets, merging, and abliteration."""

    def __init__(
        self,
        trainer: LoRATrainer,
        dataset_prep: DatasetPrep,
        merger: ModelMerger,
        abliterator: Abliterator,
        library: ModelLibrary,
        downloader: ModelDownloader,
    ):
        """
        Initialize Training Center commands.

        Args:
            trainer: LoRA trainer instance
            dataset_prep: Dataset preparation instance
            merger: Model merger instance
            abliterator: Abliterator instance
            library: Model library instance
            downloader: Model downloader instance
        """
        self.trainer = trainer
        self.dataset_prep = dataset_prep
        self.merger = merger
        self.abliterator = abliterator
        self.library = library
        self.downloader = downloader

        # Create wrapped command handlers
        self.training_cmd = TrainingCommands(trainer, library, downloader, merger)
        self.dataset_cmd = DatasetCommands(dataset_prep)
        self.merger_cmd = MergerCommands(merger, library, downloader)
        self.abliteration_cmd = AbliterationCommands(abliterator, library, downloader)

    def training_center_menu(self):
        """Training Center main menu."""
        while True:
            print_branded_header("Training Center", "LoRA, datasetit, merget ja abliterointi")

            # Show status summary
            lora_status = self.trainer.get_status()
            merger_status = self.merger.get_status()
            abliter_status = self.abliterator.get_status()
            dataset_status = self.dataset_prep.get_status()

            # Status line with colors
            status_parts = []
            lora_color = "green" if lora_status["ready"] else "yellow"
            merger_color = "green" if merger_status["ready"] else "yellow"
            abliter_color = "green" if abliter_status["ready"] else "yellow"

            status_parts.append(f"[dim]LoRA:[/dim] [{lora_color}]{'OK' if lora_status['ready'] else '—'}[/{lora_color}]")
            status_parts.append(f"[dim]Merger:[/dim] [{merger_color}]{'OK' if merger_status['ready'] else '—'}[/{merger_color}]")
            status_parts.append(f"[dim]Abliter:[/dim] [{abliter_color}]{'OK' if abliter_status['ready'] else '—'}[/{abliter_color}]")
            status_parts.append(f"[dim]Datasets:[/dim] [cyan]{dataset_status['datasets_count']}[/cyan]")

            console.print("  " + "  |  ".join(status_parts) + "\n")

            choices = [
                questionary.Separator("--- LoRA Training ---"),
                questionary.Choice(
                    title=format_menu_item("Quick Train", "Pikakoulutus oletusasetuksilla"),
                    value="lora_quick"
                ),
                questionary.Choice(
                    title=format_menu_item("Advanced Train", "Taydella parametrikontrollilla"),
                    value="lora_advanced"
                ),
                questionary.Choice(
                    title=format_menu_item("Test Adapter", "Testaa koulutettu adapteri"),
                    value="lora_test"
                ),
                questionary.Choice(
                    title=format_menu_item("Merge Adapter", "Yhdista adapteri base-malliin"),
                    value="lora_merge"
                ),
                questionary.Separator("--- Dataset Tools ---"),
                questionary.Choice(
                    title=format_menu_item("Inspect", "Tarkasta datasetin rakenne"),
                    value="dataset_inspect"
                ),
                questionary.Choice(
                    title=format_menu_item("Convert", "Muunna formaattien valilla"),
                    value="dataset_convert"
                ),
                questionary.Choice(
                    title=format_menu_item("Clean & Filter", "Siivoa, suodata, deduplikoi"),
                    value="dataset_clean"
                ),
                questionary.Choice(
                    title=format_menu_item("More Tools...", "Lisaa dataset-tyokaluja"),
                    value="dataset_more"
                ),
                questionary.Separator("--- Model Merging ---"),
                questionary.Choice(
                    title=format_menu_item("Mergekit Wizard", "Kaikki merge-tyokalut"),
                    value="mergekit_wizard"
                ),
                questionary.Separator("--- Abliteration ---"),
                questionary.Choice(
                    title=format_menu_item("Remove Censorship", "Poista kieltaytymiskaytos"),
                    value="abliterate"
                ),
                questionary.Choice(
                    title=format_menu_item("Test Model", "Testaa abliteroitu malli"),
                    value="abliter_test"
                ),
                questionary.Separator("-----------------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Back", "Palaa pavalikkoon"),
                    value="back"
                ),
            ]

            choice = questionary.select(
                "",
                choices=choices,
                style=custom_style,
                qmark="",
                pointer=">",
                instruction="(↑↓ valitse, Enter vahvista)"
            ).ask()

            if choice is None or choice == "back":
                break

            # LoRA Training
            elif choice == "lora_quick":
                if not lora_status["ready"]:
                    self._show_install_message("LoRA Trainer", self.training_cmd._install_lora_deps)
                else:
                    self.training_cmd._quick_train_wizard()
            elif choice == "lora_advanced":
                if not lora_status["ready"]:
                    self._show_install_message("LoRA Trainer", self.training_cmd._install_lora_deps)
                else:
                    self.training_cmd._advanced_train_wizard()
            elif choice == "lora_test":
                if not lora_status["ready"]:
                    self._show_install_message("LoRA Trainer", self.training_cmd._install_lora_deps)
                else:
                    self.training_cmd._test_adapter_wizard()
            elif choice == "lora_merge":
                if not lora_status["ready"]:
                    self._show_install_message("LoRA Trainer", self.training_cmd._install_lora_deps)
                else:
                    self.training_cmd._merge_adapter_wizard()

            # Dataset Tools
            elif choice == "dataset_inspect":
                self.dataset_cmd._inspect_dataset_wizard()
            elif choice == "dataset_convert":
                self.dataset_cmd._convert_format_wizard()
            elif choice == "dataset_clean":
                self._dataset_clean_menu()
            elif choice == "dataset_more":
                self._dataset_more_menu()

            # Mergekit Wizard (kaikki merge-työkalut)
            elif choice == "mergekit_wizard":
                self.merger_cmd._mergekit_main_menu()

            # Abliteration
            elif choice == "abliterate":
                if not abliter_status["ready"]:
                    self._show_install_message("Abliteration", self.abliteration_cmd._install_deps)
                else:
                    self.abliteration_cmd._full_abliteration_wizard()
            elif choice == "abliter_test":
                self.abliteration_cmd._test_model_wizard()

    def _dataset_clean_menu(self):
        """Dataset cleaning sub-menu."""
        while True:
            print_mini_banner("Clean & Filter", "Siivoa ja suodata datasetteja")

            choices = [
                questionary.Separator("--- Cleaning ---"),
                questionary.Choice(
                    title=format_menu_item("Clean Dataset", "Siivoa ja normalisoi"),
                    value="clean"
                ),
                questionary.Choice(
                    title=format_menu_item("Deduplicate", "Poista duplikaatit"),
                    value="dedupe"
                ),
                questionary.Choice(
                    title=format_menu_item("Filter by Length", "Suodata pituuden mukaan"),
                    value="filter"
                ),
                questionary.Separator("---------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Back", "Palaa valikkoon"),
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
            elif choice == "clean":
                self.dataset_cmd._clean_dataset_wizard()
            elif choice == "dedupe":
                self.dataset_cmd._dedupe_dataset_wizard()
            elif choice == "filter":
                self.dataset_cmd._filter_dataset_wizard()

    def _dataset_more_menu(self):
        """Additional dataset tools sub-menu."""
        while True:
            print_mini_banner("More Tools", "Lisaa dataset-tyokaluja")

            choices = [
                questionary.Separator("--- Operations ---"),
                questionary.Choice(
                    title=format_menu_item("Split Dataset", "Jaa train/test/val osiin"),
                    value="split"
                ),
                questionary.Choice(
                    title=format_menu_item("Count Tokens", "Laske tokenien maara"),
                    value="tokens"
                ),
                questionary.Choice(
                    title=format_menu_item("Merge Datasets", "Yhdista useita datasetteja"),
                    value="merge"
                ),
                questionary.Choice(
                    title=format_menu_item("Browse", "Selaa kaikki datasetit"),
                    value="browse"
                ),
                questionary.Separator("---------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Back", "Palaa valikkoon"),
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
            elif choice == "split":
                self.dataset_cmd._split_dataset_wizard()
            elif choice == "tokens":
                self.dataset_cmd._count_tokens_wizard()
            elif choice == "merge":
                self.dataset_cmd._merge_datasets_wizard()
            elif choice == "browse":
                self.dataset_cmd._browse_datasets()

    def _show_install_message(self, tool_name: str, install_func):
        """Show install prompt for missing dependencies."""
        console.print(Panel(
            f"[yellow]{tool_name} ei ole valmis.[/yellow]\n\n"
            f"Puuttuvat riippuvuudet on asennettava ensin.",
            title="[bold yellow]Riippuvuudet puuttuvat[/bold yellow]",
            border_style="yellow"
        ))

        if questionary.confirm(
            f"Asenna {tool_name} -riippuvuudet nyt?",
            default=True,
            style=custom_style
        ).ask():
            install_func()
        else:
            print_warning("Asennus peruutettu")
            questionary.press_any_key_to_continue(style=custom_style).ask()
