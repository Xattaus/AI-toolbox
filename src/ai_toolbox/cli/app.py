"""
AI TOOLBOX - CLI Application
============================

Main CLI application entry point.
Beautiful TUI with orange branding and clean Finnish interface.
"""

import questionary

from ..core.ui import (
    console,
    print_toolbox_banner,
    print_error,
    MENU_STYLE,
    format_menu_item,
    menu_separator,
    print_branded_footer,
)
from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader
from ..conversion.converter import GGUFConverter
from ..training.lora import LoRATrainer
from ..training.dataset import DatasetPrep
from ..inference.benchmark import BenchmarkRunner
from ..inference.chat import ai_chat_menu
from ..inference.assistant import ai_assistant_menu
from ..merging.merger import ModelMerger
from ..abliteration.abliterator import Abliterator

# Unified command modules
from .model_hub_cmd import ModelHubCommands
from .gguf_tools_cmd import GGUFToolsCommands
from .training_center_cmd import TrainingCenterCommands
from .benchmark_cmd import BenchmarkCommands
from .settings_cmd import SettingsCommands
from .ollama_cmd import run_ollama_wizard


class AIToolbox:
    """Main AI Toolbox CLI application."""

    def __init__(self):
        """Initialize the toolbox with all components."""
        # Core services
        self.library = ModelLibrary()
        self.downloader = ModelDownloader()
        self.converter = GGUFConverter()
        self.trainer = LoRATrainer()
        self.dataset_prep = DatasetPrep()
        self.benchmark = BenchmarkRunner()
        self.merger = ModelMerger()
        self.abliterator = Abliterator()

        # Unified command handlers (new structure)
        self.model_hub_cmd = ModelHubCommands(
            self.library, self.downloader, self.converter
        )
        self.gguf_tools_cmd = GGUFToolsCommands(
            self.library, self.converter, self.downloader
        )
        self.training_center_cmd = TrainingCenterCommands(
            self.trainer, self.dataset_prep, self.merger,
            self.abliterator, self.library, self.downloader
        )
        self.benchmark_cmd = BenchmarkCommands(self.benchmark, self.library)
        self.settings_cmd = SettingsCommands(self.downloader, self.library)

        # Running state
        self.running = True

    def run(self):
        """Run the main application loop."""
        print_toolbox_banner()

        consecutive_errors = 0
        while self.running:
            try:
                self.main_menu()
                consecutive_errors = 0
            except KeyboardInterrupt:
                console.print("\n")
                if self._confirm_exit():
                    break
            except EOFError:
                # Ctrl+D tai suljettu stdin - poistu siististi ettei jaa
                # ikuiseen virheluuppiin (Exception nappaisi taman muuten)
                console.print("\n")
                break
            except Exception as e:
                print_error(f"Error occurred: {e}")
                # Suojaa ikuiselta virheluupilta jos sama virhe toistuu
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print_error("Liian monta perakkaista virhetta - poistutaan.")
                    break

        print_branded_footer("Kiitos kun käytit AI Toolboxia!")

    def main_menu(self):
        """Display and handle the main menu."""
        # Build menu with consistent formatting
        choices = [
            menu_separator("Keskustelu"),
            questionary.Choice(
                title=format_menu_item("Tool Master", "AI-chat: kysy malleista ja työkaluista"),
                value="chat"
            ),
            questionary.Choice(
                title=format_menu_item("Claude Assistant", "Käynnistä Claude CLI kehitykseen"),
                value="assistant"
            ),
            menu_separator("Mallien hallinta"),
            questionary.Choice(
                title=format_menu_item("Model Hub", "Lataa, selaa ja hallitse malleja"),
                value="model_hub"
            ),
            questionary.Choice(
                title=format_menu_item("GGUF Tools", "Muunna, kvantisoi ja laske VRAM"),
                value="gguf_tools"
            ),
            questionary.Choice(
                title=format_menu_item("Ollama Manager", "Luo ja hallitse Ollama-malleja"),
                value="ollama"
            ),
            menu_separator("Kehittyneet työkalut"),
            questionary.Choice(
                title=format_menu_item("Training Center", "LoRA, datasetit, yhdistäminen"),
                value="training_center"
            ),
            questionary.Choice(
                title=format_menu_item("Benchmark Suite", "Suorituskykytestaus"),
                value="benchmark"
            ),
            menu_separator(),
            questionary.Choice(
                title=format_menu_item("Asetukset", "Polut ja konfiguraatio"),
                value="settings"
            ),
            questionary.Choice(
                title=format_menu_item("<- Poistu", ""),
                value="exit"
            ),
        ]

        choice = questionary.select(
            "Valitse toiminto:",
            choices=choices,
            style=MENU_STYLE,
            qmark="#",
            pointer=">"
        ).ask()

        if choice is None or choice == "exit":
            self.running = False
        elif choice == "chat":
            ai_chat_menu(library=self.library, downloader=self.downloader, converter=self.converter)
        elif choice == "assistant":
            ai_assistant_menu()
        elif choice == "model_hub":
            self.model_hub_cmd.model_hub_menu()
        elif choice == "gguf_tools":
            self.gguf_tools_cmd.gguf_tools_menu()
        elif choice == "ollama":
            run_ollama_wizard()
        elif choice == "training_center":
            self.training_center_cmd.training_center_menu()
        elif choice == "benchmark":
            self.benchmark_cmd.benchmark_menu()
        elif choice == "settings":
            self.settings_cmd.settings_menu()

    def _confirm_exit(self) -> bool:
        """Confirm exit from the application."""
        return questionary.confirm(
            "Poistu AI Toolboxista?",
            style=MENU_STYLE,
            default=True
        ).ask()


def main():
    """Entry point for the CLI."""
    app = AIToolbox()
    app.run()


if __name__ == "__main__":
    main()
