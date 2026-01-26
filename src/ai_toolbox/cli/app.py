"""
AI TOOLBOX - CLI Application
============================

Main CLI application entry point.
Restructured menu: 16 options → 7 options for better usability.
"""

import questionary
from questionary import Style

from ..core.ui import console, print_toolbox_banner, print_error
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

# New unified command modules
from .model_hub_cmd import ModelHubCommands
from .gguf_tools_cmd import GGUFToolsCommands
from .training_center_cmd import TrainingCenterCommands
from .benchmark_cmd import BenchmarkCommands
from .settings_cmd import SettingsCommands
from .ollama_cmd import run_ollama_wizard

# Questionary style used throughout the app
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

        while self.running:
            try:
                self.main_menu()
            except KeyboardInterrupt:
                console.print("\n")
                if self._confirm_exit():
                    break
            except Exception as e:
                print_error(f"Error occurred: {e}")

        console.print("\n[bold cyan]Thank you for using AI TOOLBOX![/bold cyan]\n")

    def main_menu(self):
        """Display and handle the main menu (7 options)."""
        choices = [
            questionary.Separator("── Chat & Assistentit ──"),
            questionary.Choice(
                title="  Tool Master           Chat with AI Master",
                value="chat"
            ),
            questionary.Choice(
                title="  Claude Assistant      Claude CLI for development",
                value="assistant"
            ),
            questionary.Separator("── Mallien hallinta ──"),
            questionary.Choice(
                title="  Model Hub             Download, browse & manage models",
                value="model_hub"
            ),
            questionary.Choice(
                title="  GGUF Tools            Convert, quantize & VRAM calc",
                value="gguf_tools"
            ),
            questionary.Choice(
                title="  Ollama Manager        Create and manage Ollama models",
                value="ollama"
            ),
            questionary.Separator("── Kehittyneet tyokalut ──"),
            questionary.Choice(
                title="  Training Center       LoRA, datasets, merging & abliteration",
                value="training_center"
            ),
            questionary.Choice(
                title="  Benchmark Suite       Performance testing & comparison",
                value="benchmark"
            ),
            questionary.Separator(),
            questionary.Choice(
                title="  Asetukset             Configuration",
                value="settings"
            ),
            questionary.Choice(
                title="  Poistu                Exit",
                value="exit"
            ),
        ]

        choice = questionary.select(
            "AI TOOLBOX:",
            choices=choices,
            style=custom_style,
            qmark=">>",
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
            "Exit AI Toolbox?",
            style=custom_style,
            default=True
        ).ask()


def main():
    """Entry point for the CLI."""
    app = AIToolbox()
    app.run()


if __name__ == "__main__":
    main()
