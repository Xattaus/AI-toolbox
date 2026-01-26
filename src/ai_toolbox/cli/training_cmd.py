"""
AI TOOLBOX - Training Commands
==============================

CLI commands for LoRA training and fine-tuning.
"""

import json
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
from ..training.lora import (
    LoRATrainer,
    LoRAConfig,
    TrainingConfig,
    FullConfig,
    UnslothCompatibilityResult,
)
from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader
from ..merging.merger import ModelMerger

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


class TrainingCommands:
    """CLI commands for LoRA training."""

    def __init__(
        self,
        trainer: LoRATrainer,
        library: ModelLibrary,
        downloader: ModelDownloader,
        merger: ModelMerger,
    ):
        """
        Initialize training commands.

        Args:
            trainer: LoRA trainer instance
            library: Model library instance
            downloader: Model downloader instance
            merger: Model merger instance
        """
        self.trainer = trainer
        self.library = library
        self.downloader = downloader
        self.merger = merger

    def lora_trainer_menu(self):
        """LoRA Trainer sub-menu."""
        while True:
            print_mini_banner("LoRA Trainer")

            # Check status
            status = self.trainer.get_status()

            if not status["ready"]:
                missing = ", ".join(status["missing_required"])
                console.print(f"[yellow]Puuttuvat riippuvuudet: {missing}[/yellow]")
                console.print("[dim]Valitse 'Install Dependencies' asentaaksesi[/dim]\n")
            else:
                # Show available features
                features = []
                if status["has_quantization"]:
                    features.append("QLoRA")

                # Unsloth status - yksityiskohtaisempi
                if status["unsloth_available"]:
                    if status["unsloth_gpu_ok"] and status["unsloth_os_ok"]:
                        features.append(f"Unsloth v{status['unsloth_version']} [green]OK[/green]")
                    else:
                        features.append(f"Unsloth [yellow]rajoitettu[/yellow]")
                else:
                    pass  # Ei mainita jos ei asennettu

                if features:
                    console.print(f"[green]Kiihdytykset: {', '.join(features)}[/green]")

            console.print(f"[dim]Output: {status['output_dir']}[/dim]")
            console.print(f"[dim]Datasets: {status['datasets_dir']}[/dim]\n")

            choices = [
                questionary.Choice(
                    title="Quick Train             Nopea aloitus oletusasetuksilla",
                    value="quick"
                ),
                questionary.Choice(
                    title="Advanced Training       Taysi kontrolli parametreihin",
                    value="advanced"
                ),
                questionary.Separator(),
                questionary.Choice(
                    title="Dataset Manager         Hallitse datasetteja",
                    value="datasets"
                ),
                questionary.Choice(
                    title="Test Adapter            Testaa valmista adapteria",
                    value="test"
                ),
                questionary.Choice(
                    title="Merge Adapter           Yhdista adapter base-malliin",
                    value="merge"
                ),
                questionary.Separator(),
                questionary.Choice(
                    title="LoRA Info               Tietoa LoRA:sta ja parametreista",
                    value="info"
                ),
            ]

            if not status["ready"]:
                choices.append(questionary.Choice(
                    title="Install Dependencies    Asenna tarvittavat kirjastot",
                    value="install"
                ))

            choices.extend([
                questionary.Separator(),
                questionary.Choice(title="Back                    Palaa", value="back"),
            ])

            choice = questionary.select(
                "LoRA Trainer:",
                choices=choices,
                style=custom_style,
                qmark=">>",
                pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "install":
                self._install_lora_deps()
            elif choice == "info":
                self._show_lora_info()
            elif choice == "datasets":
                self._dataset_manager()
            elif choice == "quick":
                if not status["ready"]:
                    print_warning("Asenna ensin riippuvuudet")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                else:
                    self._quick_train_wizard()
            elif choice == "advanced":
                if not status["ready"]:
                    print_warning("Asenna ensin riippuvuudet")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                else:
                    self._advanced_train_wizard()
            elif choice == "test":
                if not status["ready"]:
                    print_warning("Asenna ensin riippuvuudet")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                else:
                    self._test_adapter_wizard()
            elif choice == "merge":
                if not status["ready"]:
                    print_warning("Asenna ensin riippuvuudet")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                else:
                    self._merge_adapter_wizard()

    def _install_lora_deps(self):
        """Install LoRA Trainer dependencies."""
        print_mini_banner("Install Dependencies")

        console.print("[cyan]Pakolliset:[/cyan] torch, transformers, peft, datasets, trl")
        console.print("[cyan]Valinnaiset:[/cyan] bitsandbytes (QLoRA), unsloth (kiihdytys)\n")

        include_optional = questionary.confirm(
            "Asenna myos valinnaiset (bitsandbytes)?",
            default=True,
            style=custom_style
        ).ask()

        console.print("\n[cyan]Asennetaan riippuvuuksia...[/cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        if self.trainer.install_dependencies(include_optional=include_optional, progress_callback=progress_cb):
            print_success("Riippuvuudet asennettu!")
        else:
            print_error("Asennus epäonnistui")
            console.print("\n[dim]Kokeile manuaalisesti:[/dim]")
            console.print("[cyan]pip install torch transformers peft datasets trl[/cyan]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_lora_info(self):
        """Show LoRA information."""
        print_mini_banner("LoRA Info")

        info_text = """
[bold cyan]LoRA (Low-Rank Adaptation)[/bold cyan]

LoRA on tehokas fine-tuning -menetelma, joka kouluttaa vain
pienen osan mallin parametreista. Tama saastaa muistia ja
aikaa verrattuna tayteen fine-tuningiin.

[bold white]Kiihdytysvaihtoehdot:[/bold white]

[cyan]PEFT/transformers[/cyan] (oletus)
  - Vakaa ja laajasti yhteensopiva
  - Toimii kaikilla malleilla ja laitteistoilla

[cyan]Unsloth[/cyan] (suositeltu kun saatavilla)
  - 2-5x nopeampi training
  - 50-70% vahemman VRAM
  - Vaatii: NVIDIA GPU (Turing+), Linux/Windows
  - Tuetut mallit: Llama, Mistral, Qwen, Phi, Gemma

[bold white]Keskeiset parametrit:[/bold white]

[cyan]Rank (r)[/cyan]
  Matriisien dimensio. Suurempi = enemman kapasiteettia.
  - Suositus: 8-16 (aloita 16:sta)
  - Pienille malleille: 32
  - Isoille malleille: 8

[cyan]Alpha[/cyan]
  Skaalauskerroin. Yleensa 2x rank.
  - Suositus: 32 (jos rank=16)
  - Kaava: lora_weight = (alpha/rank) * learned_weight

[cyan]Target Modules[/cyan]
  Mitka kerrokset koulutetaan.
  - Tehokas: ["q_proj", "v_proj"]
  - Kattava: ["q_proj", "k_proj", "v_proj", "o_proj"]
  - Taysi: + ["gate_proj", "up_proj", "down_proj"]

[cyan]Learning Rate[/cyan]
  - LoRA: 1e-4 - 5e-4 (10x korkeampi kuin full fine-tune)
  - QLoRA: 2e-4 - 3e-4

[bold white]Dataset-formaatit:[/bold white]

[cyan]Alpaca[/cyan] (suositeltu):
  {"instruction": "...", "input": "...", "output": "..."}

[cyan]Chat/Messages[/cyan]:
  {"messages": [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}

[bold white]Muistivaatimukset (arvioit):[/bold white]

  PEFT:
    7B malli + LoRA:  ~8 GB VRAM
    7B malli + QLoRA: ~6 GB VRAM
    13B + QLoRA:      ~10 GB VRAM

  Unsloth:
    7B malli + LoRA:  ~5 GB VRAM
    7B malli + QLoRA: ~4 GB VRAM
    13B + QLoRA:      ~6 GB VRAM

[dim]Vinkki: Aloita pienella datasetilla ja tarkista tulokset
ennen isompaa ajoa.[/dim]
"""
        console.print(info_text)
        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _dataset_manager(self):
        """Dataset management."""
        while True:
            print_mini_banner("Dataset Manager")

            # List datasets
            datasets = self.trainer.list_datasets()
            if datasets:
                console.print(f"[dim]Datasetteja: {len(datasets)}[/dim]\n")
            else:
                console.print("[dim]Ei datasetteja. Luo tai kopioi datasets/-kansioon.[/dim]\n")

            choices = [
                questionary.Choice(
                    title="List Datasets           Nayta saatavilla olevat",
                    value="list"
                ),
                questionary.Choice(
                    title="Validate Dataset        Tarkista datasetin kelpoisuus",
                    value="validate"
                ),
                questionary.Choice(
                    title="Create Sample           Luo esimerkkidataset",
                    value="sample"
                ),
                questionary.Separator(),
                questionary.Choice(title="Back                    Palaa", value="back"),
            ]

            choice = questionary.select(
                "Dataset Manager:",
                choices=choices,
                style=custom_style,
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "list":
                self._list_datasets()
            elif choice == "validate":
                self._validate_dataset()
            elif choice == "sample":
                self._create_sample_dataset()

    def _list_datasets(self):
        """List datasets."""
        datasets = self.trainer.list_datasets()

        if not datasets:
            print_warning("Ei datasetteja")
            console.print(f"\n[dim]Kopioi datasetteja kansioon:[/dim]")
            console.print(f"[cyan]{self.trainer.datasets_dir}[/cyan]")
        else:
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
            table.add_column("Nimi", style="white")
            table.add_column("Formaatti", style="green")
            table.add_column("Koko", style="yellow", justify="right")

            for ds in datasets:
                size_str = format_size(ds["size_bytes"])
                table.add_row(ds["name"], ds["format"] or "?", size_str)

            console.print(table)

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _validate_dataset(self):
        """Validate dataset."""
        print_mini_banner("Validate Dataset")

        # Use common dataset selection
        dataset_path = self._select_dataset_for_training()
        if not dataset_path:
            return

        # Validate
        console.print(f"\n[cyan]Validoidaan: {dataset_path.name}...[/cyan]\n")
        result = self.trainer.validate_dataset(dataset_path)

        if result["valid"]:
            print_success("Dataset on validi!")
            console.print(f"\n[white]Formaatti:[/white] {result['format']}")
            console.print(f"[white]Naytteita:[/white] {result['num_samples']}")

            if result["sample"]:
                console.print(f"\n[white]Esimerkki:[/white]")
                console.print(f"[dim]{json.dumps(result['sample'], ensure_ascii=False, indent=2)[:500]}[/dim]")

            for warning in result["warnings"]:
                print_warning(warning)
        else:
            print_error("Dataset ei ole validi")
            for error in result["errors"]:
                console.print(f"  [red]- {error}[/red]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _create_sample_dataset(self):
        """Create sample dataset."""
        format_choices = [
            questionary.Choice(title="Alpaca (instruction/input/output)", value="alpaca"),
            questionary.Choice(title="Chat (messages)", value="chat"),
        ]

        format_type = questionary.select(
            "Valitse formaatti:",
            choices=format_choices,
            style=custom_style,
        ).ask()

        if not format_type:
            return

        num_samples = questionary.text(
            "Naytteiden maara:",
            default="10",
            style=custom_style,
        ).ask()

        try:
            num = int(num_samples)
        except ValueError:
            num = 10

        output_path = self.trainer.datasets_dir / f"sample_{format_type}.jsonl"

        if self.trainer.create_sample_dataset(output_path, format_type, num):
            print_success(f"Esimerkkidataset luotu: {output_path}")
        else:
            print_error("Luonti epäonnistui")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_model_for_training(self) -> Optional[Path]:
        """Select model for training."""
        choices = []

        # Library safetensors/pytorch models
        models = self.library.list_models(format_filter="safetensors")
        models.extend(self.library.list_models(format_filter="pytorch"))

        if models:
            choices.append(questionary.Separator("-- Kirjaston mallit --"))
            for m in models[:15]:
                size = format_size(m.size_bytes)
                choices.append(questionary.Choice(
                    title=f"[lib] {m.name[:30]:<30} {m.format.upper():<12} {size:>10}",
                    value=("library", m.path)
                ))

        # Downloaded HuggingFace models (downloads folder)
        downloaded = self.downloader.list_downloaded()
        if downloaded:
            choices.append(questionary.Separator("-- Ladatut HF-mallit --"))
            for d in downloaded[:10]:
                size = format_size(d['size'])
                name = d['model_id'].split('/')[-1] if '/' in d['model_id'] else d['model_id']
                choices.append(questionary.Choice(
                    title=f"[hf]  {name[:30]:<30} {'HF':<12} {size:>10}",
                    value=("download", d['path'])
                ))

        if not models and not downloaded:
            console.print("[yellow]Ei malleja kirjastossa tai ladattuna.[/yellow]")
            console.print(f"[dim]Lataa malli ensin: Model Download -> Search/Download[/dim]\n")

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value=("cancel", None)))

        result = questionary.select(
            "Valitse base-malli:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result[0] == "cancel":
            return None

        return Path(result[1])

    def _select_dataset_for_training(self) -> Optional[Path]:
        """Select dataset for training."""
        choices = []

        # Trainer datasets (datasets/ folder)
        datasets = self.trainer.list_datasets()
        if datasets:
            choices.append(questionary.Separator("-- Training Datasets --"))
            for ds in datasets:
                size = format_size(ds["size_bytes"])
                choices.append(questionary.Choice(
                    title=f"[ds]  {ds['name'][:30]:<30} {ds['format'] or '?':<10} {size:>10}",
                    value=ds["path"]
                ))

        # Processed datasets (datasets/processed/ folder)
        processed_dir = get_paths().processed_dir
        if processed_dir.exists():
            processed = list(processed_dir.glob("*.jsonl")) + list(processed_dir.glob("*.json"))
            if processed:
                choices.append(questionary.Separator("-- Processed Datasets --"))
                for p in processed[:10]:
                    size = format_size(p.stat().st_size)
                    choices.append(questionary.Choice(
                        title=f"[pr]  {p.name[:30]:<30} {'JSONL':<10} {size:>10}",
                        value=p
                    ))

        if not datasets and not (processed_dir.exists() and any(processed_dir.iterdir())):
            console.print("[yellow]Ei datasetteja.[/yellow]")
            console.print(f"[dim]Luo tai kasittele datasetteja: Dataset Prep[/dim]\n")

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value="cancel"))

        result = questionary.select(
            "Valitse dataset:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result == "cancel":
            return None

        return Path(result) if isinstance(result, str) else result

    def _quick_train_wizard(self):
        """Quick training wizard."""
        print_mini_banner("Quick Train")

        console.print("[cyan]Nopea aloitus oletusasetuksilla.[/cyan]")
        console.print("[dim]Kayttaa suositeltuja parametreja mallin koon mukaan.[/dim]\n")

        # 1. Select model
        console.print("[bold]1. Valitse base-malli:[/bold]")
        model_path = self._select_model_for_training()
        if not model_path:
            return

        # Get recommendations
        recommendations = self.trainer.get_recommended_config(model_path)
        console.print(f"  [green]+[/green] {model_path.name}")
        console.print(f"    [dim]Tyyppi: {recommendations['model_type']}, ~{recommendations['estimated_params_b']}B parametria[/dim]\n")

        # 2. Select dataset
        console.print("[bold]2. Valitse dataset:[/bold]")
        dataset_path = self._select_dataset_for_training()
        if not dataset_path:
            return

        # Validate
        validation = self.trainer.validate_dataset(dataset_path)
        if not validation["valid"]:
            print_error("Dataset ei ole validi")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        console.print(f"  [green]+[/green] {dataset_path.name}")
        console.print(f"    [dim]{validation['num_samples']} naytetta, formaatti: {validation['format']}[/dim]\n")

        # 3. Give name
        run_name = questionary.text(
            "Training-ajon nimi:",
            default=f"lora_{model_path.name[:20]}_{datetime.now().strftime('%Y%m%d')}",
            style=custom_style,
        ).ask()

        if not run_name:
            return

        # Create config with defaults
        preset = recommendations["preset"]
        config = FullConfig(
            model_path=str(model_path),
            model_name=model_path.name,
            lora=LoRAConfig(
                rank=preset["rank"],
                alpha=preset["rank"] * 2,
                target_modules=recommendations["target_modules"],
            ),
            training=TrainingConfig(
                batch_size=preset["batch_size"],
                learning_rate=preset["learning_rate"],
            ),
            dataset_path=str(dataset_path),
            dataset_format=validation["format"],
            run_name=run_name,
        )

        # Show summary
        console.print(Panel(
            f"[white]Malli:[/white] {model_path.name}\n"
            f"[white]Dataset:[/white] {dataset_path.name} ({validation['num_samples']} naytetta)\n"
            f"[white]LoRA rank:[/white] {config.lora.rank}, alpha: {config.lora.alpha}\n"
            f"[white]Batch size:[/white] {config.training.batch_size}\n"
            f"[white]Learning rate:[/white] {config.training.learning_rate}\n"
            f"[white]Epochs:[/white] {config.training.epochs}\n"
            f"[white]Output:[/white] {run_name}",
            title="[bold]Training Configuration[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita training?", default=True, style=custom_style).ask():
            return

        # Run training
        self._run_training(config)

    def _advanced_train_wizard(self):
        """Advanced training wizard."""
        print_mini_banner("Advanced Training")

        console.print("[cyan]Taysi kontrolli kaikkiin parametreihin.[/cyan]\n")

        # 1. Select model
        console.print("[bold]1. Base-malli:[/bold]")
        model_path = self._select_model_for_training()
        if not model_path:
            return

        recommendations = self.trainer.get_recommended_config(model_path)
        preset = recommendations["preset"]
        console.print(f"  [green]+[/green] {model_path.name}\n")

        # 2. Dataset
        console.print("[bold]2. Dataset:[/bold]")
        dataset_path = self._select_dataset_for_training()
        if not dataset_path:
            return

        validation = self.trainer.validate_dataset(dataset_path)
        if not validation["valid"]:
            print_error("Dataset ei ole validi")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return
        console.print(f"  [green]+[/green] {dataset_path.name}\n")

        # 3. LoRA parameters
        console.print("[bold]3. LoRA-parametrit:[/bold]")

        rank = questionary.text(
            f"Rank (suositus: {preset['rank']}):",
            default=str(preset["rank"]),
            style=custom_style,
        ).ask()
        rank = int(rank) if rank else preset["rank"]

        alpha = questionary.text(
            f"Alpha (suositus: {rank * 2}):",
            default=str(rank * 2),
            style=custom_style,
        ).ask()
        alpha = int(alpha) if alpha else rank * 2

        # Target modules
        module_choices = [
            questionary.Choice(title="Tehokas (q_proj, v_proj)", value="efficient"),
            questionary.Choice(title="Kattava (q,k,v,o_proj)", value="balanced"),
            questionary.Choice(title="Taysi (kaikki)", value="full"),
        ]
        module_choice = questionary.select(
            "Target modules:",
            choices=module_choices,
            style=custom_style,
        ).ask()

        if module_choice == "efficient":
            target_modules = ["q_proj", "v_proj"]
        elif module_choice == "balanced":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target_modules = recommendations["target_modules"]

        # 4. Training parameters
        console.print("\n[bold]4. Training-parametrit:[/bold]")

        learning_rate = questionary.text(
            f"Learning rate (suositus: {preset['learning_rate']}):",
            default=str(preset["learning_rate"]),
            style=custom_style,
        ).ask()
        learning_rate = float(learning_rate) if learning_rate else preset["learning_rate"]

        batch_size = questionary.text(
            f"Batch size (suositus: {preset['batch_size']}):",
            default=str(preset["batch_size"]),
            style=custom_style,
        ).ask()
        batch_size = int(batch_size) if batch_size else preset["batch_size"]

        epochs = questionary.text(
            "Epochs (suositus: 2):",
            default="2",
            style=custom_style,
        ).ask()
        epochs = int(epochs) if epochs else 2

        # 5. Quantization
        quant_choices = [
            questionary.Choice(title="Ei kvantisointia (taysi tarkkuus)", value=None),
            questionary.Choice(title="4-bit (QLoRA, saastaa muistia)", value="4bit"),
            questionary.Choice(title="8-bit", value="8bit"),
        ]
        quantization = questionary.select(
            "Quantization:",
            choices=quant_choices,
            style=custom_style,
        ).ask()

        # 6. Name
        run_name = questionary.text(
            "Training-ajon nimi:",
            default=f"lora_{model_path.name[:15]}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            style=custom_style,
        ).ask()

        # Create config
        config = FullConfig(
            model_path=str(model_path),
            model_name=model_path.name,
            lora=LoRAConfig(
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
            ),
            training=TrainingConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
            ),
            dataset_path=str(dataset_path),
            dataset_format=validation["format"],
            run_name=run_name,
            quantization=quantization,
        )

        # Show summary
        console.print(Panel(
            f"[white]Malli:[/white] {model_path.name}\n"
            f"[white]Dataset:[/white] {dataset_path.name}\n"
            f"[white]LoRA:[/white] rank={rank}, alpha={alpha}\n"
            f"[white]Modules:[/white] {', '.join(target_modules)}\n"
            f"[white]Training:[/white] lr={learning_rate}, batch={batch_size}, epochs={epochs}\n"
            f"[white]Quantization:[/white] {quantization or 'None'}\n"
            f"[white]Output:[/white] {run_name}",
            title="[bold]Advanced Configuration[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita training?", default=True, style=custom_style).ask():
            return

        # Run
        self._run_training(config)

    def _check_and_ask_unsloth(self, config: FullConfig) -> Optional[bool]:
        """
        Tarkista Unsloth-yhteensopivuus ja kysy käyttäjältä.

        Returns:
            - True: Käytä Unslothia
            - False: Älä käytä Unslothia
            - None: Anna trainerin päättää (auto-detect)
        """
        # Tarkista yhteensopivuus
        compat = self.trainer.check_unsloth_compatibility(config)

        # Jos Unsloth ei ole asennettu, ohita hiljaa
        if not compat.unsloth_installed:
            return None  # Auto-detect (käytännössä False)

        console.print("\n[bold cyan]━━━ Unsloth-kiihdytys ━━━[/bold cyan]")

        # Näytä yhteensopivuustiedot
        if compat.compatible:
            # Näytä hyödyt
            console.print(f"[green]✓[/green] Unsloth v{compat.unsloth_version} yhteensopiva!")

            # Näytä VRAM-säästö
            if compat.vram_savings_percent > 0:
                console.print(f"  [cyan]VRAM-säästö:[/cyan] ~{compat.vram_savings_percent:.0f}%")
                console.print(f"  [dim]PEFT: ~{compat.estimated_vram_peft:.1f} GB → Unsloth: ~{compat.estimated_vram_unsloth:.1f} GB[/dim]")

            # Näytä varoitukset
            if compat.warnings:
                console.print(f"\n[yellow]Huomioitavaa:[/yellow]")
                for warning in compat.warnings[:3]:  # Max 3 varoitusta
                    console.print(f"  [yellow]![/yellow] {warning}")

            # Näytä suositukset
            if compat.recommendations:
                for rec in compat.recommendations[:2]:
                    console.print(f"  [dim]→ {rec}[/dim]")

            console.print("")

            # Kysy käyttäjältä
            if compat.recommended:
                # Suositeltu - oletus Kyllä
                choices = [
                    questionary.Choice(title="Kyllä, käytä Unslothia (suositeltu)", value=True),
                    questionary.Choice(title="Ei, käytä tavallista PEFT", value=False),
                ]
            else:
                # Yhteensopiva mutta ei optimaalinen
                choices = [
                    questionary.Choice(title="Kyllä, käytä Unslothia", value=True),
                    questionary.Choice(title="Ei, käytä tavallista PEFT (turvallisempi)", value=False),
                ]

            use_unsloth = questionary.select(
                "Käytä Unsloth-kiihdytystä?",
                choices=choices,
                style=custom_style,
            ).ask()

            return use_unsloth

        else:
            # Ei yhteensopiva - näytä syyt
            console.print(f"[yellow]![/yellow] Unsloth asennettu (v{compat.unsloth_version}), mutta ei yhteensopiva:")

            for error in compat.errors[:3]:
                console.print(f"  [red]✗[/red] {error}")

            if compat.recommendations:
                console.print(f"\n[dim]Suositukset:[/dim]")
                for rec in compat.recommendations[:2]:
                    console.print(f"  [dim]→ {rec}[/dim]")

            console.print(f"\n[dim]Käytetään tavallista PEFT-koulutusta.[/dim]")

            return False  # Ei käytetä Unslothia

    def _run_training(self, config: FullConfig):
        """Run training."""
        console.print("\n[bold cyan]Aloitetaan LoRA training...[/bold cyan]")
        console.print("[dim]Tama voi kestaa pitkaanriippuen datasetin koosta.[/dim]\n")

        # Check readiness
        prep = self.trainer.prepare_training(config)
        if not prep["success"]:
            print_error(prep.get("error", "Tuntematon virhe"))
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        if prep["gpu_available"]:
            console.print(f"[green]GPU:[/green] {prep['gpu_name']} ({prep['gpu_memory_gb']:.1f} GB)")
        else:
            print_warning("GPU ei saatavilla - training on hidasta!")

        # Tarkista Unsloth-yhteensopivuus ja kysy käyttäjältä
        use_unsloth = self._check_and_ask_unsloth(config)

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.trainer.train(config, progress_callback=progress_cb, use_unsloth=use_unsloth)

        if result["success"]:
            # Näytä käytetty backend
            backend = result.get("backend", "peft")
            backend_text = "Unsloth" if backend == "unsloth" else "PEFT/transformers"

            console.print(Panel(
                f"[green]Training valmis![/green]\n\n"
                f"[white]Backend:[/white] {backend_text}\n"
                f"[white]Adapter:[/white] {result['adapter_path']}\n"
                f"[white]Aika:[/white] {result['elapsed_formatted']}\n"
                f"[white]Trainable params:[/white] {result['trainable_params']:,} / {result['total_params']:,}",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))
        else:
            print_error(f"Training epäonnistui: {result.get('error', 'Tuntematon virhe')}")
            # Näytä käytetty backend jos virhe
            backend = result.get("backend", "unknown")
            if backend == "unsloth":
                console.print("[dim]Vinkki: Kokeile PEFT-koulutusta valitsemalla 'Ei' Unsloth-kysymyksessä[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _test_adapter_wizard(self):
        """Test adapter wizard."""
        print_mini_banner("Test Adapter")

        # Select base model
        console.print("[bold]1. Valitse base-malli:[/bold]")
        base_model = self._select_model_for_training()
        if not base_model:
            return

        # Select adapter - collect all options
        console.print("\n[bold]2. Valitse adapter:[/bold]")

        choices = []

        # Downloaded LoRA adapters (models/loras/)
        downloaded_loras = self.downloader.list_downloaded_loras()
        if downloaded_loras:
            choices.append(questionary.Separator("-- Ladatut LoRA-adapterit --"))
            for lora in downloaded_loras[:10]:
                base = lora.get('base_model', '')
                if base:
                    base_short = base.split('/')[-1][:20] if '/' in base else base[:20]
                    title = f"[lora] {lora['name'][:35]:<35} ({base_short})"
                else:
                    title = f"[lora] {lora['name'][:35]}"
                choices.append(questionary.Choice(title=title, value=("lora", lora['path'])))

        # Training checkpoints
        checkpoints = self.trainer.list_checkpoints()
        if checkpoints:
            choices.append(questionary.Separator("-- Training Checkpoints --"))
            for cp in checkpoints[:10]:
                choices.append(questionary.Choice(
                    title=f"[cp]  {cp['name']}",
                    value=("checkpoint", cp["path"])
                ))

        if not downloaded_loras and not checkpoints:
            console.print("[yellow]Ei ladattuja LoRA-adaptereita eika checkpointteja.[/yellow]")
            console.print(f"[dim]Lataa LoRA: Model Download -> Download LoRA[/dim]")
            console.print(f"[dim]LoRA-kansio: {get_paths().loras_dir}[/dim]\n")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Back", value=("back", None)))

        result = questionary.select(
            "Valitse adapter:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result[0] == "back":
            return

        adapter_type, adapter_value = result

        if adapter_type == "lora":
            adapter_path = Path(adapter_value)
        elif adapter_type == "checkpoint":
            adapter_path = Path(adapter_value) if isinstance(adapter_value, str) else adapter_value
            # Check for adapter subfolder
            if (adapter_path / "adapter").exists():
                adapter_path = adapter_path / "adapter"
        else:
            return

        # Ask for prompt
        console.print("\n[bold]3. Anna testiprompt:[/bold]")
        prompt = questionary.text(
            "Prompt:",
            default="### Instruction:\nKerro lyhyesti mita tekoaly on.\n\n### Response:\n",
            style=custom_style,
            multiline=True,
        ).ask()

        if not prompt:
            return

        # Test
        console.print("\n[cyan]Testataan adapteria...[/cyan]\n")

        result = self.trainer.test_adapter(base_model, adapter_path, prompt)

        if result["success"]:
            console.print(Panel(
                result["response"],
                title="[bold]Generated Response[/bold]",
                border_style="green"
            ))
        else:
            print_error(f"Testi epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _merge_adapter_wizard(self):
        """Merge adapter into base model."""
        print_mini_banner("Merge Adapter")

        console.print("[cyan]Yhdistaa LoRA-adapterin base-malliin.[/cyan]")
        console.print("[dim]Tuloksena on taysikokoinen malli ilman adapteria.[/dim]\n")

        # Select base model
        console.print("[bold]1. Valitse base-malli:[/bold]")
        base_model = self._select_model_for_training()
        if not base_model:
            return

        # Select adapter - collect all options
        console.print("\n[bold]2. Valitse adapter:[/bold]")

        choices = []

        # Downloaded LoRA adapters (models/loras/)
        downloaded_loras = self.downloader.list_downloaded_loras()
        if downloaded_loras:
            choices.append(questionary.Separator("-- Ladatut LoRA-adapterit --"))
            for lora in downloaded_loras[:10]:
                base = lora.get('base_model', '')
                if base:
                    base_short = base.split('/')[-1][:20] if '/' in base else base[:20]
                    title = f"[lora] {lora['name'][:35]:<35} ({base_short})"
                else:
                    title = f"[lora] {lora['name'][:35]}"
                choices.append(questionary.Choice(title=title, value=("lora", lora['path'])))

        # Training checkpoints
        checkpoints = self.trainer.list_checkpoints()
        if checkpoints:
            choices.append(questionary.Separator("-- Training Checkpoints --"))
            for cp in checkpoints[:10]:
                choices.append(questionary.Choice(
                    title=f"[cp]  {cp['name']}",
                    value=("checkpoint", cp["path"])
                ))

        if not downloaded_loras and not checkpoints:
            console.print("[yellow]Ei ladattuja LoRA-adaptereita eika checkpointteja.[/yellow]")
            console.print(f"[dim]Lataa LoRA: Model Download -> Download LoRA[/dim]")
            console.print(f"[dim]LoRA-kansio: {get_paths().loras_dir}[/dim]\n")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Back", value=("back", None)))

        result = questionary.select(
            "Valitse adapter:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result[0] == "back":
            return

        adapter_type, adapter_value = result

        if adapter_type == "lora":
            adapter_path = Path(adapter_value)
        elif adapter_type == "checkpoint":
            adapter_path = Path(adapter_value) if isinstance(adapter_value, str) else adapter_value
            if (adapter_path / "adapter").exists():
                adapter_path = adapter_path / "adapter"
        else:
            return

        # Validate base model compatibility
        adapter_config_file = adapter_path / "adapter_config.json"
        if adapter_config_file.exists():
            try:
                import json
                with open(adapter_config_file, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)

                expected_base = adapter_config.get('base_model_name_or_path', '')
                if expected_base:
                    # Extract model name for comparison
                    expected_name = expected_base.split('/')[-1].lower() if '/' in expected_base else expected_base.lower()
                    selected_name = base_model.name.lower()

                    # Check compatibility - same architecture family is OK
                    # Llama 3.1 pohjaiset mallit (Poro, etc.) ovat yhteensopivia
                    llama_variants = ['llama', 'poro', 'viking', 'finllama']
                    mistral_variants = ['mistral', 'mixtral']
                    qwen_variants = ['qwen']

                    def get_family(name):
                        name_lower = name.lower()
                        if any(v in name_lower for v in llama_variants):
                            return 'llama'
                        if any(v in name_lower for v in mistral_variants):
                            return 'mistral'
                        if any(v in name_lower for v in qwen_variants):
                            return 'qwen'
                        return name_lower

                    expected_family = get_family(expected_name)
                    selected_family = get_family(selected_name)

                    is_compatible = (
                        expected_name in selected_name or
                        selected_name in expected_name or
                        expected_family == selected_family
                    )

                    if not is_compatible:
                        console.print(Panel(
                            f"[bold red]VAROITUS: Yhteensopivuusongelma![/bold red]\n\n"
                            f"[white]Adapterin alkuperainen base-malli:[/white]\n"
                            f"  [cyan]{expected_base}[/cyan]\n\n"
                            f"[white]Valitsemasi base-malli:[/white]\n"
                            f"  [yellow]{base_model.name}[/yellow]\n\n"
                            f"[dim]LoRA-adapterit toimivat vain alkuperaisen\n"
                            f"base-mallinsa kanssa. Vaara malli tuottaa\n"
                            f"rikkinaisen tuloksen.[/dim]",
                            title="[bold red]Compatibility Warning[/bold red]",
                            border_style="red"
                        ))

                        if not questionary.confirm(
                            "Haluatko silti jatkaa? (EI SUOSITELLA)",
                            default=False,
                            style=custom_style
                        ).ask():
                            return
            except Exception:
                pass  # If we can't read config, continue anyway

        # Output name
        output_name = questionary.text(
            "Tulosmallin nimi:",
            default=f"merged_{base_model.name}_{adapter_path.name}",
            style=custom_style,
        ).ask()

        if not output_name:
            return

        output_path = self.merger.output_dir / output_name

        # Show summary
        console.print(Panel(
            f"[white]Base:[/white] {base_model.name}\n"
            f"[white]Adapter:[/white] {adapter_path}\n"
            f"[white]Output:[/white] {output_path}",
            title="[bold]Merge Configuration[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Run
        console.print("\n[cyan]Yhdistetaan...[/cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.trainer.merge_adapter(base_model, adapter_path, output_path, progress_callback=progress_cb)

        if result["success"]:
            console.print(Panel(
                f"[green]Merge valmis![/green]\n\n"
                f"[white]Output:[/white] {result['output_path']}",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged_lora",
                    source_id=f"{base_model.name}+{adapter_path.name}",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {result.get('error')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()
