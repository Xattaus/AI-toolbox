"""
AI TOOLBOX - Abliteration Commands
==================================

CLI commands for model abliteration: remove refusal behavior.
"""

from pathlib import Path
from typing import Optional

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel

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
from ..abliteration.abliterator import Abliterator, AbliterationConfig
from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader

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


class AbliterationCommands:
    """CLI commands for model abliteration."""

    def __init__(
        self,
        abliterator: Abliterator,
        library: ModelLibrary,
        downloader: ModelDownloader,
    ):
        """
        Initialize abliteration commands.

        Args:
            abliterator: Abliterator instance
            library: Model library instance
            downloader: Model downloader instance
        """
        self.abliterator = abliterator
        self.library = library
        self.downloader = downloader

    def abliteration_menu(self):
        """Abliteration sub-menu."""
        while True:
            print_mini_banner("Abliteration Tool")

            # Check status
            status = self.abliterator.get_status()

            if not status["ready"]:
                missing = ", ".join(status["missing"])
                console.print(f"[yellow]Puuttuvat riippuvuudet: {missing}[/yellow]")
                console.print("[dim]Valitse 'Install Dependencies' asentaaksesi[/dim]\n")

            console.print(f"[dim]Output: {status['output_dir']}[/dim]")
            if status.get("cuda_available"):
                console.print("[green]CUDA available[/green]\n")
            else:
                console.print("[yellow]CPU mode (CUDA not available)[/yellow]\n")

            choices = [
                questionary.Choice(
                    title="Full Abliteration      Poista refusal kokonaan",
                    value="full"
                ),
                questionary.Choice(
                    title="Test Model             Testaa abliteroitua mallia",
                    value="test"
                ),
                questionary.Separator(),
                questionary.Choice(
                    title="Abliteration Info      Mita abliteration on?",
                    value="info"
                ),
            ]

            if not status["ready"]:
                choices.append(questionary.Choice(
                    title="Install Dependencies    Asenna torch & transformers",
                    value="install"
                ))

            choices.extend([
                questionary.Separator(),
                questionary.Choice(title="Back                    Palaa", value="back"),
            ])

            choice = questionary.select(
                "Abliteration Tool:",
                choices=choices,
                style=custom_style,
                qmark=">>",
                pointer=">"
            ).ask()

            if choice is None or choice == "back":
                break
            elif choice == "install":
                self._install_deps()
            elif choice == "info":
                self._show_info()
            elif choice == "full":
                if not status["ready"]:
                    print_warning("Asenna ensin riippuvuudet")
                    questionary.press_any_key_to_continue(style=custom_style).ask()
                else:
                    self._full_abliteration_wizard()
            elif choice == "test":
                self._test_model_wizard()

    def _install_deps(self):
        """Install abliteration dependencies."""
        print_mini_banner("Install Dependencies")

        console.print("[cyan]Asennetaan torch, transformers, safetensors...[/cyan]")
        console.print("[dim]Tama voi kestaa hetken...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        if self.abliterator.install_dependencies(progress_callback=progress_cb):
            print_success("Riippuvuudet asennettu!")
        else:
            print_error("Asennus epäonnistui")
            console.print("\n[dim]Kokeile manuaalisesti:[/dim]")
            console.print("[cyan]pip install torch transformers safetensors accelerate[/cyan]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_info(self):
        """Show information about abliteration."""
        print_mini_banner("Abliteration Info")

        info_text = """
[bold cyan]Mita Abliteration on?[/bold cyan]
------------------------------------------
Abliteration on tekniikka, jolla poistetaan kielimallin
kieltaytymiskayttaytyminen (refusal behavior).

[white]Miten se toimii:[/white]
1. Ajetaan mallin lapi "haitallisia" ja "harmittomia" prompteja
2. Kerataan aktivaatiot keskikerroksista
3. Lasketaan "refusal direction" - suunta joka erottaa haitalliset
4. Projisoidaan tama suunta pois mallin painoista
5. Tulos: Malli joka vastaa ilman sensurointia

[white]Tuetut mallit:[/white]
- Llama 3.1 (vahvin refusal, paras kohde)
- Llama 2/3
- Mistral
- Qwen 2.5
- Muut transformer-pohjaiset

[white]Parametrit:[/white]
- [cyan]Strength[/cyan]: 0.0-2.0, kuinka voimakas poisto
  - 0.5 = osittainen (sailyttaa joitain rajoja)
  - 1.0 = taysi poisto (oletus)
  - 1.5-2.0 = aggressiivinen (voi rikkoa mallin)

- [cyan]Method[/cyan]:
  - mean_diff: Yksinkertainen keskiarvoerotus (nopea)
  - pca: PCA-analyysi (tarkempi, vaatii sklearn)

[dim]------------------------------------------[/dim]
[bold yellow]VAROITUS[/bold yellow]
[yellow]Abliteroitu malli voi tuottaa haitallista sisaltoa.
Kayta vastuullisesti vain tutkimus- ja testaustarkoituksiin.[/yellow]
"""
        console.print(info_text)
        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_model(self, prompt: str = "Valitse malli:") -> Optional[Path]:
        """Select model from library for abliteration."""
        choices = []

        def format_date(date_str: str) -> str:
            """Format ISO date to readable format."""
            if not date_str:
                return "?"
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime("%d.%m.%Y")
            except:
                return date_str[:10] if len(date_str) >= 10 else date_str

        def format_source(source: str) -> str:
            """Format source for display."""
            source_map = {
                "huggingface": "HF",
                "local": "Local",
                "merged": "Merged",
                "abliterated": "Ablit",
            }
            return source_map.get(source, source[:6])

        # Library safetensors models only (needed for abliteration)
        models = self.library.list_models(format_filter="safetensors")

        if models:
            choices.append(questionary.Separator("-- Kirjaston SafeTensors-mallit --"))
            for m in models[:15]:
                size = format_size(m.size_bytes)
                date = format_date(m.added_date)
                src = format_source(m.source)
                # Rivi 1: Nimi ja koko
                # Rivi 2: Lisatiedot
                title = (
                    f"{m.name[:42]:<42} {size:>10}\n"
                    f"      {src:<8} | Lisatty: {date}"
                )
                choices.append(questionary.Choice(
                    title=title,
                    value=("library", m.path)
                ))

        # Downloaded HuggingFace models
        downloaded = self.downloader.list_downloaded()
        if downloaded:
            choices.append(questionary.Separator("-- Ladatut HF-mallit --"))
            for d in downloaded[:10]:
                size = format_size(d['size'])
                name = d['model_id'].split('/')[-1] if '/' in d['model_id'] else d['model_id']
                title = (
                    f"{name[:42]:<42} {size:>10}\n"
                    f"      HF       | HuggingFace"
                )
                choices.append(questionary.Choice(
                    title=title,
                    value=("download", d['path'])
                ))

        if not models and not downloaded:
            print_warning("Ei SafeTensors-malleja kirjastossa tai ladattuna.")
            console.print("[dim]Lataa malli ensin: Model Download[/dim]")
            console.print("[dim]Abliteration vaatii SafeTensors-muotoisen mallin.[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return None

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value=("cancel", None)))

        result = questionary.select(
            prompt,
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result[0] == "cancel":
            return None

        return Path(result[1])

    def _full_abliteration_wizard(self):
        """Full abliteration wizard."""
        print_mini_banner("Full Abliteration")

        # Warning
        console.print(Panel(
            "[bold yellow]VAROITUS[/bold yellow]\n\n"
            "Abliteration poistaa mallin turvaominaisuudet.\n"
            "Abliteroitu malli voi tuottaa:\n"
            "- Haitallista sisaltoa\n"
            "- Vaarallisia ohjeita\n"
            "- Epaeettisia vastauksia\n\n"
            "[dim]Kayta vastuullisesti vain tutkimus- ja testaustarkoituksiin.[/dim]",
            title="[bold red]Turvallisuusvaroitus[/bold red]",
            border_style="red"
        ))

        if not questionary.confirm(
            "Ymmärran riskit ja haluan jatkaa?",
            default=False,
            style=custom_style
        ).ask():
            return

        # =====================================================================
        # 1. SOURCE MODEL
        # =====================================================================
        console.print("\n[bold cyan]1. SOURCE MODEL[/bold cyan]")
        console.print("[dim]   Valitse lahdmalli SafeTensors-muodossa[/dim]\n")

        model_path = self._select_model("Source model:")
        if not model_path:
            return

        info = self.abliterator.get_model_info(str(model_path))
        console.print(f"\n   [green]+[/green] {info.get('name', model_path.name)}")
        if info.get("architecture"):
            console.print(f"     [dim]Architecture: {info['architecture']}[/dim]")
            console.print(f"     [dim]Layers: {info.get('num_layers', '?')}, Hidden: {info.get('hidden_size', '?')}[/dim]")
        if info.get("is_llama31"):
            console.print("     [yellow]Detected: Llama 3.1 (vahva refusal)[/yellow]")

        # =====================================================================
        # 2. STRENGTH
        # =====================================================================
        console.print("\n[bold cyan]2. STRENGTH[/bold cyan]")
        console.print("[dim]   Abliteroinnin voimakkuus (0.0-2.0)[/dim]")
        console.print("[dim]   • 0.5 = osittainen[/dim]")
        console.print("[dim]   • 1.0 = taysi (suositeltu)[/dim]")
        console.print("[dim]   • 1.5+ = aggressiivinen[/dim]\n")

        strength_str = questionary.text(
            "Strength (default 1.0):",
            default="1.0",
            style=custom_style,
        ).ask()

        try:
            strength = float(strength_str)
            strength = max(0.0, min(2.0, strength))
        except ValueError:
            strength = 1.0

        # =====================================================================
        # 3. METHOD
        # =====================================================================
        console.print("\n[bold cyan]3. METHOD[/bold cyan]")
        console.print("[dim]   Refusal direction -laskentamenetelma[/dim]\n")

        method_choice = questionary.select(
            "Method:",
            choices=[
                questionary.Choice(title="mean_diff  (nopea, suositeltu)", value="mean_diff"),
                questionary.Choice(title="pca        (tarkempi, vaatii sklearn)", value="pca"),
            ],
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if method_choice is None:
            return

        # =====================================================================
        # 4. BATCH SIZE
        # =====================================================================
        console.print("\n[bold cyan]4. BATCH SIZE[/bold cyan]")
        console.print("[dim]   Promptien kasittely kerralla[/dim]")
        console.print("[dim]   • Suurempi = nopeampi, enemman VRAM[/dim]")
        console.print("[dim]   • Suositus: 4-8 (GPU), 1-2 (CPU)[/dim]\n")

        batch_size_str = questionary.text(
            "Batch size (default 8):",
            default="8",
            style=custom_style,
        ).ask()

        try:
            batch_size = int(batch_size_str)
            batch_size = max(1, min(64, batch_size))
        except ValueError:
            batch_size = 8

        # =====================================================================
        # 5. PROMPT SOURCE
        # =====================================================================
        console.print("\n[bold cyan]5. PROMPT SOURCE[/bold cyan]")
        console.print("[dim]   Mista harmful/harmless promptit ladataan[/dim]\n")

        # Check for saved prompt files in datasets/abliteration/
        from ..core.paths import get_paths
        paths = get_paths()
        abliter_dataset_dir = paths.datasets_dir / "abliteration"
        saved_harmful = abliter_dataset_dir / "harmful_prompts.txt"
        saved_harmless = abliter_dataset_dir / "harmless_prompts.txt"
        has_saved_prompts = saved_harmful.exists() and saved_harmless.exists()

        # Build choices
        prompt_choices = [
            questionary.Choice(
                title="Built-in (77 EN + Llama 3.1)",
                value="builtin"
            ),
        ]

        if has_saved_prompts:
            # Count prompts in saved files
            try:
                from ..abliteration.prompts import load_prompts_from_file
                h_count = len(load_prompts_from_file(str(saved_harmful)))
                hl_count = len(load_prompts_from_file(str(saved_harmless)))
                prompt_choices.insert(0, questionary.Choice(
                    title=f"Saved datasets ({h_count} + {hl_count} prompts)",
                    value="saved"
                ))
            except Exception:
                pass  # Skip if loading fails

        prompt_choices.append(questionary.Choice(
            title="Custom files (select path)",
            value="custom"
        ))

        prompt_source = questionary.select(
            "Source:",
            choices=prompt_choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if prompt_source is None:
            return

        harmful_file = None
        harmless_file = None

        if prompt_source == "saved":
            # Use saved dataset files
            harmful_file = str(saved_harmful)
            harmless_file = str(saved_harmless)
            from ..abliteration.prompts import load_prompts_from_file
            h_prompts = load_prompts_from_file(harmful_file)
            hl_prompts = load_prompts_from_file(harmless_file)
            console.print(f"\n[green]+[/green] Kaytetaan tallennettuja datasetteja")
            console.print(f"[green]+[/green] {len(h_prompts)} harmful, {len(hl_prompts)} harmless prompts")

        elif prompt_source == "custom":
            console.print("\n[cyan]Valitse prompt-tiedostot:[/cyan]")
            console.print("[dim]Formaatti: yksi prompt per rivi, UTF-8 encoding[/dim]\n")

            # Harmful prompts file
            harmful_path = questionary.path(
                "Harmful prompts file:",
                style=custom_style,
            ).ask()

            if not harmful_path:
                console.print("[yellow]Peruutettu[/yellow]")
                return

            harmful_file = harmful_path

            # Check file exists
            if not Path(harmful_file).exists():
                console.print(f"[red]Tiedostoa ei loydy: {harmful_file}[/red]")
                return

            # Harmless prompts file
            harmless_path = questionary.path(
                "Harmless prompts file:",
                style=custom_style,
            ).ask()

            if not harmless_path:
                console.print("[yellow]Peruutettu[/yellow]")
                return

            harmless_file = harmless_path

            if not Path(harmless_file).exists():
                console.print(f"[red]Tiedostoa ei loydy: {harmless_file}[/red]")
                return

            # Load and show count
            try:
                from ..abliteration.prompts import load_prompts_from_file
                h_prompts = load_prompts_from_file(harmful_file)
                hl_prompts = load_prompts_from_file(harmless_file)
                console.print(f"\n[green]+[/green] Ladattu {len(h_prompts)} harmful prompts")
                console.print(f"[green]+[/green] Ladattu {len(hl_prompts)} harmless prompts")
            except Exception as e:
                console.print(f"[red]Virhe tiedostoja ladattaessa: {e}[/red]")
                return

        # =====================================================================
        # 6. EXTRA TARGETS (experimental)
        # =====================================================================
        console.print("\n[bold cyan]6. EXTRA TARGETS (experimental)[/bold cyan]")
        console.print("[dim]   Valinnaiset lisakohteet abliteroinnille[/dim]")
        console.print("[dim]   • embed_tokens = embedding layer[/dim]")
        console.print("[dim]   • lm_head = output layer[/dim]\n")

        abliterate_embeddings = questionary.confirm(
            "Abliterate embed_tokens?",
            default=False,
            style=custom_style,
        ).ask()

        abliterate_lm_head = questionary.confirm(
            "Abliterate lm_head?",
            default=False,
            style=custom_style,
        ).ask()

        # =====================================================================
        # 7. OUTPUT NAME
        # =====================================================================
        console.print("\n[bold cyan]7. OUTPUT NAME[/bold cyan]")
        console.print("[dim]   Tulostiedoston nimi[/dim]\n")

        default_name = f"{info.get('name', 'model')[:30]}-abliterated"
        output_name = questionary.text(
            "Name:",
            default=default_name,
            style=custom_style,
        ).ask()

        if not output_name:
            return

        # Estimate requirements
        reqs = self.abliterator.estimate_requirements(str(model_path))

        # Determine prompt source text for summary
        if harmful_file:
            prompt_info = f"Custom: {Path(harmful_file).name} + {Path(harmless_file).name}"
        else:
            prompt_info = "Built-in (77 EN + Llama 3.1)"

        # Build extra targets string
        extra_targets = []
        if abliterate_embeddings:
            extra_targets.append("embed_tokens")
        if abliterate_lm_head:
            extra_targets.append("lm_head")
        extra_targets_str = ", ".join(extra_targets) if extra_targets else "None"

        # Show summary
        console.print("\n")
        console.print(Panel(
            f"[bold]Source Model:[/bold]   {model_path.name}\n"
            f"[bold]Architecture:[/bold]   {info.get('architecture', 'Unknown')}\n"
            f"[bold]Strength:[/bold]       {strength:.1f}\n"
            f"[bold]Method:[/bold]         {method_choice}\n"
            f"[bold]Batch Size:[/bold]     {batch_size}\n"
            f"[bold]Prompts:[/bold]        {prompt_info}\n"
            f"[bold]Extra Targets:[/bold]  {extra_targets_str}\n"
            f"[bold]Output:[/bold]         {output_name}\n\n"
            f"[dim]Arvioitu RAM:[/dim]    {reqs.get('estimated_ram_gb', '?')} GB\n"
            f"[dim]Arvioitu VRAM:[/dim]   {reqs.get('estimated_vram_gb', '?')} GB",
            title="[bold cyan]Configuration Summary[/bold cyan]",
            border_style="cyan"
        ))

        if not questionary.confirm("Aloitetaanko abliteration?", default=True, style=custom_style).ask():
            return

        # Create config
        config = AbliterationConfig(
            model_path=str(model_path),
            output_name=output_name,
            strength=strength,
            method=method_choice,
            include_llama31_prompts=info.get("is_llama31", False),
            harmful_prompts_file=harmful_file,
            harmless_prompts_file=harmless_file,
            num_harmful=0,  # 0 = use all prompts from file
            num_harmless=0,
            batch_size=batch_size,
            abliterate_embeddings=abliterate_embeddings or False,
            abliterate_lm_head=abliterate_lm_head or False,
        )

        # Run abliteration
        console.print("\n[bold cyan]Suoritetaan abliteration...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja (malli ladataan muistiin)...[/dim]\n")

        from rich.progress import Progress, SpinnerColumn, TextColumn

        current_phase = ["Starting..."]

        def progress_cb(msg, prog):
            current_phase[0] = msg

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(description="Starting...", total=None)

            import threading
            result_holder = [None]

            def run_abliteration():
                def update_progress(msg, prog):
                    progress.update(task, description=msg)
                result_holder[0] = self.abliterator.full_abliteration(config, update_progress)

            thread = threading.Thread(target=run_abliteration)
            thread.start()
            thread.join()

            result = result_holder[0]

        if result and result.success:
            console.print(Panel(
                f"[green]Abliteration valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result.output_path}\n"
                f"[white]Muokatut kerrokset:[/white] {len(result.modified_layers)}\n"
                f"[white]Muokatut painot:[/white] {result.modified_weights}\n"
                f"[white]Aika:[/white] {result.elapsed_seconds:.1f}s",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result.output_path,
                    source="abliterated",
                    source_id=str(model_path),
                    abliteration_info={
                        "method": result.method_used,
                        "strength": result.strength_applied,
                        "source_model": str(model_path),
                        "modified_layers": result.modified_layers,
                    }
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")

            # Ask if want to test
            if questionary.confirm("Haluatko testata mallia?", default=True, style=custom_style).ask():
                self._test_specific_model(result.output_path)
        else:
            error_msg = result.error if result else "Unknown error"
            print_error(f"Abliteration epäonnistui: {error_msg}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _test_model_wizard(self):
        """Test an abliterated model via Ollama."""
        print_mini_banner("Test Abliterated Model (Ollama)")

        try:
            from ..integrations.ollama import OllamaManager
            from ..abliteration.testing import AbliterationTester
            from ..abliteration.prompts import get_category_list, TEST_PROMPT_CATEGORIES
        except ImportError as e:
            print_error(f"Tarvittavat moduulit puuttuvat: {e}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        manager = OllamaManager()
        models = manager.list_models()

        if not models:
            print_warning("Ei Ollama-malleja saatavilla.")
            console.print("[dim]Konvertoi ja kvantisoi abliteroitu malli ensin, sitten luo Ollama-malli.[/dim]")
            console.print("[dim]Workflow: Abliterate -> Convert -> Quantize -> Create Ollama model[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Step 1: Model selection
        console.print("\n[bold]Vaihe 1: Valitse testattava Ollama-malli[/bold]\n")
        choices = [
            questionary.Choice(f"{m.name} ({m.size})", value=m.name)
            for m in models
        ]
        choices.append(questionary.Separator())
        choices.append(questionary.Choice("<-  Peruuta", value=None))

        selected = questionary.select(
            "Valitse malli:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if selected is None:
            return

        # Step 2: Language selection
        console.print("\n[bold]Vaihe 2: Valitse testikieli[/bold]\n")
        lang = questionary.select(
            "Valitse kieli:",
            choices=[
                questionary.Choice("Suomi", value="fi"),
                questionary.Choice("English", value="en"),
            ],
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if lang is None:
            return

        # Step 3: Number of tests
        console.print("\n[bold]Vaihe 3: Testien maara[/bold]\n")
        num_tests = questionary.select(
            "Kuinka monta testia ajetaan?",
            choices=[
                questionary.Choice("5 testia (nopea)", value=5),
                questionary.Choice("10 testia (suositeltu)", value=10),
                questionary.Choice("15 testia (kattava)", value=15),
                questionary.Choice("20 testia (perusteellinen)", value=20),
            ],
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if num_tests is None:
            return

        # Step 4: Category selection
        console.print("\n[bold]Vaihe 4: Valitse kategoriat[/bold]\n")
        categories = get_category_list(lang)

        cat_mode = questionary.select(
            "Mitka kategoriat testataan?",
            choices=[
                questionary.Choice("Kaikki kategoriat (satunnainen valikoima)", value="all"),
                questionary.Choice("Valitse kategoriat itse", value="select"),
            ],
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if cat_mode is None:
            return

        selected_categories = None
        if cat_mode == "select":
            cat_choices = [
                questionary.Choice(
                    f"{name} ({len(TEST_PROMPT_CATEGORIES[key][f'prompts_{lang}'])} promptia)",
                    value=key
                )
                for key, name in categories
            ]
            selected_categories = questionary.checkbox(
                "Valitse kategoriat (valilyonti valitsee):",
                choices=cat_choices,
                style=custom_style,
            ).ask()

            if not selected_categories:
                console.print("[yellow]Ei valittuja kategorioita, kaytetaan kaikkia.[/yellow]")
                selected_categories = None

        # Show test configuration
        cat_display = "Kaikki" if selected_categories is None else ", ".join(selected_categories)
        console.print()
        console.print(Panel(
            f"[bold]Malli:[/bold]      {selected}\n"
            f"[bold]Kieli:[/bold]      {'Suomi' if lang == 'fi' else 'English'}\n"
            f"[bold]Testeja:[/bold]    {num_tests}\n"
            f"[bold]Kategoriat:[/bold] {cat_display}",
            title="[bold cyan]Testin asetukset[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        # Run the test
        tester = AbliterationTester(manager)

        from rich.progress import Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Ajetaan testeja...", total=num_tests)

            def update_progress(current, total, msg):
                progress.update(task, completed=current, description=f"[{current}/{total}] {msg}")

            report = tester.run_test(
                model_name=selected,
                num_prompts=num_tests,
                language=lang,
                categories=selected_categories,
                progress_callback=update_progress,
            )

        # Display results
        self._display_test_results(report)

        # Ask to save
        if questionary.confirm("Tallenna raportti tiedostoon?", default=True, style=custom_style).ask():
            filepath = tester.save_report(report)
            print_success(f"Raportti tallennettu: {filepath}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _display_test_results(self, report):
        """Display test results in a formatted table."""
        from rich.table import Table
        from rich import box

        # Summary panel
        status_color = "green" if report.refusal_rate < 0.3 else "yellow" if report.refusal_rate < 0.7 else "red"
        console.print(Panel(
            f"[bold]Malli:[/bold] {report.model_name}\n"
            f"[bold]Kieli:[/bold] {report.language}\n"
            f"[bold]Testeja:[/bold] {report.total_tests}\n"
            f"[bold]Vastasi:[/bold] [{status_color}]{report.answered_count}[/{status_color}] "
            f"({100*(1-report.refusal_rate):.0f}%)\n"
            f"[bold]Kieltaytyi:[/bold] {report.refused_count} ({100*report.refusal_rate:.0f}%)\n"
            f"[bold]Virheita:[/bold] {report.error_count}\n"
            f"[bold]Kesto:[/bold] {report.duration_seconds:.1f}s",
            title="[bold]Tulokset[/bold]",
            border_style=status_color
        ))

        # Category breakdown
        if report.category_stats:
            cat_table = Table(
                title="Kategorioittain",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )
            cat_table.add_column("Kategoria", style="white")
            cat_table.add_column("Testeja", justify="right")
            cat_table.add_column("Vastasi", justify="right", style="green")
            cat_table.add_column("Kieltaytyi", justify="right", style="red")
            cat_table.add_column("Vastaus-%", justify="right")

            for cs in report.category_stats:
                answer_rate = 100 * (1 - cs.refusal_rate)
                rate_color = "green" if answer_rate > 70 else "yellow" if answer_rate > 30 else "red"
                cat_table.add_row(
                    cs.category_name,
                    str(cs.total),
                    str(cs.answered),
                    str(cs.refused),
                    f"[{rate_color}]{answer_rate:.0f}%[/{rate_color}]"
                )

            console.print(cat_table)

        # Individual results table
        results_table = Table(
            title="Yksittaiset tulokset",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            expand=True
        )
        results_table.add_column("#", style="dim", width=3, justify="right")
        results_table.add_column("Kategoria", style="cyan", width=20)
        results_table.add_column("Tulos", style="bold", width=12, justify="center")
        results_table.add_column("Prompti", style="white", max_width=30)
        results_table.add_column("Vastaus", style="dim", max_width=40)

        for i, r in enumerate(report.results, 1):
            prompt_preview = r.prompt[:27] + "..." if len(r.prompt) > 30 else r.prompt
            response_preview = r.response[:37] + "..." if len(r.response) > 40 else r.response
            response_preview = response_preview.replace('\n', ' ')

            if r.error:
                status = "[yellow]VIRHE[/yellow]"
            elif r.refused:
                status = "[red]KIELTAYTYI[/red]"
            else:
                status = "[green]VASTASI[/green]"

            results_table.add_row(
                str(i),
                r.category_name[:20],
                status,
                prompt_preview,
                response_preview
            )

        console.print(results_table)

    def _test_specific_model(self, model_path: str):
        """Legacy: redirect to wizard."""
        console.print("[yellow]Huom: Testaus toimii nyt Ollama-mallien kautta.[/yellow]")
        console.print("[dim]Konvertoi malli GGUF:ksi ja luo Ollama-malli ensin.[/dim]")
        questionary.press_any_key_to_continue(style=custom_style).ask()
