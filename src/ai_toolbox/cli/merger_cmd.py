"""
AI TOOLBOX - Merger Commands
============================

CLI commands for model merging: SLERP, TIES, Frankenmerge.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

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
    print_section_header,
    print_key_value,
    print_divider,
    create_summary_panel,
    create_result_panel,
    select_model_from_table,
    MENU_STYLE,
)
from ..core.paths import get_paths
from ..merging.merger import ModelMerger
from ..merging.mergekit_wrapper import MergekitWrapper, MergekitMethod, MergekitConfig
from ..merging.presets import (
    PRESETS, get_preset, list_presets, get_recommended_preset,
    get_presets_by_category, PresetCategory,
)
from ..merging.config_manager import MergeConfigManager
from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader

# Use unified menu style from ui module
custom_style = MENU_STYLE


class MergerCommands:
    """CLI commands for model merging."""

    def __init__(
        self,
        merger: ModelMerger,
        library: ModelLibrary,
        downloader: ModelDownloader,
    ):
        """
        Initialize merger commands.

        Args:
            merger: ModelMerger instance
            library: Model library instance
            downloader: Model downloader instance
        """
        self.merger = merger
        self.library = library
        self.downloader = downloader

        # Initialize mergekit components
        paths = get_paths()
        self.mergekit = MergekitWrapper(output_dir=paths.merged_dir)
        self.config_manager = MergeConfigManager(config_dir=paths.root / "models" / "merge_configs")

    def model_merger_menu(self):
        """Model Merger sub-menu - ohjaa Mergekit Wizard päävalikkoon."""
        # Kaikki merge-työkalut ovat nyt Mergekit Wizardin alla
        self._mergekit_main_menu()

    def _install_merger_deps(self):
        """Install Model Merger dependencies."""
        print_mini_banner("Install Dependencies")

        console.print("[cyan]Asennetaan torch ja safetensors...[/cyan]")
        console.print("[dim]Tama voi kestaa hetken...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        if self.merger.install_dependencies(progress_callback=progress_cb):
            print_success("Riippuvuudet asennettu!")
        else:
            print_error("Asennus epäonnistui")
            console.print("\n[dim]Kokeile manuaalisesti:[/dim]")
            console.print("[cyan]pip install torch safetensors[/cyan]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_merge_info(self):
        """Show information about merge methods."""
        print_mini_banner("Merge Methods")

        info_text = """
[bold cyan]SLERP (Spherical Linear Interpolation)[/bold cyan]
-------------------------------------------
Yhdistaa kaksi mallia interpoloimalla painot pallopinnalla.
Parempi kuin lineaarinen interpolointi, koska sailyttaa
painojen normit.

[white]Kaytto:[/white] Kahden samankaltaisen mallin yhdistaminen
[white]Parametri:[/white] Ratio (0.0 = malli 1, 1.0 = malli 2)
[white]Esim:[/white] Yhdista base-malli ja finetuned-versio


[bold cyan]TIES (Trim, Elect Sign, Disjoint Merge)[/bold cyan]
-------------------------------------------
Edistynyt menetelma useamman mallin yhdistamiseen.
Toimii laskemalla delta-painot base-mallista ja
yhdistamalla ne alykkasti.

[white]Vaiheet:[/white]
  1. TRIM: Poistaa pienet muutokset (kohina)
  2. ELECT: Valitsee etumerkin enemmistoaanestuksella
  3. MERGE: Yhdistaa deltat base-malliin

[white]Kaytto:[/white] 2+ mallin yhdistaminen, sailyttaa parhaat osat
[white]Parametri:[/white] Density (0.0-1.0, kuinka paljon deltoja sailytetaan)


[bold cyan]Frankenmerge (Layer Swapping)[/bold cyan]
-------------------------------------------
Valitsee kokonaisia kerroksia eri malleista.
Kokeellinen menetelma, voi tuottaa yllattavia tuloksia.

[white]Kaytto:[/white] Yhdistele mallien vahvuuksia kerroksittain
[white]Esim:[/white] Malli A:n kerrokset 0-15, Malli B:n 16-31

[dim]-------------------------------------------[/dim]
[dim]Huom: Kaikki metodit vaativat yhteensopivat mallit
(sama arkkitehtuuri, hidden_size, kerrosmaara)[/dim]
"""
        console.print(info_text)
        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _select_model_for_merge(self, prompt: str = "Valitse malli:") -> Optional[Path]:
        """Select model from library for merging using table-based selection."""
        # Get safetensors and pytorch models (mergeable formats)
        models = self.library.list_models(format_filter="safetensors")
        models.extend(self.library.list_models(format_filter="pytorch"))

        if not models:
            print_warning("Ei malleja kirjastossa.")
            console.print("[dim]Lataa malli ensin: Model Download[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return None

        # Use table-based selection
        selected = select_model_from_table(
            models=models[:20],  # Max 20
            title="Model Merger",
            subtitle=prompt,
            show_size=True,
            show_quant=False,  # Not relevant for merge
            show_format=True,
        )

        if selected is None:
            return None

        return Path(selected.path)

    def _slerp_merge_wizard(self):
        """SLERP merge wizard."""
        print_mini_banner("SLERP Merge")

        console.print("[cyan]SLERP yhdistaa kaksi mallia interpoloimalla.[/cyan]")
        console.print("[dim]Valitse kaksi yhteensopivaa mallia.[/dim]\n")

        # Select model 1
        console.print("[bold]Malli 1:[/bold]")
        model1 = self._select_model_for_merge("Valitse ensimmainen malli:")
        if not model1:
            return

        # Show model info
        info1 = self.merger.get_model_info(model1)
        if info1.get("architecture"):
            console.print(f"  [green]+[/green] {info1.get('name')} ({info1.get('architecture')})")
            console.print(f"    [dim]Kerroksia: {info1.get('num_layers')}, Hidden: {info1.get('hidden_size')}[/dim]\n")

        # Select model 2
        console.print("[bold]Malli 2:[/bold]")
        model2 = self._select_model_for_merge("Valitse toinen malli:")
        if not model2:
            return

        info2 = self.merger.get_model_info(model2)
        if info2.get("architecture"):
            console.print(f"  [green]+[/green] {info2.get('name')} ({info2.get('architecture')})")
            console.print(f"    [dim]Kerroksia: {info2.get('num_layers')}, Hidden: {info2.get('hidden_size')}[/dim]\n")

        # Check compatibility
        compatible, msg, _ = self.merger.validate_models_compatible([model1, model2])
        if not compatible:
            print_error(f"Mallit eivat ole yhteensopivia: {msg}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        print_success("Mallit ovat yhteensopivia!")

        # Select ratio
        ratio_str = questionary.text(
            "SLERP ratio (0.0-1.0, oletus 0.5):",
            default="0.5",
            style=custom_style,
        ).ask()

        try:
            ratio = float(ratio_str)
            if ratio < 0.0 or ratio > 1.0:
                print_warning(f"Ratio {ratio:.2f} rajattu välille 0.0-1.0")
                ratio = max(0.0, min(1.0, ratio))
        except ValueError:
            print_warning("Virheellinen ratio, käytetään oletusta: 0.5")
            ratio = 0.5

        console.print(f"\n[dim]Ratio {ratio:.2f}: {100-ratio*100:.0f}% malli 1, {ratio*100:.0f}% malli 2[/dim]")

        # Output name
        default_name = f"slerp_{info1.get('name', 'model1')[:15]}_{info2.get('name', 'model2')[:15]}"
        output_name = questionary.text(
            "Tulosteen nimi:",
            default=default_name,
            style=custom_style,
        ).ask()

        # Show summary
        reqs = self.merger.estimate_merge_requirements([model1, model2])
        console.print(Panel(
            f"[white]Malli 1:[/white] {model1.name}\n"
            f"[white]Malli 2:[/white] {model2.name}\n"
            f"[white]Ratio:[/white] {ratio:.2f}\n"
            f"[white]Tuloste:[/white] {output_name}\n\n"
            f"[yellow]Arvioitu RAM-tarve:[/yellow] {reqs['estimated_ram_gb']:.1f} GB",
            title="[bold]SLERP Merge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Run merge
        console.print("\n[bold cyan]Suoritetaan SLERP merge...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.merger.merge_slerp(
            model1_path=model1,
            model2_path=model2,
            ratio=ratio,
            output_name=output_name,
            progress_callback=progress_cb,
        )

        if result["success"]:
            console.print(Panel(
                f"[green]Merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result['output_path']}\n"
                f"[white]Koko:[/white] {result['file_size_gb']:.2f} GB",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged",
                    source_id=f"slerp:{model1.name}+{model2.name}",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {result.get('error', 'Tuntematon virhe')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _ties_merge_wizard(self):
        """TIES merge wizard."""
        print_mini_banner("TIES Merge")

        console.print("[cyan]TIES yhdistaa useamman mallin alykkasti.[/cyan]")
        console.print("[dim]Tarvitset base-mallin ja 2+ yhdistettavaa mallia.[/dim]\n")

        # Select base model
        console.print("[bold]Base Model (alkuperainen pretrained):[/bold]")
        base_model = self._select_model_for_merge("Valitse base model:")
        if not base_model:
            return

        base_info = self.merger.get_model_info(base_model)
        console.print(f"  [green]+[/green] {base_info.get('name')}\n")

        # Select models to merge
        models = []
        console.print("[bold]Mallit yhdistettavaksi (vahintaan 2):[/bold]")

        while True:
            model = self._select_model_for_merge(f"Valitse malli {len(models)+1} (tai peruuta lopettaaksesi):")
            if not model:
                break

            # Check compatibility with base model
            compatible, msg, _ = self.merger.validate_models_compatible([base_model, model])
            if not compatible:
                print_warning(f"Ei yhteensopiva: {msg}")
                continue

            models.append(model)
            console.print(f"  [green]+[/green] Lisatty: {model.name}")

            if len(models) >= 2:
                if not questionary.confirm("Lisaa toinen malli?", default=False, style=custom_style).ask():
                    break

        if len(models) < 2:
            print_warning("TIES vaatii vahintaan 2 mallia")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Select density
        density_str = questionary.text(
            "TIES density (0.0-1.0, oletus 0.5):",
            default="0.5",
            style=custom_style,
        ).ask()

        try:
            density = float(density_str)
            if density < 0.0 or density > 1.0:
                print_warning(f"Density {density:.2f} rajattu välille 0.0-1.0")
                density = max(0.0, min(1.0, density))
        except ValueError:
            print_warning("Virheellinen density, käytetään oletusta: 0.5")
            density = 0.5

        # Output name
        output_name = questionary.text(
            "Tulosteen nimi:",
            default=f"ties_merge_{len(models)}models",
            style=custom_style,
        ).ask()

        # Show summary
        all_models = [base_model] + models
        reqs = self.merger.estimate_merge_requirements(all_models)

        model_list = "\n".join([f"  - {m.name}" for m in models])
        console.print(Panel(
            f"[white]Base:[/white] {base_model.name}\n"
            f"[white]Mallit:[/white]\n{model_list}\n"
            f"[white]Density:[/white] {density:.2f}\n\n"
            f"[yellow]Arvioitu RAM-tarve:[/yellow] {reqs['estimated_ram_gb']:.1f} GB",
            title="[bold]TIES Merge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Run merge
        console.print("\n[bold cyan]Suoritetaan TIES merge...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.merger.merge_ties(
            models=models,
            base_model=base_model,
            density=density,
            output_name=output_name,
            progress_callback=progress_cb,
        )

        if result["success"]:
            console.print(Panel(
                f"[green]Merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result['output_path']}\n"
                f"[white]Koko:[/white] {result['file_size_gb']:.2f} GB",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged",
                    source_id=f"ties:{len(models)}models",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {result.get('error', 'Tuntematon virhe')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _frankenmerge_wizard(self):
        """Frankenmerge wizard."""
        print_mini_banner("Frankenmerge")

        console.print("[cyan]Frankenmerge valitsee kerroksia eri malleista.[/cyan]")
        console.print("[dim]Maarita jokaiselle mallille kerrosalue.[/dim]\n")

        # Collect models and their layer ranges
        layer_config: Dict[Path, Tuple[int, int]] = {}
        total_layers = None

        while True:
            # Select model
            model = self._select_model_for_merge(f"Valitse malli (tai peruuta lopettaaksesi):")
            if not model:
                break

            info = self.merger.get_model_info(model)
            num_layers = info.get("num_layers", 32)

            if total_layers is None:
                total_layers = num_layers
            elif num_layers != total_layers:
                print_warning(f"Eri kerrosmaara ({num_layers} vs {total_layers})")
                continue

            console.print(f"  [green]+[/green] {info.get('name')} ({num_layers} kerrosta)")

            # Ask layer range
            start_str = questionary.text(
                f"Aloituskerros (0-{num_layers-1}):",
                default="0",
                style=custom_style,
            ).ask()

            end_str = questionary.text(
                f"Lopetuskerros (0-{num_layers-1}):",
                default=str(num_layers-1),
                style=custom_style,
            ).ask()

            try:
                start = int(start_str)
                end = int(end_str)
                start = max(0, min(num_layers-1, start))
                end = max(0, min(num_layers-1, end))
                # Varmista että start <= end
                if start > end:
                    print_warning(f"Aloituskerros ({start}) > lopetuskerros ({end}), vaihdetaan järjestys")
                    start, end = end, start
            except ValueError:
                print_warning("Virheellinen kerrosarvo, käytetään oletuksia")
                start, end = 0, num_layers - 1

            layer_config[model] = (start, end)
            console.print(f"    Kerrokset {start}-{end}\n")

            if len(layer_config) >= 2:
                if not questionary.confirm("Lisaa toinen malli?", default=False, style=custom_style).ask():
                    break

        if len(layer_config) < 2:
            print_warning("Frankenmerge vaatii vahintaan 2 mallia")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Output name
        output_name = questionary.text(
            "Tulosteen nimi:",
            default="frankenmerge_custom",
            style=custom_style,
        ).ask()

        # Show summary
        config_text = ""
        for model, (start, end) in layer_config.items():
            config_text += f"  {model.name}: kerrokset {start}-{end}\n"

        console.print(Panel(
            f"[white]Kerrosten lahteet:[/white]\n{config_text}\n"
            f"[white]Tuloste:[/white] {output_name}",
            title="[bold]Frankenmerge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Run merge
        console.print("\n[bold cyan]Suoritetaan Frankenmerge...[/bold cyan]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.merger.merge_frankenmerge(
            models=layer_config,
            output_name=output_name,
            progress_callback=progress_cb,
        )

        if result["success"]:
            console.print(Panel(
                f"[green]Merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result['output_path']}\n"
                f"[white]Koko:[/white] {result['file_size_gb']:.2f} GB",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged",
                    source_id="frankenmerge",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {result.get('error', 'Tuntematon virhe')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _lora_merge_wizard(self):
        """LoRA Merge wizard - yhdista LoRA-adapter base-malliin."""
        print_mini_banner("LoRA Merge")

        console.print("[cyan]Yhdistaa LoRA-adapterin base-malliin.[/cyan]")
        console.print("[dim]Tuloksena on taysikokoinen malli ilman adapteria.[/dim]\n")

        # 1. Valitse base-malli
        console.print("[bold]1. Valitse base-malli:[/bold]")
        base_model = self._select_model_for_merge("Valitse base-malli:")
        if not base_model:
            return

        base_info = self.merger.get_model_info(base_model)
        console.print(f"  [green]+[/green] {base_info.get('name')}\n")

        # 2. Valitse LoRA adapter
        console.print("[bold]2. Valitse LoRA-adapter:[/bold]")

        choices = []

        # Ladatut LoRA-adapterit
        downloaded_loras = self.downloader.list_downloaded_loras()
        if downloaded_loras:
            choices.append(questionary.Separator("-- Ladatut LoRA-adapterit --"))
            for lora in downloaded_loras[:15]:
                base = lora.get('base_model', '')
                size = format_size(lora.get('size', 0)) if lora.get('size') else "?"
                base_short = base.split('/')[-1][:25] if base and '/' in base else (base[:25] if base else "Tuntematon")
                # Rivi 1: Nimi ja koko
                # Rivi 2: Base model
                title = (
                    f"{lora['name'][:42]:<42} {size:>10}\n"
                    f"      Base: {base_short}"
                )
                choices.append(questionary.Choice(title=title, value=("lora", lora['path'])))

        if not downloaded_loras:
            console.print("[yellow]Ei ladattuja LoRA-adaptereita.[/yellow]")
            console.print("[dim]Lataa LoRA: Model Download -> Download LoRA[/dim]\n")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value=("back", None)))

        result = questionary.select(
            "Valitse adapter:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if result is None or result[0] == "back":
            return

        adapter_path = Path(result[1])

        # 3. Tarkista yhteensopivuus
        adapter_config_file = adapter_path / "adapter_config.json"
        expected_base = None
        if adapter_config_file.exists():
            try:
                import json
                with open(adapter_config_file, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                expected_base = adapter_config.get('base_model_name_or_path', '')

                if expected_base:
                    # Tarkista yhteensopivuus
                    llama_variants = ['llama', 'poro', 'viking', 'finllama']
                    def get_family(name):
                        name_lower = name.lower()
                        if any(v in name_lower for v in llama_variants):
                            return 'llama'
                        return name_lower

                    expected_family = get_family(expected_base)
                    selected_family = get_family(base_model.name)

                    if expected_family != selected_family:
                        console.print(Panel(
                            f"[bold yellow]VAROITUS: Eri arkkitehtuuriperhe[/bold yellow]\n\n"
                            f"[white]Adapterin base:[/white] {expected_base}\n"
                            f"[white]Valittu malli:[/white] {base_model.name}\n\n"
                            f"[dim]LoRA toimii parhaiten alkuperaisella base-mallilla.[/dim]",
                            title="[yellow]Compatibility[/yellow]",
                            border_style="yellow"
                        ))
                        if not questionary.confirm("Jatka silti?", default=False, style=custom_style).ask():
                            return
            except Exception:
                pass

        # 4. Output-nimi
        default_name = f"{base_model.name}_{adapter_path.name}_merged"
        output_name = questionary.text(
            "Tulosmallin nimi:",
            default=default_name[:50],
            style=custom_style,
        ).ask()

        if not output_name:
            return

        # 5. Yhteenveto
        console.print(Panel(
            f"[white]Base-malli:[/white] {base_model.name}\n"
            f"[white]Adapter:[/white] {adapter_path.name}\n"
            f"[white]Adapterin base:[/white] {expected_base or 'Tuntematon'}\n"
            f"[white]Tuloste:[/white] {output_name}",
            title="[bold]LoRA Merge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # 6. Suorita merge
        console.print("\n[bold cyan]Suoritetaan LoRA merge...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        merge_result = self.merger.merge_lora(
            base_model_path=base_model,
            adapter_path=adapter_path,
            output_name=output_name,
            progress_callback=progress_cb,
        )

        if merge_result["success"]:
            console.print(Panel(
                f"[green]LoRA merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {merge_result['output_path']}\n"
                f"[white]Koko:[/white] {merge_result['file_size_gb']:.2f} GB",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Lisaa kirjastoon
            try:
                entry = self.library.add_model(
                    path=merge_result['output_path'],
                    source="merged",
                    source_id=f"lora:{base_model.name}+{adapter_path.name}",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {merge_result.get('error', 'Tuntematon virhe')}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _advanced_merge_wizard(self):
        """Advanced merge wizard - handles vocab/precision differences."""
        from ..merging.merger import AdvancedMergeConfig, MergeMethod

        print_mini_banner("Advanced Merge")

        console.print("[cyan]Advanced Merge kasittelee malleja joilla on:[/cyan]")
        console.print("  - Eri vocab_size (embedding trimaus/padding)")
        console.print("  - Eri presisio (FP16/BF16/FP32)")
        console.print("  - Eri RoPE scaling konfiguraatiot\n")

        # Select model 1
        console.print("[bold]Malli 1:[/bold]")
        model1 = self._select_model_for_merge("Valitse ensimmainen malli:")
        if not model1:
            return

        info1 = self.merger.get_model_info(model1)
        if info1.get("architecture"):
            console.print(f"  [green]+[/green] {info1.get('name')} ({info1.get('architecture')})")
            console.print(f"    [dim]Kerroksia: {info1.get('num_layers')}, Hidden: {info1.get('hidden_size')}, Vocab: {info1.get('vocab_size')}[/dim]\n")

        # Select model 2
        console.print("[bold]Malli 2:[/bold]")
        model2 = self._select_model_for_merge("Valitse toinen malli:")
        if not model2:
            return

        info2 = self.merger.get_model_info(model2)
        if info2.get("architecture"):
            console.print(f"  [green]+[/green] {info2.get('name')} ({info2.get('architecture')})")
            console.print(f"    [dim]Kerroksia: {info2.get('num_layers')}, Hidden: {info2.get('hidden_size')}, Vocab: {info2.get('vocab_size')}[/dim]\n")

        # Check compatibility with strict=False
        compatible, msg, details = self.merger.validate_models_compatible(
            [model1, model2], strict=False
        )

        if not compatible:
            print_error(f"Mallit eivat ole yhteensopivia: {msg}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Show differences
        if details.get("vocab_warning"):
            vocab_sizes = details.get('vocab_sizes', [])
            vocab_info = ""
            if len(vocab_sizes) >= 2:
                vocab_info = f"\n[white]Vocab sizes:[/white]\n  Malli 1: {vocab_sizes[0]}\n  Malli 2: {vocab_sizes[1]}\n"
            console.print(Panel(
                f"[yellow]{details['vocab_warning']}[/yellow]\n{vocab_info}\n"
                f"[dim]Advanced Merge kasittelee eron automaattisesti.[/dim]",
                title="[yellow]Huomio[/yellow]",
                border_style="yellow"
            ))

        # RoPE scaling info
        if details.get("rope_scaling"):
            rope1, rope2 = details["rope_scaling"]
            if rope1 != rope2:
                has_rope = "Malli 1" if rope1 else "Malli 2"
                console.print(f"[dim]RoPE scaling: {has_rope} - kaytetaan pidemman kontekstin konfiguraatiota[/dim]\n")

        print_success("Mallit voidaan yhdistaa Advanced Mergella!")

        # Vocab strategy
        vocab_strategy = questionary.select(
            "Vocab_size strategia:",
            choices=[
                questionary.Choice("Minimum - trimataan isompi (suositeltu)", value="minimum"),
                questionary.Choice("Maximum - paddataan pienempi", value="maximum"),
                questionary.Choice("Malli 1:n vocab", value="first"),
                questionary.Choice("Malli 2:n vocab", value="second"),
            ],
            default="minimum",
            style=custom_style,
        ).ask()

        if vocab_strategy is None:
            return

        # Target precision
        target_dtype = questionary.select(
            "Tulosteen presisio:",
            choices=[
                questionary.Choice("BFloat16 (suositeltu)", value="bfloat16"),
                questionary.Choice("Float16", value="float16"),
                questionary.Choice("Float32 (isoin)", value="float32"),
            ],
            default="bfloat16",
            style=custom_style,
        ).ask()

        if target_dtype is None:
            return

        # SLERP ratio
        ratio_str = questionary.text(
            "SLERP ratio (0.0-1.0, oletus 0.5):",
            default="0.5",
            style=custom_style,
        ).ask()

        try:
            ratio = float(ratio_str) if ratio_str else 0.5
            ratio = max(0.0, min(1.0, ratio))
        except ValueError:
            ratio = 0.5

        console.print(f"\n[dim]Ratio {ratio:.2f}: {100-ratio*100:.0f}% malli 1, {ratio*100:.0f}% malli 2[/dim]")

        # Tokenizer source
        tokenizer_source = questionary.select(
            "Tokenizer lahde:",
            choices=[
                questionary.Choice(f"Malli 1 ({model1.name[:30]})", value="first"),
                questionary.Choice(f"Malli 2 ({model2.name[:30]})", value="second"),
            ],
            default="first",
            style=custom_style,
        ).ask()

        if tokenizer_source is None:
            return

        # Output name
        default_name = f"advanced_{info1.get('name', 'model1')[:15]}_{info2.get('name', 'model2')[:15]}"
        output_name = questionary.text(
            "Tulosteen nimi:",
            default=default_name,
            style=custom_style,
        ).ask()

        if not output_name:
            return

        # Create config
        config = AdvancedMergeConfig(
            method=MergeMethod.SLERP,
            models=[model1, model2],
            output_name=output_name,
            vocab_strategy=vocab_strategy,
            target_dtype=target_dtype,
            slerp_ratio=ratio,
            config_source="first",
            merge_rope_scaling=True,
            tokenizer_source=tokenizer_source,
        )

        # Show summary
        reqs = self.merger.estimate_merge_requirements([model1, model2])

        console.print(Panel(
            f"[white]Malli 1:[/white] {model1.name}\n"
            f"[white]Malli 2:[/white] {model2.name}\n"
            f"[white]Vocab strategia:[/white] {vocab_strategy}\n"
            f"[white]Presisio:[/white] {target_dtype}\n"
            f"[white]Ratio:[/white] {ratio:.2f}\n"
            f"[white]Tokenizer:[/white] {'Malli 1' if tokenizer_source == 'first' else 'Malli 2'}\n"
            f"[white]Tuloste:[/white] {output_name}\n\n"
            f"[yellow]Arvioitu RAM-tarve:[/yellow] {reqs['estimated_ram_gb']:.1f} GB",
            title="[bold]Advanced Merge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Run merge
        console.print("\n[bold cyan]Suoritetaan Advanced Merge...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja...[/dim]\n")

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.merger.merge_advanced(
            model1_path=model1,
            model2_path=model2,
            config=config,
            progress_callback=progress_cb,
        )

        if result["success"]:
            original_vocabs = result.get("original_vocab_sizes", [])
            vocab_info = ""
            if len(original_vocabs) >= 2:
                vocab_info = f"\n[white]Alkuperaiset vocab:[/white] {original_vocabs[0]} / {original_vocabs[1]}"

            console.print(Panel(
                f"[green]Advanced Merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result['output_path']}\n"
                f"[white]Koko:[/white] {result['file_size_gb']:.2f} GB\n"
                f"[white]Lopullinen vocab:[/white] {result.get('final_vocab_size', 'N/A')}"
                f"{vocab_info}\n"
                f"[white]Presisio:[/white] {result.get('target_dtype', 'N/A')}",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged",
                    source_id=f"advanced:{model1.name}+{model2.name}",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            print_error(f"Merge epäonnistui: {result.get('error', 'Tuntematon virhe')}")
            if result.get("traceback"):
                console.print(f"\n[dim]{result['traceback'][:500]}[/dim]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    # =========================================================================
    # Mergekit Wizard Methods
    # =========================================================================

    def _mergekit_main_menu(self):
        """Mergekit Wizard päävalikko - kaikki merge-työkalut yhden valikon alla."""
        while True:
            print_mini_banner("Mergekit Wizard", "Yhdista malleja ammattilaistyokaluilla")

            # Check mergekit availability
            mergekit_installed, _ = self.mergekit.check_mergekit()

            # VRAM detection
            vram_gb, optimizations = self.mergekit.detect_vram()

            # Status display
            status_items = []
            if vram_gb > 0:
                vram_color = "green" if vram_gb >= 12 else ("yellow" if vram_gb >= 8 else "red")
                status_items.append(f"[dim]VRAM:[/dim] [{vram_color}]{vram_gb:.0f} GB[/{vram_color}]")

            if mergekit_installed:
                status_items.append("[dim]Mergekit:[/dim] [green]OK[/green]")
            else:
                status_items.append("[dim]Mergekit:[/dim] [yellow]Ei asennettu[/yellow]")

            # List saved configs count
            configs = self.config_manager.list_configs()
            if configs:
                status_items.append(f"[dim]Configs:[/dim] [cyan]{len(configs)}[/cyan]")

            if status_items:
                console.print("  " + "  |  ".join(status_items))
                console.print()

            # Build menu choices with consistent formatting
            choices = [
                questionary.Separator("--- Merge Tools ---"),
                questionary.Choice(
                    title=format_menu_item("New Merge", "Luo uusi merge wizardilla"),
                    value="new_merge"
                ),
                questionary.Choice(
                    title=format_menu_item("Presets", "Valmiit konfiguraatiot"),
                    value="presets"
                ),
                questionary.Choice(
                    title=format_menu_item("Config Manager", f"YAML-hallinta ({len(configs)} tallennettua)"),
                    value="config_manager"
                ),
                questionary.Separator("--- Info ---"),
                questionary.Choice(
                    title=format_menu_item("Merge Methods", "Tietoa eri menetelmista"),
                    value="info"
                ),
            ]

            if not mergekit_installed:
                choices.append(questionary.Separator("--- Setup ---"))
                choices.append(questionary.Choice(
                    title=format_menu_item("Install Mergekit", "Asenna mergekit-kirjasto"),
                    value="install"
                ))

            choices.extend([
                questionary.Separator("---------------------------"),
                questionary.Choice(
                    title=format_menu_item("<- Back", "Palaa edelliseen valikkoon"),
                    value="back"
                ),
            ])

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
            elif choice == "new_merge":
                if not mergekit_installed:
                    self._install_mergekit()
                else:
                    self._mergekit_wizard()
            elif choice == "presets":
                if not mergekit_installed:
                    self._install_mergekit()
                else:
                    self._presets_menu()
            elif choice == "config_manager":
                self._config_manager_menu()
            elif choice == "info":
                self._show_merge_info()
            elif choice == "install":
                self._install_mergekit()

    def _install_mergekit(self):
        """Install mergekit library."""
        print_mini_banner("Install Mergekit")

        console.print("[cyan]Mergekit on tehokas kirjasto mallien yhdistamiseen.[/cyan]")
        console.print("[dim]Tukee SLERP, DARE-TIES, TIES, DELLA ja muita metodeja.[/dim]\n")

        if not questionary.confirm("Asenna mergekit?", default=True, style=custom_style).ask():
            return

        console.print("\n[bold cyan]Asennetaan mergekit...[/bold cyan]")

        success = self.mergekit.install_mergekit()

        if success:
            print_success("Mergekit asennettu!")
            console.print("[dim]Voit nyt kayttaa Mergekit Wizard -toimintoa.[/dim]")
        else:
            print_error("Asennus epaonnistui")
            console.print("\n[dim]Kokeile manuaalisesti:[/dim]")
            console.print("[cyan]pip install mergekit[/cyan]")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _mergekit_wizard(self):
        """Mergekit merge wizard - main guided merge flow."""
        import time

        print_mini_banner("New Merge", "Luo uusi merge valitsemalla metodi ja mallit")

        # VRAM detection
        vram_gb, optimizations = self.mergekit.detect_vram()

        # Show VRAM status
        if vram_gb > 0:
            vram_color = "green" if vram_gb >= 12 else ("yellow" if vram_gb >= 8 else "red")
            opt_list = []
            if optimizations.get("lazy_unpickle"):
                opt_list.append("Lazy Load")
            if optimizations.get("low_cpu_memory"):
                opt_list.append("Low Memory")
            opt_str = f" [dim]({', '.join(opt_list)})[/dim]" if opt_list else ""
            console.print(f"  [dim]VRAM:[/dim] [{vram_color}]{vram_gb:.0f} GB[/{vram_color}]{opt_str}\n")

        # 1. Select merge method
        print_section_header("Step 1: Merge Method", "Valitse yhdistamistapa")

        method_choices = [
            questionary.Separator("--- Interpolation ---"),
            questionary.Choice(
                title=format_menu_item("SLERP", "Kaksi mallia, tasainen interpolointi"),
                value="slerp"
            ),
            questionary.Choice(
                title=format_menu_item("LINEAR", "Painotettu keskiarvo"),
                value="linear"
            ),
            questionary.Separator("--- Task Vectors ---"),
            questionary.Choice(
                title=format_menu_item("DARE-TIES", "2+ mallia, alykas harvennus (suositeltu)"),
                value="dare_ties"
            ),
            questionary.Choice(
                title=format_menu_item("DARE-LINEAR", "2+ mallia, lineaarinen DARE"),
                value="dare_linear"
            ),
            questionary.Choice(
                title=format_menu_item("TIES", "Trim-Elect-Merge"),
                value="ties"
            ),
            questionary.Choice(
                title=format_menu_item("Task Arithmetic", "Additiivinen task vector merge"),
                value="task_arithmetic"
            ),
            questionary.Separator("--- Advanced ---"),
            questionary.Choice(
                title=format_menu_item("DELLA", "DARE + pruning + rescale"),
                value="della"
            ),
            questionary.Separator("---------------------------"),
            questionary.Choice(
                title=format_menu_item("<- Cancel", "Peruuta"),
                value="back"
            ),
        ]

        method = questionary.select(
            "",
            choices=method_choices,
            style=custom_style,
            qmark="",
            pointer=">",
            instruction="(↑↓ valitse)"
        ).ask()

        if method is None or method == "back":
            return

        # Method configuration
        needs_base = method in {"dare_ties", "dare_linear", "ties", "task_arithmetic", "della"}

        if method == "slerp":
            min_models, max_models = 2, 2
        elif needs_base:
            min_models, max_models = 1, 10  # base + 1-10 finetuned mallia
        else:  # linear
            min_models, max_models = 2, 10

        # Method description
        method_desc = {
            "slerp": "Yhdistaa tasan 2 mallia sfaarisella interpoloinnilla",
            "linear": "Yhdistaa 2+ mallia painotetulla keskiarvolla",
            "dare_ties": "Vaatii base modelin + finetuned-malleja",
            "dare_linear": "Vaatii base modelin + finetuned-malleja",
            "ties": "Vaatii base modelin + finetuned-malleja",
            "task_arithmetic": "Vaatii base modelin + finetuned-malleja",
            "della": "Vaatii base modelin + finetuned-malleja",
        }

        console.print(f"\n  [bold cyan]Valittu:[/bold cyan] {method.upper()}")
        console.print(f"  [dim]{method_desc.get(method, '')}[/dim]\n")

        # 2. Select base model (if needed)
        base_model = None
        if needs_base:
            print_section_header("Step 2: Base Model", "Valitse alkuperainen pretrained-malli")
            base_model = self._select_model_for_merge("Base model:")
            if not base_model:
                return

            base_info = self.merger.get_model_info(base_model)
            console.print(f"  [green]✓[/green] {base_info.get('name')}\n")

        # 3. Select models to merge
        step_num = 3 if needs_base else 2
        print_section_header(f"Step {step_num}: Models", f"Valitse yhdistettavat mallit (max {max_models})")
        models = []

        while len(models) < max_models:
            model = self._select_model_for_merge(f"Model {len(models)+1}:")
            if not model:
                break

            model_info = self.merger.get_model_info(model)
            console.print(f"  [green]✓[/green] {model_info.get('name')}")

            # Basic compatibility check
            if base_model:
                compatible, msg, _ = self.merger.validate_models_compatible(
                    [base_model, model], strict=False
                )
                if not compatible:
                    console.print(f"    [yellow]⚠ {msg}[/yellow]")

            models.append(str(model))

            if method == "slerp" and len(models) == 2:
                break

            if len(models) >= min_models and len(models) < max_models:
                add_more = questionary.confirm(
                    "Lisaa toinen malli?",
                    default=False,
                    style=custom_style
                ).ask()
                if not add_more:
                    break

        if len(models) < min_models:
            console.print(f"\n  [red]✗[/red] Tarvitaan vahintaan {min_models} mallia")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        console.print()

        # 3.5 ARCHITECTURE COMPATIBILITY CHECK - NEW!
        console.print("[bold cyan]Tarkistetaan mallien yhteensopivuus...[/bold cyan]\n")

        all_model_paths = [Path(m) for m in models]
        if base_model:
            all_model_paths.append(base_model)

        validation = self.mergekit.validate_models(
            models=[Path(m) for m in models],
            method=MergekitMethod(method),
            base_model=base_model,
        )

        # Show model architecture comparison table
        if validation.get("model_info"):
            from rich.table import Table

            arch_table = Table(title="Mallien arkkitehtuurit", box=box.ROUNDED)
            arch_table.add_column("Malli", style="cyan")
            arch_table.add_column("vocab_size", justify="right")
            arch_table.add_column("max_position", justify="right")
            arch_table.add_column("rope_scaling", justify="center")
            arch_table.add_column("layers", justify="right")

            for info in validation["model_info"]:
                if info.get("config_found"):
                    rope_str = info.get("rope_scaling_type") or "null"
                    arch_table.add_row(
                        info.get("name", "?")[:30],
                        str(info.get("vocab_size", "?")),
                        str(info.get("max_position_embeddings", "?")),
                        rope_str,
                        str(info.get("num_layers", "?")),
                    )
                else:
                    arch_table.add_row(info.get("name", "?")[:30], "?", "?", "?", "?")

            console.print(arch_table)
            console.print()

        # Show compatibility result
        arch_check = validation.get("architecture_check", {})

        if arch_check.get("identical"):
            console.print("[bold green]✅ Mallit ovat täysin yhteensopivat![/bold green]")
            console.print("[dim]Kaikki merge-metodit toimivat.[/dim]\n")
        elif arch_check.get("compatible_slerp") and not arch_check.get("compatible_dare"):
            console.print("[bold yellow]⚠️  Mallit ovat OSITTAIN yhteensopivat[/bold yellow]")
            console.print("[yellow]Eri arkkitehtuurit: SLERP/LINEAR toimii, DARE-TIES EI![/yellow]\n")
        elif not arch_check.get("compatible_slerp"):
            console.print("[bold red]❌ Mallit EIVÄT OLE yhteensopivat![/bold red]\n")

        # Show warnings
        for warning in validation.get("warnings", []):
            print_warning(warning)

        # Show recommendation
        if validation.get("recommendation"):
            console.print(f"\n[bold cyan]Suositus:[/bold cyan] {validation['recommendation']}")

        # Show compatible methods
        compatible_methods = validation.get("compatible_methods", [])
        if compatible_methods:
            console.print("\n[dim]Yhteensopivat metodit:[/dim]")
            for m in compatible_methods:
                status = "✅" if m["compatible"] else "❌"
                rec = " (suositeltu)" if m.get("recommended") else ""
                console.print(f"  {status} {m['method'].value.upper()}{rec}")
            console.print()

        # Check if selected method is compatible
        method_compat = next((m for m in compatible_methods if m["method"].value == method), None)

        if method_compat and not method_compat["compatible"]:
            console.print(Panel(
                f"[bold red]VAROITUS:[/bold red] {method.upper()} ei ole yhteensopiva näille malleille!\n\n"
                f"[white]Syy:[/white] {method_compat['reason']}\n\n"
                f"[yellow]Suositus:[/yellow] Käytä SLERP tai LINEAR sen sijaan.",
                title="[red]Yhteensopivuusongelma[/red]",
                border_style="red"
            ))

            # Ask if user wants to continue anyway or switch method
            action = questionary.select(
                "Mitä haluat tehdä?",
                choices=[
                    questionary.Choice("Vaihda metodiksi SLERP (suositeltu)", value="slerp"),
                    questionary.Choice("Vaihda metodiksi LINEAR", value="linear"),
                    questionary.Choice("Jatka silti (todennäköisesti epäonnistuu)", value="continue"),
                    questionary.Choice("Peruuta", value="cancel"),
                ],
                style=custom_style,
            ).ask()

            if action == "cancel" or action is None:
                return
            elif action == "slerp":
                method = "slerp"
                needs_base = False
                console.print("[green]Vaihdettu metodiksi SLERP[/green]\n")
            elif action == "linear":
                method = "linear"
                needs_base = False
                console.print("[green]Vaihdettu metodiksi LINEAR[/green]\n")
            # else: continue with incompatible method (user's choice)

        console.print()

        # 4. Configure parameters based on method
        slerp_t = 0.5
        ties_density = 0.5
        normalize = True

        if method == "slerp":
            ratio_str = questionary.text(
                "SLERP t (0.0-1.0, oletus 0.5):",
                default="0.5",
                style=custom_style,
            ).ask()
            try:
                slerp_t = float(ratio_str) if ratio_str else 0.5
                if slerp_t < 0.0 or slerp_t > 1.0:
                    print_warning(f"SLERP t {slerp_t:.2f} rajattu välille 0.0-1.0")
                    slerp_t = max(0.0, min(1.0, slerp_t))
            except ValueError:
                print_warning("Virheellinen t, käytetään oletusta: 0.5")
                slerp_t = 0.5
            console.print(f"[dim]t={slerp_t:.2f}: {(1-slerp_t)*100:.0f}% malli 1, {slerp_t*100:.0f}% malli 2[/dim]")

        elif method in {"dare_ties", "dare_linear", "della", "ties"}:
            density_str = questionary.text(
                "Density (0.0-1.0, oletus 0.5):",
                default="0.5",
                style=custom_style,
            ).ask()
            try:
                ties_density = float(density_str) if density_str else 0.5
                if ties_density < 0.0 or ties_density > 1.0:
                    print_warning(f"Density {ties_density:.2f} rajattu välille 0.0-1.0")
                    ties_density = max(0.0, min(1.0, ties_density))
            except ValueError:
                print_warning("Virheellinen density, käytetään oletusta: 0.5")
                ties_density = 0.5
            console.print(f"[dim]Density {ties_density:.2f}: sailytetaan {ties_density*100:.0f}% painoista[/dim]")

        # 5. Output name - use clean naming
        from ..models.library import generate_merge_name
        default_name = generate_merge_name(
            method=method,
            model_names=models,
            base_model_name=base_model.name if base_model else None,
        )

        output_name = questionary.text(
            "Output name:",
            default=default_name,
            style=custom_style,
        ).ask()

        if not output_name:
            return

        # 6. Show summary
        model_list = "\n".join([f"  - {Path(m).name}" for m in models])

        summary_text = f"[white]Metodi:[/white] {method.upper()}\n"
        if base_model:
            summary_text += f"[white]Base:[/white] {base_model.name}\n"
        summary_text += f"[white]Mallit:[/white]\n{model_list}\n"

        if method == "slerp":
            summary_text += f"[white]t:[/white] {slerp_t:.2f}\n"
        elif method in {"dare_ties", "dare_linear", "della", "ties"}:
            summary_text += f"[white]Density:[/white] {ties_density:.2f}\n"

        summary_text += f"[white]Tuloste:[/white] {output_name}\n\n"

        if optimizations:
            opt_list = []
            if optimizations.get("cuda"):
                opt_list.append("CUDA")
            if optimizations.get("lazy_unpickle"):
                opt_list.append("Lazy loading")
            if optimizations.get("low_cpu_memory"):
                opt_list.append("Low CPU memory")
            summary_text += f"[yellow]Optimoinnit:[/yellow] {', '.join(opt_list)}"

        console.print(Panel(
            summary_text,
            title="[bold]Mergekit Merge[/bold]",
            border_style="yellow"
        ))

        # 7. Save config option
        save_config = questionary.confirm(
            "Tallenna konfiguraatio YAML:ksi?",
            default=False,
            style=custom_style
        ).ask()

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # 8. Create config
        try:
            # Build output path
            paths = get_paths()
            output_path = paths.merged_dir / output_name

            config = MergekitConfig(
                method=MergekitMethod(method),
                models=[Path(m) for m in models],
                output_path=output_path,
                base_model=base_model if base_model else None,
                slerp_t=slerp_t,
                ties_density=ties_density,
                normalize=normalize,
                dtype="bfloat16",
                # Apply VRAM optimizations
                cuda=optimizations.get("cuda", True),
                low_cpu_memory=optimizations.get("low_cpu_memory", True),
                lazy_unpickle=optimizations.get("lazy_unpickle", True),
                out_shard_size=optimizations.get("out_shard_size", "5B"),
            )

            # Save config if requested
            if save_config:
                config_dict = config.to_yaml_dict()
                self.config_manager.save_config(
                    config=config_dict,
                    name=output_name,
                    description=f"{method.upper()} merge: {len(models)} models",
                    tags=[method],
                )
                print_info(f"Konfiguraatio tallennettu: {output_name}.yaml")

        except Exception as e:
            print_error(f"Konfiguraatiovirhe: {e}")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # 9. Run merge
        console.print("\n[bold cyan]Suoritetaan merge...[/bold cyan]")
        console.print("[dim]Tama voi kestaa useita minuutteja mallien koosta riippuen...[/dim]\n")

        start_time = time.time()

        def progress_cb(msg):
            console.print(f"  [dim]{msg}[/dim]")

        result = self.mergekit.merge(config, progress_callback=progress_cb)

        duration = time.time() - start_time

        # 10. Handle result
        if result.get("success"):
            console.print(Panel(
                f"[green]Merge valmis![/green]\n\n"
                f"[white]Tiedosto:[/white] {result['output_path']}\n"
                f"[white]Kesto:[/white] {duration/60:.1f} min",
                title="[bold green]Success[/bold green]",
                border_style="green"
            ))

            # Add to history
            self.config_manager.add_history_entry(
                config_name=output_name,
                method=method,
                models=models,
                output_path=str(result['output_path']),
                success=True,
                duration_seconds=duration,
            )

            # Add to library
            try:
                entry = self.library.add_model(
                    path=result['output_path'],
                    source="merged",
                    source_id=f"mergekit:{method}",
                )
                print_success(f"Lisatty kirjastoon: {entry.name}")
            except Exception as e:
                print_warning(f"Kirjastolisays epäonnistui: {e}")
        else:
            error_msg = result.get("error", "Tuntematon virhe")
            print_error(f"Merge epaonnistui: {error_msg}")

            # Add failed entry to history
            self.config_manager.add_history_entry(
                config_name=output_name,
                method=method,
                models=models,
                output_path="",
                success=False,
                duration_seconds=duration,
                error_message=error_msg,
            )

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _presets_menu(self):
        """Presets menu - browse and use preset configurations."""
        while True:
            print_mini_banner("Presets", "Valmiit konfiguraatiot yleisiin kayttotapauksiin")

            # Group presets by category
            by_category = get_presets_by_category()

            choices = []
            for category in PresetCategory:
                if category in by_category:
                    # Category separator
                    cat_name = {
                        "interpolation": "Interpolation",
                        "task_vectors": "Task Vectors",
                        "advanced": "Advanced",
                        "experimental": "Experimental",
                    }.get(category.value, category.value.upper())
                    choices.append(questionary.Separator(f"--- {cat_name} ---"))

                    for preset in by_category[category]:
                        preset_key = [k for k, v in PRESETS.items() if v == preset][0]
                        title = format_menu_item(preset.name, preset.description[:35])
                        choices.append(questionary.Choice(title=title, value=preset_key))

            choices.append(questionary.Separator("---------------------------"))
            choices.append(questionary.Choice(
                title=format_menu_item("<- Back", "Palaa valikkoon"),
                value="back"
            ))

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

            self._use_preset(choice)

    def _use_preset(self, preset_name: str):
        """Use a specific preset to create a merge."""
        preset = get_preset(preset_name)
        if not preset:
            print_error(f"Presetia ei loydy: {preset_name}")
            return

        print_mini_banner(preset.name)

        console.print(f"[cyan]{preset.description}[/cyan]")
        console.print(f"[dim]Metodi: {preset.method.upper()}[/dim]")
        console.print(f"[dim]Malleja: {preset.min_models}-{preset.max_models}[/dim]")
        if preset.requires_base:
            console.print("[dim]Vaatii base modelin[/dim]")
        console.print()

        # Select base model if required
        base_model = None
        if preset.requires_base:
            console.print("[bold]Base Model:[/bold]")
            base_model = self._select_model_for_merge("Valitse base model:")
            if not base_model:
                return
            console.print(f"  [green]+[/green] {base_model.name}\n")

        # Select models
        models = []
        console.print(f"[bold]Mallit ({preset.min_models}-{preset.max_models}):[/bold]")

        while len(models) < preset.max_models:
            model = self._select_model_for_merge(f"Valitse malli {len(models)+1}:")
            if not model:
                break

            models.append(str(model))
            console.print(f"  [green]+[/green] {model.name}")

            if len(models) >= preset.min_models and len(models) < preset.max_models:
                if not questionary.confirm("Lisaa toinen malli?", default=False, style=custom_style).ask():
                    break

        if len(models) < preset.min_models:
            print_warning(f"Tarvitaan vahintaan {preset.min_models} mallia")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        # Output name
        default_name = f"{preset_name}_{Path(models[0]).stem[:15]}"
        output_name = questionary.text(
            "Tulosteen nimi:",
            default=default_name,
            style=custom_style,
        ).ask()

        if not output_name:
            return

        # Show summary
        console.print(Panel(
            f"[white]Preset:[/white] {preset.name}\n"
            f"[white]Metodi:[/white] {preset.method.upper()}\n"
            f"[white]Mallit:[/white] {len(models)} kpl\n"
            f"[white]Tuloste:[/white] {output_name}",
            title="[bold]Preset Merge[/bold]",
            border_style="yellow"
        ))

        if not questionary.confirm("Aloita merge?", default=True, style=custom_style).ask():
            return

        # Create MergekitConfig and run
        try:
            paths = get_paths()
            output_path = paths.merged_dir / output_name

            # Extract parameters from preset defaults
            default_params = preset.default_params
            config = MergekitConfig(
                method=MergekitMethod(preset.method),
                models=[Path(m) for m in models],
                output_path=output_path,
                base_model=base_model if base_model else None,
                slerp_t=default_params.get("t", 0.5),
                ties_density=default_params.get("density", 0.5),
                normalize=default_params.get("normalize", True),
                int8_mask=default_params.get("int8_mask", True),
                dtype="bfloat16",
            )

            console.print("\n[bold cyan]Suoritetaan merge...[/bold cyan]\n")

            def progress_cb(msg):
                console.print(f"  [dim]{msg}[/dim]")

            result = self.mergekit.merge(config, progress_callback=progress_cb)

            if result.get("success"):
                print_success(f"Merge valmis: {result['output_path']}")

                try:
                    self.library.add_model(
                        path=result['output_path'],
                        source="merged",
                        source_id=f"preset:{preset_name}",
                    )
                except Exception:
                    pass
            else:
                print_error(f"Merge epaonnistui: {result.get('error')}")

        except Exception as e:
            print_error(f"Virhe: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _config_manager_menu(self):
        """Config manager menu - manage saved YAML configurations."""
        while True:
            print_mini_banner("Config Manager", "Hallitse YAML-konfiguraatioita")

            # List saved configs
            configs = self.config_manager.list_configs()

            if configs:
                table = Table(
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold cyan",
                    border_style="dim"
                )
                table.add_column("Name", style="white")
                table.add_column("Method", style="yellow")
                table.add_column("Models", justify="center")
                table.add_column("Modified", style="dim")

                for cfg in configs[:10]:
                    table.add_row(
                        cfg["name"][:25],
                        cfg["method"].upper(),
                        str(cfg["models_count"]),
                        cfg["modified"][:10],
                    )

                console.print(table)
                console.print()
            else:
                console.print("  [dim]Ei tallennettuja konfiguraatioita[/dim]\n")

            choices = [
                questionary.Separator("--- Actions ---"),
                questionary.Choice(
                    title=format_menu_item("Load & Run", "Lataa YAML ja aja merge"),
                    value="load"
                ),
                questionary.Choice(
                    title=format_menu_item("Import", "Tuo YAML toisesta sijainnista"),
                    value="import"
                ),
                questionary.Choice(
                    title=format_menu_item("Export", "Vie YAML toiseen sijaintiin"),
                    value="export"
                ),
                questionary.Separator("--- History ---"),
                questionary.Choice(
                    title=format_menu_item("Merge History", "Nae aiemmat merget"),
                    value="history"
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
            elif choice == "load":
                self._load_and_run_config()
            elif choice == "import":
                self._import_config()
            elif choice == "export":
                self._export_config()
            elif choice == "history":
                self._show_merge_history()

    def _load_and_run_config(self):
        """Load a saved config and run the merge."""
        configs = self.config_manager.list_configs()

        if not configs:
            print_warning("Ei tallennettuja konfiguraatioita")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices = []
        for cfg in configs:
            title = f"{cfg['name']:<25} {cfg['method']:<12} {cfg['models_count']} mallia"
            choices.append(questionary.Choice(title=title, value=cfg["name"]))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value="back"))

        choice = questionary.select(
            "Valitse konfiguraatio:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if choice is None or choice == "back":
            return

        try:
            config_dict = self.config_manager.load_config(choice)

            # Validate
            is_valid, errors = self.config_manager.validate_config(config_dict)
            if not is_valid:
                print_error(f"Virheellinen konfiguraatio: {', '.join(errors)}")
                questionary.press_any_key_to_continue(style=custom_style).ask()
                return

            # Show config
            import yaml
            console.print(Panel(
                yaml.dump(config_dict, default_flow_style=False),
                title=f"[bold]{choice}[/bold]",
                border_style="cyan"
            ))

            if not questionary.confirm("Suorita merge?", default=True, style=custom_style).ask():
                return

            # Run merge
            console.print("\n[bold cyan]Suoritetaan merge...[/bold cyan]\n")

            def progress_cb(msg):
                console.print(f"  [dim]{msg}[/dim]")

            result = self.mergekit.merge_from_dict(config_dict, progress_callback=progress_cb)

            if result.get("success"):
                print_success(f"Merge valmis: {result['output_path']}")
            else:
                print_error(f"Merge epaonnistui: {result.get('error')}")

        except Exception as e:
            print_error(f"Virhe: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _import_config(self):
        """Import a config from another location."""
        console.print("[dim]Anna polku YAML-tiedostoon:[/dim]")

        source_path = questionary.path(
            "Lahdetiedosto:",
            style=custom_style,
        ).ask()

        if not source_path:
            return

        name = questionary.text(
            "Nimi (oletus: tiedostonimi):",
            default="",
            style=custom_style,
        ).ask()

        try:
            result = self.config_manager.import_config(
                source_path=source_path,
                name=name if name else None,
            )
            print_success(f"Tuotu: {result}")
        except Exception as e:
            print_error(f"Tuonti epaonnistui: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _export_config(self):
        """Export a config to another location."""
        configs = self.config_manager.list_configs()

        if not configs:
            print_warning("Ei tallennettuja konfiguraatioita")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        choices = [
            questionary.Choice(title=cfg["name"], value=cfg["name"])
            for cfg in configs
        ]
        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="<-  Peruuta", value="back"))

        choice = questionary.select(
            "Valitse vietava:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if choice is None or choice == "back":
            return

        output_path = questionary.path(
            "Kohdepolku:",
            style=custom_style,
        ).ask()

        if not output_path:
            return

        try:
            result = self.config_manager.export_config(choice, output_path)
            print_success(f"Viety: {result}")
        except Exception as e:
            print_error(f"Vienti epaonnistui: {e}")

        questionary.press_any_key_to_continue(style=custom_style).ask()

    def _show_merge_history(self):
        """Show merge history."""
        print_mini_banner("Merge Historia")

        history = self.config_manager.get_history(limit=20)

        if not history:
            console.print("[dim]Ei historiaa.[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return

        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Aika", style="dim")
        table.add_column("Config", style="cyan")
        table.add_column("Metodi", style="yellow")
        table.add_column("Tila")
        table.add_column("Kesto")

        for entry in history:
            status = "[green]OK[/green]" if entry.success else "[red]FAIL[/red]"
            duration = f"{entry.duration_seconds/60:.1f}m" if entry.duration_seconds else "-"
            table.add_row(
                entry.timestamp[:16],
                entry.config_name[:20],
                entry.method,
                status,
                duration,
            )

        console.print(table)

        if questionary.confirm("Tyhjenna historia?", default=False, style=custom_style).ask():
            count = self.config_manager.clear_history()
            print_info(f"Poistettu {count} merkintaa")

        questionary.press_any_key_to_continue(style=custom_style).ask()
