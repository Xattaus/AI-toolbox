"""
AI TOOLBOX - Merge Config Manager
=================================

YAML-pohjainen konfiguraatioiden hallinta mergeille.
"""

import yaml
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .presets import MergePreset, get_preset, PRESETS


@dataclass
class MergeHistoryEntry:
    """Merge-historian merkinta."""
    timestamp: str
    config_name: str
    method: str
    models: List[str]
    output_path: str
    success: bool
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Muunna dictiksi."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeHistoryEntry":
        """Luo dictista."""
        return cls(**data)


@dataclass
class MergeConfigMetadata:
    """Konfiguraation metadata."""
    name: str
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    preset_used: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class MergeConfigManager:
    """
    Hallitsee merge-konfiguraatioita YAML-muodossa.

    Ominaisuudet:
    - Tallenna ja lataa konfiguraatioita
    - Historia merge-operaatioista
    - Import/export konfiguraatioita
    - Preset-pohjaiset konfiguraatiot
    """

    DEFAULT_CONFIG_DIR = Path("models/merge_configs")
    HISTORY_FILE = "merge_history.json"

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Alusta config manager.

        Args:
            config_dir: Konfiguraatiokansio (oletus: models/merge_configs)
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Etsi projektin juuresta
            self.config_dir = self._find_project_root() / self.DEFAULT_CONFIG_DIR

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.config_dir / self.HISTORY_FILE

    def _find_project_root(self) -> Path:
        """Etsi projektin juurikansio."""
        # Yrita loytyy pyproject.toml tai .git
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
            if (parent / ".git").exists():
                return parent
        return current

    # =========================================================================
    # Konfiguraatioiden hallinta
    # =========================================================================

    def save_config(
        self,
        config: Dict[str, Any],
        name: str,
        description: str = "",
        preset_used: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Tallenna merge-konfiguraatio YAML-tiedostoon.

        Args:
            config: Mergekit-yhteensopiva config dict
            name: Konfiguraation nimi (ilman .yaml)
            description: Kuvaus
            preset_used: Kaytetty preset (jos kaytettiin)
            tags: Tagit kategorisointiin

        Returns:
            Tallennetun tiedoston polku
        """
        # Sanitoi nimi
        safe_name = self._sanitize_filename(name)
        file_path = self.config_dir / f"{safe_name}.yaml"

        # Lisaa metadata kommentteina
        metadata = MergeConfigMetadata(
            name=name,
            description=description,
            preset_used=preset_used,
            tags=tags or [],
        )

        # Muodosta YAML-sisalto
        yaml_content = self._create_yaml_with_metadata(config, metadata)

        # Tallenna
        file_path.write_text(yaml_content, encoding="utf-8")

        return file_path

    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Lataa merge-konfiguraatio YAML-tiedostosta.

        Args:
            name: Konfiguraation nimi (ilman .yaml)

        Returns:
            Mergekit-yhteensopiva config dict

        Raises:
            FileNotFoundError: Jos konfiguraatiota ei loydy
        """
        safe_name = self._sanitize_filename(name)
        file_path = self.config_dir / f"{safe_name}.yaml"

        if not file_path.exists():
            # Yrita myos suoraan nimella
            file_path = self.config_dir / f"{name}.yaml"
            if not file_path.exists():
                raise FileNotFoundError(f"Konfiguraatiota ei loydy: {name}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def load_config_with_metadata(
        self, name: str
    ) -> tuple[Dict[str, Any], MergeConfigMetadata]:
        """
        Lataa konfiguraatio ja metadata.

        Args:
            name: Konfiguraation nimi

        Returns:
            Tuple: (config, metadata)
        """
        config = self.load_config(name)

        # Parsi metadata kommenteista
        safe_name = self._sanitize_filename(name)
        file_path = self.config_dir / f"{safe_name}.yaml"
        if not file_path.exists():
            file_path = self.config_dir / f"{name}.yaml"

        metadata = self._parse_metadata_from_file(file_path)

        return config, metadata

    def delete_config(self, name: str) -> bool:
        """
        Poista konfiguraatio.

        Args:
            name: Konfiguraation nimi

        Returns:
            True jos poistettiin, False jos ei loytynyt
        """
        safe_name = self._sanitize_filename(name)
        file_path = self.config_dir / f"{safe_name}.yaml"

        if not file_path.exists():
            file_path = self.config_dir / f"{name}.yaml"

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_configs(
        self,
        tag: Optional[str] = None,
        method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Listaa tallennetut konfiguraatiot.

        Args:
            tag: Suodata tagin mukaan
            method: Suodata metodin mukaan

        Returns:
            Lista: [{name, description, method, models_count, modified, tags}]
        """
        configs = []

        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                if config is None:
                    continue

                metadata = self._parse_metadata_from_file(yaml_file)

                # Suodatus
                if method and config.get("merge_method") != method:
                    continue
                if tag and tag not in metadata.tags:
                    continue

                # Laske mallien maara
                models = config.get("models", [])
                models_count = len(models) if isinstance(models, list) else 0

                configs.append({
                    "name": yaml_file.stem,
                    "description": metadata.description,
                    "method": config.get("merge_method", "unknown"),
                    "models_count": models_count,
                    "modified": metadata.modified,
                    "tags": metadata.tags,
                    "file_path": str(yaml_file),
                })
            except Exception:
                # Skipataan virheelliset tiedostot
                continue

        # Jarjesta muokkausajan mukaan (uusin ensin)
        configs.sort(key=lambda x: x["modified"], reverse=True)

        return configs

    # =========================================================================
    # Preset-pohjaiset konfiguraatiot
    # =========================================================================

    def create_from_preset(
        self,
        preset_name: str,
        models: List[str],
        name: str,
        base_model: Optional[str] = None,
        dtype: str = "bfloat16",
        **param_overrides,
    ) -> Path:
        """
        Luo konfiguraatio presetista.

        Args:
            preset_name: Presetin nimi (esim. "slerp_balanced")
            models: Lista mallipoluista
            name: Tallennettava nimi
            base_model: Base model polku
            dtype: Datan tyyppi
            **param_overrides: Parametrien ylikirjoitukset

        Returns:
            Tallennetun tiedoston polku

        Raises:
            ValueError: Jos presetia ei loydy
        """
        preset = get_preset(preset_name)
        if not preset:
            raise ValueError(f"Presetia ei loydy: {preset_name}")

        # Luo config presetista
        config = preset.to_config_dict(
            models=models,
            base_model=base_model,
            dtype=dtype,
            **param_overrides,
        )

        # Tallenna
        return self.save_config(
            config=config,
            name=name,
            description=preset.description,
            preset_used=preset_name,
            tags=[preset.category.value],
        )

    def get_preset_suggestions(
        self,
        models: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Ehdota sopivia presetteja mallien perusteella.

        Args:
            models: Lista mallien nimista/poluista

        Returns:
            Lista ehdotuksista: [{preset_name, preset, score, reason}]
        """
        model_names_lower = [Path(m).name.lower() for m in models]
        suggestions = []

        for preset_name, preset in PRESETS.items():
            score = 0
            reasons = []

            # Mallien maara
            if preset.min_models <= len(models) <= preset.max_models:
                score += 10
            else:
                continue  # Ei sovi mallimaaralle

            # Avainsanat
            if "*" not in preset.recommended_keywords:
                for keyword in preset.recommended_keywords:
                    for model_name in model_names_lower:
                        if keyword.lower() in model_name:
                            score += 5
                            reasons.append(f"Malli sisaltaa '{keyword}'")

            # Base model -vaatimus
            if preset.requires_base:
                reasons.append("Vaatii base modelin")

            if score > 0:
                suggestions.append({
                    "preset_name": preset_name,
                    "preset": preset,
                    "score": score,
                    "reasons": reasons,
                })

        # Jarjesta pisteiden mukaan
        suggestions.sort(key=lambda x: x["score"], reverse=True)

        return suggestions[:5]  # Top 5

    # =========================================================================
    # Import/Export
    # =========================================================================

    def export_config(
        self,
        name: str,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Vie konfiguraatio toiseen sijaintiin.

        Args:
            name: Konfiguraation nimi
            output_path: Kohdepolku

        Returns:
            Viedyn tiedoston polku
        """
        safe_name = self._sanitize_filename(name)
        source = self.config_dir / f"{safe_name}.yaml"

        if not source.exists():
            source = self.config_dir / f"{name}.yaml"
            if not source.exists():
                raise FileNotFoundError(f"Konfiguraatiota ei loydy: {name}")

        output = Path(output_path)
        if output.is_dir():
            output = output / source.name

        shutil.copy2(source, output)
        return output

    def import_config(
        self,
        source_path: Union[str, Path],
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Tuo konfiguraatio toisesta sijainnista.

        Args:
            source_path: Lahdetiedoston polku
            name: Uusi nimi (oletus: alkuperainen)
            overwrite: Ylikirjoita olemassa oleva

        Returns:
            Tuodun tiedoston polku
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Lahdetiedostoa ei loydy: {source_path}")

        # Validoi etta on YAML
        try:
            with open(source, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Tiedosto ei sisalla YAML dict:ia")
        except yaml.YAMLError as e:
            raise ValueError(f"Virheellinen YAML: {e}")

        # Maarita kohdenimi
        target_name = name or source.stem
        safe_name = self._sanitize_filename(target_name)
        target = self.config_dir / f"{safe_name}.yaml"

        if target.exists() and not overwrite:
            raise FileExistsError(f"Konfiguraatio on jo olemassa: {target_name}")

        shutil.copy2(source, target)
        return target

    def import_from_huggingface(
        self,
        config_yaml: str,
        name: str,
    ) -> Path:
        """
        Tuo konfiguraatio HuggingFace-muotoisesta YAML-stringista.

        Args:
            config_yaml: YAML-sisalto stringina
            name: Tallennettava nimi

        Returns:
            Tuodun tiedoston polku
        """
        # Validoi YAML
        try:
            config = yaml.safe_load(config_yaml)
            if not isinstance(config, dict):
                raise ValueError("YAML ei sisalla dict:ia")
        except yaml.YAMLError as e:
            raise ValueError(f"Virheellinen YAML: {e}")

        # Tallenna
        return self.save_config(
            config=config,
            name=name,
            description="Tuotu HuggingFace-muodosta",
            tags=["imported"],
        )

    # =========================================================================
    # Historia
    # =========================================================================

    def add_history_entry(
        self,
        config_name: str,
        method: str,
        models: List[str],
        output_path: str,
        success: bool,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> MergeHistoryEntry:
        """
        Lisaa merkinta merge-historiaan.

        Args:
            config_name: Kaytetty konfiguraatio
            method: Merge-metodi
            models: Kaytetyt mallit
            output_path: Tulospolku
            success: Onnistuiko
            duration_seconds: Kesto sekunneissa
            error_message: Virheviesti (jos epaonnistui)

        Returns:
            Luotu historiamerkinta
        """
        entry = MergeHistoryEntry(
            timestamp=datetime.now().isoformat(),
            config_name=config_name,
            method=method,
            models=models,
            output_path=output_path,
            success=success,
            duration_seconds=duration_seconds,
            error_message=error_message,
        )

        # Lataa historia
        history = self._load_history()
        history.append(entry.to_dict())

        # Rajoita historia 100 merkintaan
        if len(history) > 100:
            history = history[-100:]

        # Tallenna
        self._save_history(history)

        return entry

    def get_history(
        self,
        limit: int = 20,
        success_only: bool = False,
    ) -> List[MergeHistoryEntry]:
        """
        Hae merge-historia.

        Args:
            limit: Maksimimaara
            success_only: Vain onnistuneet

        Returns:
            Lista historiamerkinnosta (uusin ensin)
        """
        history = self._load_history()

        if success_only:
            history = [h for h in history if h.get("success", False)]

        # Kaanna jarjestys (uusin ensin) ja rajoita
        history = history[::-1][:limit]

        return [MergeHistoryEntry.from_dict(h) for h in history]

    def clear_history(self) -> int:
        """
        Tyhjenna historia.

        Returns:
            Poistettujen merkintoje maara
        """
        history = self._load_history()
        count = len(history)
        self._save_history([])
        return count

    def _load_history(self) -> List[Dict[str, Any]]:
        """Lataa historia tiedostosta."""
        if not self._history_path.exists():
            return []
        try:
            with open(self._history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        """Tallenna historia tiedostoon."""
        with open(self._history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # Apumetodit
    # =========================================================================

    def _sanitize_filename(self, name: str) -> str:
        """Poista tiedostonimesta kielletyt merkit."""
        # Korvaa kielletyt merkit alaviivalla
        forbidden = '<>:"/\\|?*'
        result = name
        for char in forbidden:
            result = result.replace(char, "_")
        return result.strip()

    def _create_yaml_with_metadata(
        self,
        config: Dict[str, Any],
        metadata: MergeConfigMetadata,
    ) -> str:
        """Luo YAML metadatan kanssa."""
        lines = [
            f"# {metadata.name}",
            f"# {metadata.description}" if metadata.description else None,
            f"# Created: {metadata.created}",
            f"# Modified: {metadata.modified}",
            f"# Preset: {metadata.preset_used}" if metadata.preset_used else None,
            f"# Tags: {', '.join(metadata.tags)}" if metadata.tags else None,
            "#",
            "",
        ]

        # Poista None-arvot
        header = "\n".join([l for l in lines if l is not None])

        # Muunna config YAML:ksi
        yaml_content = yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

        return header + yaml_content

    def _parse_metadata_from_file(self, file_path: Path) -> MergeConfigMetadata:
        """Parsi metadata YAML-tiedoston kommenteista."""
        metadata = MergeConfigMetadata(name=file_path.stem)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("#"):
                        break

                    line = line[1:].strip()

                    if line.startswith("Created:"):
                        metadata.created = line[8:].strip()
                    elif line.startswith("Modified:"):
                        metadata.modified = line[9:].strip()
                    elif line.startswith("Preset:"):
                        metadata.preset_used = line[7:].strip()
                    elif line.startswith("Tags:"):
                        tags_str = line[5:].strip()
                        metadata.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                    elif not any(line.startswith(x) for x in ["Created", "Modified", "Preset", "Tags", ""]):
                        # Ensimmainen ei-tyhja kommentti on kuvaus
                        if not metadata.description:
                            metadata.description = line
        except IOError:
            pass

        return metadata

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validoi merge-konfiguraatio.

        Args:
            config: Konfiguraatio dict

        Returns:
            Tuple: (is_valid, list of errors)
        """
        errors = []

        # Vaaditut kentat
        if "merge_method" not in config:
            errors.append("Puuttuu: merge_method")

        if "models" not in config:
            errors.append("Puuttuu: models")
        elif not isinstance(config["models"], list):
            errors.append("models pitaa olla lista")
        elif len(config["models"]) < 2:
            errors.append("models pitaa sisaltaa vahintaan 2 mallia")

        # Metodikohtaiset validoinnit
        method = config.get("merge_method", "")

        if method in {"dare_ties", "dare_linear", "ties", "task_arithmetic", "della"}:
            if "base_model" not in config:
                errors.append(f"{method} vaatii base_model:in")

        if method == "slerp":
            models = config.get("models", [])
            if len(models) != 2:
                errors.append("SLERP tukee vain 2 mallia")

            params = config.get("parameters", {})
            t = params.get("t")
            if t is not None and not (0 <= t <= 1):
                errors.append("SLERP t-parametri pitaa olla valilla 0-1")

        # Dtype
        valid_dtypes = {"float32", "float16", "bfloat16"}
        dtype = config.get("dtype", "bfloat16")
        if dtype not in valid_dtypes:
            errors.append(f"Virheellinen dtype: {dtype}. Sallitut: {valid_dtypes}")

        return len(errors) == 0, errors
