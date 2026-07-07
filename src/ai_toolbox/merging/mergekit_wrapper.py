"""
AI TOOLBOX - Mergekit Wrapper
=============================

Wrapper mergekit-kirjastolle, tarjoaa:
- Yksinkertaistettu API mallien yhdistamiseen
- VRAM-optimoitu oletuskonfiguraatio (10GB yhteensopiva)
- Automaattinen vocab_size kasittely
- Progress callback -tuki
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml
from rich.console import Console

from ..core.paths import get_paths

console = Console()


class MergekitMethod(str, Enum):
    """Tuetut merge-metodit."""
    SLERP = "slerp"
    LINEAR = "linear"
    TIES = "ties"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    TASK_ARITHMETIC = "task_arithmetic"
    BREADCRUMBS = "breadcrumbs"
    DELLA = "della"
    MODEL_STOCK = "model_stock"
    PASSTHROUGH = "passthrough"


# Metodit jotka vaativat base modelin
METHODS_REQUIRING_BASE = {
    MergekitMethod.TIES,
    MergekitMethod.DARE_TIES,
    MergekitMethod.DARE_LINEAR,
    MergekitMethod.TASK_ARITHMETIC,
    MergekitMethod.DELLA,
    MergekitMethod.BREADCRUMBS,
}

# Metodit jotka tukevat vain 2 mallia
TWO_MODEL_METHODS = {
    MergekitMethod.SLERP,
}


@dataclass
class MergekitConfig:
    """Mergekit merge-konfiguraatio."""
    method: MergekitMethod
    models: List[Path]
    output_path: Path

    # Mallikohtaiset parametrit
    model_weights: Optional[List[float]] = None
    model_densities: Optional[List[float]] = None

    # Metodikohtaiset parametrit
    slerp_t: float = 0.5
    ties_density: float = 0.5
    normalize: bool = True
    rescale: bool = True
    int8_mask: bool = True

    # Base model (TIES/DARE vaatii)
    base_model: Optional[Path] = None

    # Output asetukset
    dtype: str = "bfloat16"
    out_shard_size: str = "5B"
    tokenizer_source: str = "base"  # "base", "union", tai model path

    # VRAM-optimoinnit (oletukset 10GB VRAM:lle)
    cuda: bool = True
    low_cpu_memory: bool = True
    lazy_unpickle: bool = True

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Muunna mergekit YAML-muotoon."""
        config: Dict[str, Any] = {
            "merge_method": self.method.value,
            "dtype": self.dtype,
        }

        # Methods that require explicit base_model from user
        methods_requiring_explicit_base = {
            MergekitMethod.TIES,
            MergekitMethod.DARE_TIES,
            MergekitMethod.DARE_LINEAR,
            MergekitMethod.TASK_ARITHMETIC,
            MergekitMethod.DELLA,
        }

        # Models
        if self.method == MergekitMethod.SLERP:
            # SLERP: 2 mallia, t-parametri
            # SLERP needs base_model to determine which model is t=0.0
            # Use first model as base_model
            config["base_model"] = str(self.models[0])
            config["models"] = [
                {"model": str(self.models[0])},
                {"model": str(self.models[1])},
            ]
            config["parameters"] = {"t": self.slerp_t}
            # Tokenizer from first model
            config["tokenizer_source"] = "base"

        elif self.method in {MergekitMethod.TIES, MergekitMethod.DARE_TIES,
                             MergekitMethod.DARE_LINEAR, MergekitMethod.TASK_ARITHMETIC}:
            # TIES/DARE: base_model + N mallia, weight + density
            if self.base_model:
                config["base_model"] = str(self.base_model)
                config["tokenizer_source"] = "base"

            models_list = []
            for i, model in enumerate(self.models):
                model_def: Dict[str, Any] = {"model": str(model)}
                params: Dict[str, Any] = {}

                if self.model_weights and i < len(self.model_weights):
                    params["weight"] = self.model_weights[i]
                else:
                    params["weight"] = 1.0 / len(self.models)

                if self.model_densities and i < len(self.model_densities):
                    params["density"] = self.model_densities[i]
                elif self.method in {MergekitMethod.DARE_TIES, MergekitMethod.DARE_LINEAR}:
                    params["density"] = self.ties_density

                if params:
                    model_def["parameters"] = params
                models_list.append(model_def)

            config["models"] = models_list
            config["parameters"] = {
                "normalize": self.normalize,
            }
            if self.method in {MergekitMethod.DARE_TIES, MergekitMethod.DARE_LINEAR}:
                config["parameters"]["int8_mask"] = self.int8_mask

        elif self.method == MergekitMethod.LINEAR:
            # LINEAR: N mallia, weight
            models_list = []
            for i, model in enumerate(self.models):
                model_def: Dict[str, Any] = {"model": str(model)}
                if self.model_weights and i < len(self.model_weights):
                    model_def["parameters"] = {"weight": self.model_weights[i]}
                else:
                    model_def["parameters"] = {"weight": 1.0 / len(self.models)}
                models_list.append(model_def)
            config["models"] = models_list
            config["parameters"] = {"normalize": self.normalize}

        elif self.method == MergekitMethod.DELLA:
            # DELLA: base_model + N mallia
            if self.base_model:
                config["base_model"] = str(self.base_model)
                config["tokenizer_source"] = "base"

            models_list = []
            for i, model in enumerate(self.models):
                model_def: Dict[str, Any] = {"model": str(model)}
                params: Dict[str, Any] = {}
                if self.model_weights and i < len(self.model_weights):
                    params["weight"] = self.model_weights[i]
                if self.model_densities and i < len(self.model_densities):
                    params["density"] = self.model_densities[i]
                if params:
                    model_def["parameters"] = params
                models_list.append(model_def)
            config["models"] = models_list
            config["parameters"] = {
                "normalize": self.normalize,
                "rescale": self.rescale,
            }

        else:
            # Oletus: lista malleista
            config["models"] = [{"model": str(m)} for m in self.models]

        return config

    def to_yaml(self) -> str:
        """Palauta YAML-merkkijono."""
        return yaml.dump(self.to_yaml_dict(), default_flow_style=False, allow_unicode=True)


class MergekitWrapper:
    """
    Wrapper mergekit-kirjastolle.

    Tarjoaa helpon API:n mallien yhdistamiseen optimoiduilla
    oletusasetuksilla (10GB VRAM yhteensopiva).
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Alusta wrapper."""
        paths = get_paths()
        self.output_dir = output_dir or paths.models_dir / "merged"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = paths.root / "config" / "merge_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._mergekit_available: Optional[bool] = None

    def check_mergekit(self) -> Tuple[bool, str]:
        """
        Tarkista mergekit-asennus.

        Returns:
            (available, message)
        """
        if self._mergekit_available is not None:
            return self._mergekit_available, ""

        try:
            import mergekit
            self._mergekit_available = True
            return True, f"mergekit {getattr(mergekit, '__version__', 'installed')}"
        except ImportError:
            self._mergekit_available = False
            return False, "mergekit ei asennettu. Asenna: pip install mergekit"

    def install_mergekit(self, progress_callback: Optional[Callable] = None) -> bool:
        """Asenna mergekit."""
        try:
            if progress_callback:
                progress_callback("Asennetaan mergekit...")

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "mergekit==0.1.4", "-q"],
                check=True,
                capture_output=True,
            )
            self._mergekit_available = True
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Mergekit asennus epaonnistui: {e}[/red]")
            return False

    def get_available_methods(self) -> List[Dict[str, Any]]:
        """
        Palauta lista tuetuista metodeista.

        Returns:
            Lista: [{"name": str, "value": str, "description": str, "requires_base": bool}]
        """
        methods = [
            {
                "name": "SLERP",
                "value": "slerp",
                "description": "Spherical interpolation (2 mallia)",
                "requires_base": False,
                "max_models": 2,
            },
            {
                "name": "DARE-TIES",
                "value": "dare_ties",
                "description": "Drop And REscale + TIES (2+ mallia)",
                "requires_base": True,
                "max_models": 10,
            },
            {
                "name": "DARE-LINEAR",
                "value": "dare_linear",
                "description": "Drop And REscale + Linear (2+ mallia)",
                "requires_base": True,
                "max_models": 10,
            },
            {
                "name": "TIES",
                "value": "ties",
                "description": "Task vector merge (2+ mallia)",
                "requires_base": True,
                "max_models": 10,
            },
            {
                "name": "Task Arithmetic",
                "value": "task_arithmetic",
                "description": "Additive task vectors (2+ mallia)",
                "requires_base": True,
                "max_models": 10,
            },
            {
                "name": "LINEAR",
                "value": "linear",
                "description": "Painotettu keskiarvo (2+ mallia)",
                "requires_base": False,
                "max_models": 10,
            },
            {
                "name": "DELLA",
                "value": "della",
                "description": "Density + pruning (2+ mallia)",
                "requires_base": True,
                "max_models": 10,
            },
        ]
        return methods

    def detect_vram(self) -> Tuple[float, Dict[str, Any]]:
        """
        Tunnista VRAM ja palauta suositellut asetukset.

        Returns:
            (vram_gb, recommended_options)
        """
        vram_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass

        # Suositellut asetukset VRAM:n mukaan
        if vram_gb < 12:
            options = {
                "cuda": True,
                "low_cpu_memory": True,
                "lazy_unpickle": True,
                "out_shard_size": "4B",
                "description": "Low VRAM mode (< 12GB)",
            }
        elif vram_gb < 24:
            options = {
                "cuda": True,
                "low_cpu_memory": False,
                "lazy_unpickle": True,
                "out_shard_size": "5B",
                "description": "Medium VRAM mode (12-24GB)",
            }
        else:
            options = {
                "cuda": True,
                "low_cpu_memory": False,
                "lazy_unpickle": False,
                "out_shard_size": "5B",
                "description": "High VRAM mode (> 24GB)",
            }

        return vram_gb, options

    def read_model_config(self, model_path: Path) -> Dict[str, Any]:
        """
        Lue mallin config.json ja palauta olennaiset tiedot.

        Returns:
            Dict with model info or empty dict if not found
        """
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"

        if not config_file.exists():
            return {"path": str(model_path), "name": model_path.name, "config_found": False}

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Normalisoi rope_scaling vertailua varten
            rope_scaling = config.get("rope_scaling")
            rope_type = None
            if rope_scaling:
                rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "custom"

            return {
                "path": str(model_path),
                "name": model_path.name,
                "config_found": True,
                "architecture": config.get("architectures", ["Unknown"])[0] if config.get("architectures") else "Unknown",
                "hidden_size": config.get("hidden_size", 0),
                "num_layers": config.get("num_hidden_layers", 0),
                "vocab_size": config.get("vocab_size", 0),
                "max_position_embeddings": config.get("max_position_embeddings", 0),
                "rope_scaling": rope_scaling,
                "rope_scaling_type": rope_type,
                "rope_theta": config.get("rope_theta", 0),
            }
        except Exception as e:
            return {"path": str(model_path), "name": model_path.name, "config_found": False, "error": str(e)}

    def check_architecture_compatibility(self, model_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tarkista mallien arkkitehtuurien yhteensopivuus.

        Returns:
            {
                "identical": bool,  # Täysin identtiset arkkitehtuurit
                "compatible_slerp": bool,  # SLERP toimii (sama tensor shape)
                "compatible_dare": bool,  # DARE-TIES/LINEAR toimii (identtiset)
                "issues": List[str],  # Ongelmat
                "warnings": List[str],  # Varoitukset
                "recommendation": str,  # Suositeltu metodi
            }
        """
        issues = []
        warnings = []

        if len(model_infos) < 2:
            return {"identical": False, "compatible_slerp": False, "compatible_dare": False,
                    "issues": ["Vahintaan 2 mallia vaaditaan"], "warnings": [], "recommendation": None}

        # Kerää arvot vertailuun
        hidden_sizes = [m.get("hidden_size") for m in model_infos if m.get("hidden_size")]
        num_layers = [m.get("num_layers") for m in model_infos if m.get("num_layers")]
        vocab_sizes = [m.get("vocab_size") for m in model_infos if m.get("vocab_size")]
        max_positions = [m.get("max_position_embeddings") for m in model_infos if m.get("max_position_embeddings")]
        rope_types = [m.get("rope_scaling_type") for m in model_infos]

        # Kriittiset: hidden_size ja num_layers PITÄÄ olla samat
        compatible_slerp = True
        compatible_dare = True

        if len(set(hidden_sizes)) > 1:
            issues.append(f"Eri hidden_size: {hidden_sizes} - mallit eivat ole yhteensopivia!")
            compatible_slerp = False
            compatible_dare = False

        if len(set(num_layers)) > 1:
            issues.append(f"Eri kerrosmaarat: {num_layers} - mallit eivat ole yhteensopivia!")
            compatible_slerp = False
            compatible_dare = False

        # vocab_size: varoitus mutta SLERP voi toimia
        vocab_sizes_valid = [v for v in vocab_sizes if v is not None and v > 0]
        if len(set(vocab_sizes_valid)) > 1:
            warnings.append(f"Eri vocab_size: {vocab_sizes_valid}")
            # Iso ero on ongelma
            if vocab_sizes_valid and max(vocab_sizes_valid) - min(vocab_sizes_valid) > 100:
                issues.append(f"Suuri vocab_size ero ({max(vocab_sizes_valid) - min(vocab_sizes_valid)}) voi aiheuttaa ongelmia")
                compatible_dare = False

        # max_position_embeddings ja rope_scaling: kriittinen DARE:lle
        positions_match = len(set(max_positions)) <= 1
        rope_match = len(set(rope_types)) <= 1

        if not positions_match:
            warnings.append(f"Eri max_position_embeddings: {max_positions}")
            compatible_dare = False

        if not rope_match:
            rope_desc = []
            for m in model_infos:
                name = m.get('name') or '?'
                rope_type = m.get('rope_scaling_type') or 'null'
                rope_desc.append(f"{name[:20]}={rope_type}")
            warnings.append(f"Eri rope_scaling: {', '.join(rope_desc)}")
            compatible_dare = False

        # Onko täysin identtinen?
        identical = (
            len(set(hidden_sizes)) <= 1 and
            len(set(num_layers)) <= 1 and
            len(set(vocab_sizes)) <= 1 and
            positions_match and
            rope_match
        )

        # Suositus
        recommendation = None
        if identical:
            recommendation = "Kaikki metodit toimivat. DARE-TIES suositeltu parhaaseen laatuun."
        elif compatible_slerp and not compatible_dare:
            recommendation = "Kayta SLERP tai LINEAR - DARE-TIES ei toimi eri arkkitehtuureilla!"
        elif compatible_slerp:
            recommendation = "SLERP suositeltu."
        else:
            recommendation = "Mallit eivat ole yhteensopivia mergeen."

        return {
            "identical": identical,
            "compatible_slerp": compatible_slerp,
            "compatible_dare": compatible_dare,
            "issues": issues,
            "warnings": warnings,
            "recommendation": recommendation,
        }

    def get_compatible_methods(self, model_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Palauta lista yhteensopivista metodeista mallien perusteella.

        Returns:
            Lista: [{"method": MergekitMethod, "compatible": bool, "reason": str}]
        """
        compat = self.check_architecture_compatibility(model_infos)

        methods = []

        # SLERP - toimii jos tensor shape sama
        methods.append({
            "method": MergekitMethod.SLERP,
            "compatible": compat["compatible_slerp"] and len(model_infos) == 2,
            "reason": "Vaatii 2 mallia, sama hidden_size/layers" if compat["compatible_slerp"] else "Eri tensor shape",
            "recommended": compat["compatible_slerp"] and not compat["identical"],
        })

        # LINEAR - toimii jos tensor shape sama
        methods.append({
            "method": MergekitMethod.LINEAR,
            "compatible": compat["compatible_slerp"],
            "reason": "Painotettu keskiarvo" if compat["compatible_slerp"] else "Eri tensor shape",
            "recommended": False,
        })

        # DARE-TIES - vaatii identtiset arkkitehtuurit
        methods.append({
            "method": MergekitMethod.DARE_TIES,
            "compatible": compat["compatible_dare"],
            "reason": "Paras laatu identtisille malleille" if compat["compatible_dare"] else "Vaatii identtiset arkkitehtuurit (max_position, rope_scaling)",
            "recommended": compat["identical"],
        })

        # DARE-LINEAR
        methods.append({
            "method": MergekitMethod.DARE_LINEAR,
            "compatible": compat["compatible_dare"],
            "reason": "Kuten DARE-TIES" if compat["compatible_dare"] else "Vaatii identtiset arkkitehtuurit",
            "recommended": False,
        })

        # TIES
        methods.append({
            "method": MergekitMethod.TIES,
            "compatible": compat["compatible_dare"],
            "reason": "Task vector merge" if compat["compatible_dare"] else "Vaatii identtiset arkkitehtuurit",
            "recommended": False,
        })

        # TASK_ARITHMETIC
        methods.append({
            "method": MergekitMethod.TASK_ARITHMETIC,
            "compatible": compat["compatible_dare"],
            "reason": "Additiiviset task vectorit" if compat["compatible_dare"] else "Vaatii identtiset arkkitehtuurit",
            "recommended": False,
        })

        return methods

    def validate_models(
        self,
        models: List[Path],
        method: MergekitMethod,
        base_model: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Validoi mallien yhteensopivuus valitulle metodille.

        Returns:
            {
                "compatible": bool,
                "errors": List[str],
                "warnings": List[str],
                "model_info": List[Dict],
                "architecture_check": Dict,  # Arkkitehtuurivertailu
                "compatible_methods": List[Dict],  # Yhteensopivat metodit
                "recommendation": str,  # Suositus
            }
        """
        errors: List[str] = []
        warnings: List[str] = []
        model_info: List[Dict[str, Any]] = []

        # Kerää kaikkien mallien tiedot (mukaan lukien base_model)
        all_models = list(models)
        if base_model and base_model not in all_models:
            all_models.append(base_model)

        # Lue config.json jokaisesta mallista
        for model in all_models:
            if not model.exists():
                errors.append(f"Mallia ei loydy: {model}")
                continue

            info = self.read_model_config(model)
            model_info.append(info)

            if not info.get("config_found"):
                warnings.append(f"config.json puuttuu: {model.name}")

        # Tarkista arkkitehtuurien yhteensopivuus
        arch_check = self.check_architecture_compatibility(model_info)
        compatible_methods = self.get_compatible_methods(model_info)

        # Lisää arkkitehtuurivaroitukset
        for issue in arch_check["issues"]:
            errors.append(issue)
        for warning in arch_check["warnings"]:
            warnings.append(warning)

        # Tarkista mallien maara
        if method in TWO_MODEL_METHODS and len(models) != 2:
            errors.append(f"{method.value} vaatii tasan 2 mallia, annettu {len(models)}")

        if len(models) < 2:
            errors.append("Vahintaan 2 mallia vaaditaan")

        # Tarkista base model
        if method in METHODS_REQUIRING_BASE and not base_model:
            errors.append(f"{method.value} vaatii base modelin")

        # Tarkista onko valittu metodi yhteensopiva
        method_compat = next((m for m in compatible_methods if m["method"] == method), None)
        if method_compat and not method_compat["compatible"]:
            errors.append(f"{method.value} ei ole yhteensopiva naille malleille: {method_compat['reason']}")

            # Ehdota vaihtoehtoista metodia
            alternatives = [m for m in compatible_methods if m["compatible"]]
            if alternatives:
                alt_names = [m["method"].value.upper() for m in alternatives[:3]]
                warnings.append(f"Kokeile sen sijaan: {', '.join(alt_names)}")

        return {
            "compatible": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "model_info": model_info,
            "architecture_check": arch_check,
            "compatible_methods": compatible_methods,
            "recommendation": arch_check["recommendation"],
        }

    def create_config(
        self,
        method: MergekitMethod,
        models: List[Path],
        output_name: str,
        base_model: Optional[Path] = None,
        **kwargs,
    ) -> MergekitConfig:
        """
        Luo merge-konfiguraatio.

        Args:
            method: Merge-metodi
            models: Lista malleista
            output_name: Tulosteen nimi
            base_model: Base model (TIES/DARE)
            **kwargs: Metodikohtaiset parametrit

        Returns:
            MergekitConfig valmis ajettavaksi
        """
        output_path = self.output_dir / output_name

        # VRAM-optimoinnit
        _, vram_options = self.detect_vram()

        config = MergekitConfig(
            method=method,
            models=models,
            output_path=output_path,
            base_model=base_model,
            cuda=vram_options.get("cuda", True),
            low_cpu_memory=vram_options.get("low_cpu_memory", True),
            lazy_unpickle=vram_options.get("lazy_unpickle", True),
            out_shard_size=vram_options.get("out_shard_size", "5B"),
        )

        # Metodikohtaiset parametrit
        if "slerp_t" in kwargs:
            config.slerp_t = kwargs["slerp_t"]
        if "t" in kwargs:
            config.slerp_t = kwargs["t"]
        if "density" in kwargs:
            config.ties_density = kwargs["density"]
        if "weights" in kwargs:
            config.model_weights = kwargs["weights"]
        if "densities" in kwargs:
            config.model_densities = kwargs["densities"]
        if "normalize" in kwargs:
            config.normalize = kwargs["normalize"]
        if "dtype" in kwargs:
            config.dtype = kwargs["dtype"]
        if "tokenizer_source" in kwargs:
            config.tokenizer_source = kwargs["tokenizer_source"]

        return config

    def merge(
        self,
        config: MergekitConfig,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Suorita merge.

        Args:
            config: Merge-konfiguraatio
            progress_callback: Callback(message)

        Returns:
            {"success": bool, "output_path": str, "error": str, ...}
        """
        available, msg = self.check_mergekit()
        if not available:
            return {"success": False, "error": msg}

        try:
            # Luo valiaikainen YAML-tiedosto
            yaml_content = config.to_yaml()

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.yaml',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(yaml_content)
                yaml_path = f.name

            if progress_callback:
                progress_callback(f"Kaytetaan konfiguraatiota: {config.method.value}")
                progress_callback(f"Mallit: {len(config.models)} kpl")

            # Rakenna mergekit-komento
            cmd = [
                sys.executable, "-m", "mergekit.scripts.run_yaml",
                yaml_path,
                str(config.output_path),
            ]

            if config.cuda:
                cmd.append("--cuda")
            if config.low_cpu_memory:
                cmd.append("--low-cpu-memory")
            if config.lazy_unpickle:
                cmd.append("--lazy-unpickle")
            if config.out_shard_size:
                cmd.extend(["--out-shard-size", config.out_shard_size])

            if progress_callback:
                progress_callback("Kaynnistetaan mergekit...")
                progress_callback(f"Output: {config.output_path}")

            # Suorita
            # encoding pakotettava: Windowsin cp1252-oletus kaatuisi
            # mergekitin unicode-tulosteeseen kesken pitkan ajon
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
            )

            # Lue output
            output_lines = []
            if process.stdout:
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        output_lines.append(line)
                        if progress_callback:
                            # Nayta vain merkitykselliset rivit
                            if any(kw in line.lower() for kw in
                                   ['loading', 'merging', 'saving', 'writing', 'done', 'error', '%']):
                                progress_callback(line[:100])

            return_code = process.wait()

            # Siivoa valiaikainen tiedosto
            Path(yaml_path).unlink(missing_ok=True)

            if return_code != 0:
                error_msg = "\n".join(output_lines[-10:]) if output_lines else "Unknown error"
                return {"success": False, "error": f"Mergekit error (code {return_code}): {error_msg}"}

            # Laske output-koko
            output_size = 0
            if config.output_path.exists():
                output_size = sum(
                    f.stat().st_size for f in config.output_path.rglob("*") if f.is_file()
                )

            return {
                "success": True,
                "output_path": str(config.output_path),
                "method": config.method.value,
                "file_size_gb": round(output_size / (1024**3), 2),
                "num_models": len(config.models),
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def merge_from_yaml(
        self,
        yaml_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Suorita merge YAML-tiedostosta."""
        available, msg = self.check_mergekit()
        if not available:
            return {"success": False, "error": msg}

        if not yaml_path.exists():
            return {"success": False, "error": f"YAML-tiedostoa ei loydy: {yaml_path}"}

        try:
            # VRAM-optimoinnit
            _, vram_options = self.detect_vram()

            cmd = [
                sys.executable, "-m", "mergekit.scripts.run_yaml",
                str(yaml_path),
                str(output_path),
            ]

            if vram_options.get("cuda", True):
                cmd.append("--cuda")
            if vram_options.get("low_cpu_memory", True):
                cmd.append("--low-cpu-memory")
            if vram_options.get("lazy_unpickle", True):
                cmd.append("--lazy-unpickle")

            if progress_callback:
                progress_callback(f"Suoritetaan: {yaml_path.name}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
            )

            if process.stdout:
                for line in process.stdout:
                    if progress_callback and line.strip():
                        progress_callback(line.strip()[:100])

            return_code = process.wait()

            if return_code != 0:
                return {"success": False, "error": f"Mergekit error (code {return_code})"}

            output_size = sum(
                f.stat().st_size for f in output_path.rglob("*") if f.is_file()
            ) if output_path.exists() else 0

            return {
                "success": True,
                "output_path": str(output_path),
                "file_size_gb": round(output_size / (1024**3), 2),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def merge_from_dict(
        self,
        config_dict: Dict[str, Any],
        output_path: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Suorita merge dict-konfiguraatiosta.

        Args:
            config_dict: Merge-konfiguraatio (YAML dict)
            output_path: Output-polku (jos ei ole config_dictissä)
            progress_callback: Progress callback

        Returns:
            Merge-tulos
        """
        available, msg = self.check_mergekit()
        if not available:
            return {"success": False, "error": msg}

        try:
            # Maarittele output path
            if output_path is None:
                if "output_path" in config_dict:
                    output_path = Path(config_dict["output_path"])
                else:
                    # Generoi nimi
                    method = config_dict.get("merge_method", "merge")
                    output_path = self.output_dir / f"{method}_{len(config_dict.get('models', []))}models"

            # Luo valiaikainen YAML-tiedosto
            yaml_content = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.yaml',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(yaml_content)
                yaml_path = Path(f.name)

            # Kayta merge_from_yaml
            result = self.merge_from_yaml(yaml_path, output_path, progress_callback)

            # Siivoa
            yaml_path.unlink(missing_ok=True)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Convenience metodit
    # =========================================================================

    def slerp_merge(
        self,
        model1: Path,
        model2: Path,
        t: float = 0.5,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        SLERP merge kahdelle mallille.

        Args:
            model1: Ensimmainen malli
            model2: Toinen malli
            t: Interpolointi (0.0 = model1, 1.0 = model2)
            output_name: Tulosteen nimi
            progress_callback: Progress callback

        Returns:
            Merge-tulos
        """
        if not output_name:
            output_name = f"slerp_{model1.name[:15]}_{model2.name[:15]}_t{t:.2f}"

        config = self.create_config(
            method=MergekitMethod.SLERP,
            models=[model1, model2],
            output_name=output_name,
            slerp_t=t,
            **kwargs,
        )

        return self.merge(config, progress_callback)

    def dare_ties_merge(
        self,
        models: List[Path],
        base_model: Path,
        weights: Optional[List[float]] = None,
        density: float = 0.5,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        DARE-TIES merge usealle mallille.

        Args:
            models: Lista malleista
            base_model: Base model
            weights: Mallien painot
            density: DARE density (0.0-1.0)
            output_name: Tulosteen nimi
            progress_callback: Progress callback

        Returns:
            Merge-tulos
        """
        if not output_name:
            output_name = f"dare_ties_{len(models)}models_d{density:.2f}"

        config = self.create_config(
            method=MergekitMethod.DARE_TIES,
            models=models,
            output_name=output_name,
            base_model=base_model,
            density=density,
            weights=weights,
            **kwargs,
        )

        return self.merge(config, progress_callback)

    def linear_merge(
        self,
        models: List[Path],
        weights: Optional[List[float]] = None,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Linear (painotettu keskiarvo) merge.

        Args:
            models: Lista malleista
            weights: Mallien painot
            output_name: Tulosteen nimi
            progress_callback: Progress callback

        Returns:
            Merge-tulos
        """
        if not output_name:
            output_name = f"linear_{len(models)}models"

        config = self.create_config(
            method=MergekitMethod.LINEAR,
            models=models,
            output_name=output_name,
            weights=weights,
            **kwargs,
        )

        return self.merge(config, progress_callback)

    def task_arithmetic_merge(
        self,
        models: List[Path],
        base_model: Path,
        weights: Optional[List[float]] = None,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Task Arithmetic merge.

        Args:
            models: Lista malleista (task vectors)
            base_model: Base model
            weights: Task vector painot
            output_name: Tulosteen nimi
            progress_callback: Progress callback

        Returns:
            Merge-tulos
        """
        if not output_name:
            output_name = f"task_arithmetic_{len(models)}models"

        config = self.create_config(
            method=MergekitMethod.TASK_ARITHMETIC,
            models=models,
            output_name=output_name,
            base_model=base_model,
            weights=weights,
            **kwargs,
        )

        return self.merge(config, progress_callback)
