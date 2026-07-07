"""
AI TOOLBOX - LoRA Trainer
=========================

Fine-tuning työkalu LoRA-adaptereille.
Tukee PEFT/transformers -ekosysteemiä ja automaattista Unsloth-kiihdytystä.

Unsloth-integraatio:
- Automaattinen yhteensopivuustarkistus (GPU, malli, asetukset)
- 2-5x nopeampi training, 50-70% vähemmän VRAM
- Fallback tavalliseen PEFT:iin jos yhteensopivuusongelmia
"""

import json
import csv
import sys
import time
import shutil
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..core.ui import console

from ..core.paths import get_paths



# =============================================================================
# UNSLOTH COMPATIBILITY SYSTEM
# =============================================================================

# Unslothin tukemat malliarkkitehtuurit (päivitetty 2024)
UNSLOTH_SUPPORTED_ARCHITECTURES = {
    "llama": {"min_version": "0.0.1", "optimized": True},
    "mistral": {"min_version": "0.0.1", "optimized": True},
    "qwen": {"min_version": "0.1.0", "optimized": True},
    "qwen2": {"min_version": "0.1.0", "optimized": True},
    "phi": {"min_version": "0.2.0", "optimized": True},
    "phi3": {"min_version": "0.2.0", "optimized": True},
    "gemma": {"min_version": "0.1.5", "optimized": True},
    "gemma2": {"min_version": "0.2.0", "optimized": True},
    "yi": {"min_version": "0.1.0", "optimized": False},
    "deepseek": {"min_version": "0.2.0", "optimized": False},
    "olmo": {"min_version": "0.2.0", "optimized": False},
    "cohere": {"min_version": "0.2.0", "optimized": False},
}

# GPU-vaatimukset
UNSLOTH_GPU_REQUIREMENTS = {
    "min_compute_capability": (7, 5),  # Turing (RTX 20xx) minimum
    "recommended_compute_capability": (8, 0),  # Ampere (RTX 30xx) suositeltu
    "min_vram_gb": 6,  # Minimi VRAM
    "recommended_vram_gb": 12,  # Suositeltu VRAM
}


@dataclass
class UnslothCompatibilityResult:
    """Unsloth-yhteensopivuustarkistuksen tulos."""

    # Kokonaistila
    compatible: bool = False
    recommended: bool = False

    # Yksityiskohtaiset tulokset
    unsloth_installed: bool = False
    unsloth_version: Optional[str] = None

    # Laitteisto
    gpu_compatible: bool = False
    gpu_name: Optional[str] = None
    gpu_compute_capability: Optional[Tuple[int, int]] = None
    gpu_vram_gb: float = 0

    # OS
    os_compatible: bool = False
    os_name: str = ""
    os_warning: Optional[str] = None

    # Malli
    model_compatible: bool = False
    model_architecture: Optional[str] = None
    model_optimized: bool = False

    # Asetukset
    settings_compatible: bool = False
    settings_issues: List[str] = field(default_factory=list)

    # Resurssit
    resources_sufficient: bool = False
    estimated_vram_unsloth: float = 0
    estimated_vram_peft: float = 0
    vram_savings_percent: float = 0

    # Ongelmat ja varoitukset
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Suositukset
    recommendations: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Palauta yhteenveto suomeksi."""
        if not self.unsloth_installed:
            return "Unsloth ei asennettu"
        if not self.compatible:
            return f"Ei yhteensopiva: {self.errors[0] if self.errors else 'tuntematon syy'}"
        if self.recommended:
            savings = f"~{self.vram_savings_percent:.0f}% VRAM-säästö"
            return f"Suositeltu! {savings}"
        return "Yhteensopiva (varoituksia)"


class UnslothCompatibilityChecker:
    """
    Kattava Unsloth-yhteensopivuustarkistin.

    Tarkistaa:
    1. Unsloth-asennus ja versio
    2. GPU-yhteensopivuus (arkkitehtuuri, VRAM)
    3. Käyttöjärjestelmä (Linux/Windows)
    4. Malliarkkitehtuuri
    5. LoRA-asetukset
    6. Resurssit (VRAM vs. arvioitu tarve)
    """

    def __init__(self):
        """Alusta tarkistin."""
        self._unsloth_available = False
        self._unsloth_version = None
        self._check_unsloth_installation()

    def _check_unsloth_installation(self):
        """Tarkista Unsloth-asennus."""
        try:
            import unsloth
            self._unsloth_available = True
            self._unsloth_version = getattr(unsloth, "__version__", "unknown")
        except ImportError:
            self._unsloth_available = False
            self._unsloth_version = None

    def check_full_compatibility(
        self,
        model_path: Optional[Path] = None,
        lora_config: Optional['LoRAConfig'] = None,
        training_config: Optional['TrainingConfig'] = None,
        quantization: Optional[str] = None,
    ) -> UnslothCompatibilityResult:
        """
        Suorita täysi yhteensopivuustarkistus.

        Args:
            model_path: Polku malliin (config.json:n lukemiseen)
            lora_config: LoRA-konfiguraatio
            training_config: Training-konfiguraatio
            quantization: Kvantisointityyppi ("4bit", "8bit", None)

        Returns:
            UnslothCompatibilityResult: Täydelliset tulokset
        """
        result = UnslothCompatibilityResult()

        # 1. Tarkista Unsloth-asennus
        result.unsloth_installed = self._unsloth_available
        result.unsloth_version = self._unsloth_version

        if not self._unsloth_available:
            result.errors.append("Unsloth ei ole asennettu")
            result.recommendations.append("Asenna: pip install unsloth")
            return result

        # 2. Tarkista GPU
        self._check_gpu_compatibility(result)

        # 3. Tarkista käyttöjärjestelmä
        self._check_os_compatibility(result)

        # 4. Tarkista malliarkkitehtuuri (jos annettu)
        if model_path:
            self._check_model_compatibility(result, model_path)
        else:
            result.model_compatible = True  # Oleta yhteensopiva jos ei tietoa

        # 5. Tarkista LoRA-asetukset (jos annettu)
        if lora_config or training_config:
            self._check_settings_compatibility(result, lora_config, training_config, quantization)
        else:
            result.settings_compatible = True

        # 6. Tarkista resurssit
        if model_path:
            self._check_resource_requirements(result, model_path, lora_config, quantization)
        else:
            result.resources_sufficient = True

        # 7. Määritä kokonaistila
        result.compatible = (
            result.unsloth_installed and
            result.gpu_compatible and
            result.os_compatible and
            result.model_compatible and
            result.settings_compatible and
            result.resources_sufficient
        )

        # Suositeltu jos yhteensopiva JA ei vakavia varoituksia
        result.recommended = (
            result.compatible and
            len(result.warnings) <= 2 and
            result.model_optimized
        )

        # Lisää suosituksia
        if result.compatible and not result.recommended:
            if not result.model_optimized:
                result.recommendations.append(
                    "Malli toimii Unslothilla, mutta ei ole täysin optimoitu"
                )

        return result

    def _check_gpu_compatibility(self, result: UnslothCompatibilityResult):
        """Tarkista GPU-yhteensopivuus."""
        try:
            import torch

            if not torch.cuda.is_available():
                result.gpu_compatible = False
                result.errors.append("CUDA ei saatavilla - Unsloth vaatii NVIDIA GPU:n")
                return

            result.gpu_name = torch.cuda.get_device_name(0)

            # Compute capability
            capability = torch.cuda.get_device_capability(0)
            result.gpu_compute_capability = capability

            min_cap = UNSLOTH_GPU_REQUIREMENTS["min_compute_capability"]
            rec_cap = UNSLOTH_GPU_REQUIREMENTS["recommended_compute_capability"]

            if capability < min_cap:
                result.gpu_compatible = False
                result.errors.append(
                    f"GPU liian vanha: SM {capability[0]}.{capability[1]}, "
                    f"vaaditaan vähintään SM {min_cap[0]}.{min_cap[1]} (Turing)"
                )
                return

            if capability < rec_cap:
                result.warnings.append(
                    f"GPU SM {capability[0]}.{capability[1]} toimii, mutta "
                    f"SM {rec_cap[0]}.{rec_cap[1]}+ (Ampere) on optimaalinen"
                )

            # VRAM
            props = torch.cuda.get_device_properties(0)
            result.gpu_vram_gb = props.total_memory / (1024**3)

            min_vram = UNSLOTH_GPU_REQUIREMENTS["min_vram_gb"]
            rec_vram = UNSLOTH_GPU_REQUIREMENTS["recommended_vram_gb"]

            if result.gpu_vram_gb < min_vram:
                result.gpu_compatible = False
                result.errors.append(
                    f"Liian vähän VRAM: {result.gpu_vram_gb:.1f} GB, "
                    f"vaaditaan vähintään {min_vram} GB"
                )
                return

            if result.gpu_vram_gb < rec_vram:
                result.warnings.append(
                    f"VRAM {result.gpu_vram_gb:.1f} GB riittää pienille malleille. "
                    f"Suositus: {rec_vram}+ GB"
                )

            result.gpu_compatible = True

        except Exception as e:
            result.gpu_compatible = False
            result.errors.append(f"GPU-tarkistus epäonnistui: {e}")

    def _check_os_compatibility(self, result: UnslothCompatibilityResult):
        """Tarkista käyttöjärjestelmäyhteensopivuus."""
        result.os_name = platform.system()

        if result.os_name == "Linux":
            result.os_compatible = True
        elif result.os_name == "Windows":
            result.os_compatible = True
            result.os_warning = (
                "Windows-tuki on kokeellinen. "
                "Linux on suositeltu Unslothille."
            )
            result.warnings.append(result.os_warning)
        elif result.os_name == "Darwin":  # macOS
            result.os_compatible = False
            result.errors.append(
                "macOS ei ole tuettu - Unsloth vaatii CUDA:n"
            )
        else:
            result.os_compatible = False
            result.errors.append(f"Tuntematon käyttöjärjestelmä: {result.os_name}")

    def _check_model_compatibility(self, result: UnslothCompatibilityResult, model_path: Path):
        """Tarkista malliarkkitehtuurin yhteensopivuus."""
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"

        if not config_file.exists():
            result.model_compatible = True  # Oleta yhteensopiva
            result.warnings.append("Mallin config.json ei löytynyt - yhteensopivuutta ei voida varmistaa")
            return

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            model_type = config.get("model_type", "").lower()
            result.model_architecture = model_type

            # Tarkista tuetut arkkitehtuurit
            for arch_name, arch_info in UNSLOTH_SUPPORTED_ARCHITECTURES.items():
                if arch_name in model_type:
                    result.model_compatible = True
                    result.model_optimized = arch_info["optimized"]

                    if not arch_info["optimized"]:
                        result.warnings.append(
                            f"Malli '{model_type}' toimii, mutta ei ole täysin optimoitu Unslothille"
                        )
                    return

            # Ei löytynyt tuetuista
            result.model_compatible = False
            result.errors.append(
                f"Malliarkkitehtuuri '{model_type}' ei ole Unslothin tukema. "
                f"Tuetut: {', '.join(UNSLOTH_SUPPORTED_ARCHITECTURES.keys())}"
            )

        except Exception as e:
            result.model_compatible = True
            result.warnings.append(f"Mallin config.json:n luku epäonnistui: {e}")

    def _check_settings_compatibility(
        self,
        result: UnslothCompatibilityResult,
        lora_config: Optional['LoRAConfig'],
        training_config: Optional['TrainingConfig'],
        quantization: Optional[str],
    ):
        """Tarkista asetusten yhteensopivuus Unslothin kanssa."""
        result.settings_compatible = True
        issues = []

        if lora_config:
            # Unsloth suosii tiettyjä rank-arvoja
            if lora_config.rank > 256:
                issues.append(f"LoRA rank {lora_config.rank} on suuri - Unsloth toimii parhaiten rank <= 128")

            # Target modules - Unsloth käyttää omia optimoituja moduleita
            # Tämä on vain informatiivinen
            if lora_config.target_modules:
                pass  # Unsloth valitsee automaattisesti optimaaliset

        if training_config:
            # Gradient checkpointing - Unsloth käyttää omaa toteutusta
            if not training_config.gradient_checkpointing:
                result.recommendations.append(
                    "Gradient checkpointing suositellaan Unslothin kanssa"
                )

        if quantization:
            if quantization not in ["4bit", "8bit"]:
                issues.append(f"Tuntematon kvantisointityyppi: {quantization}")

        if issues:
            result.settings_issues = issues
            for issue in issues:
                result.warnings.append(issue)

    def _check_resource_requirements(
        self,
        result: UnslothCompatibilityResult,
        model_path: Path,
        lora_config: Optional['LoRAConfig'],
        quantization: Optional[str],
    ):
        """Arvioi resurssitarpeet ja vertaa Unsloth vs PEFT."""
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"

        # Oletus: 7B malli
        params_b = 7.0

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                vocab_size = config.get("vocab_size", 32000)

                # Karkea arvio parametreista
                params = 12 * num_layers * hidden_size * hidden_size + 2 * vocab_size * hidden_size
                params_b = params / 1e9
            except Exception:
                pass

        # VRAM-arviot (karkeita, riippuu monesta tekijästä)
        # PEFT + 4bit: ~1.2 bytes per param + overhead
        # Unsloth + 4bit: ~0.7 bytes per param + overhead

        if quantization == "4bit":
            result.estimated_vram_peft = params_b * 1.2 + 2  # +2 GB overhead
            result.estimated_vram_unsloth = params_b * 0.7 + 1.5  # +1.5 GB overhead
        elif quantization == "8bit":
            result.estimated_vram_peft = params_b * 1.5 + 2
            result.estimated_vram_unsloth = params_b * 1.0 + 1.5
        else:  # Full precision
            result.estimated_vram_peft = params_b * 4 + 4
            result.estimated_vram_unsloth = params_b * 2.5 + 3

        # LoRA rank vaikuttaa
        if lora_config and lora_config.rank > 32:
            extra = (lora_config.rank - 32) * 0.1  # Lisää VRAM per rank
            result.estimated_vram_peft += extra
            result.estimated_vram_unsloth += extra * 0.5

        # Laske säästö
        if result.estimated_vram_peft > 0:
            savings = (result.estimated_vram_peft - result.estimated_vram_unsloth) / result.estimated_vram_peft * 100
            result.vram_savings_percent = max(0, savings)

        # Tarkista riittääkö VRAM
        if result.gpu_vram_gb > 0:
            if result.estimated_vram_unsloth > result.gpu_vram_gb:
                result.resources_sufficient = False
                result.errors.append(
                    f"Arvioitu VRAM-tarve ({result.estimated_vram_unsloth:.1f} GB) "
                    f"ylittää saatavilla olevan ({result.gpu_vram_gb:.1f} GB)"
                )
                result.recommendations.append(
                    f"Kokeile pienempää mallia tai vahvempaa kvantisointia"
                )
            else:
                result.resources_sufficient = True

                # Varoitus jos tiukka
                headroom = result.gpu_vram_gb - result.estimated_vram_unsloth
                if headroom < 2:
                    result.warnings.append(
                        f"VRAM-marginaali on pieni ({headroom:.1f} GB). "
                        f"Pienennä batch_size jos ongelmia"
                    )
        else:
            result.resources_sufficient = True  # Ei voida tarkistaa

    def get_quick_status(self) -> Dict[str, Any]:
        """Palauta nopea tilatarkistus (ilman mallitietoja)."""
        result = self.check_full_compatibility()

        return {
            "available": self._unsloth_available,
            "version": self._unsloth_version,
            "gpu_ok": result.gpu_compatible,
            "os_ok": result.os_compatible,
            "gpu_name": result.gpu_name,
            "gpu_vram_gb": result.gpu_vram_gb,
            "warnings": result.warnings,
            "errors": result.errors,
        }


class DatasetFormat(Enum):
    """Tuetut dataset-formaatit."""
    JSONL = "jsonl"
    ALPACA = "alpaca"
    CHAT = "chat"
    CSV = "csv"
    SHAREGPT = "sharegpt"


@dataclass
class LoRAConfig:
    """LoRA-konfiguraatio."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"  # "none", "all", "lora_only"


@dataclass
class TrainingConfig:
    """Training-konfiguraatio."""
    # Perusasetukset
    epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4

    # Scheduler
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 50
    logging_steps: int = 10

    # Memory optimization
    gradient_checkpointing: bool = True
    # None = auto-detect GPU:n perusteella, True/False = pakota
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None

    # Dataset
    max_seq_length: int = 2048
    packing: bool = False  # Pack multiple samples into one sequence


@dataclass
class FullConfig:
    """Täydellinen training-konfiguraatio."""
    # Model
    model_path: str = ""
    model_name: str = ""

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Dataset
    dataset_path: str = ""
    dataset_format: str = "alpaca"
    validation_split: float = 0.1

    # Output
    output_dir: str = ""
    run_name: str = ""

    # Advanced
    use_unsloth: bool = False
    quantization: Optional[str] = None  # "4bit", "8bit", None

    def to_dict(self) -> Dict[str, Any]:
        """Muunna dict-muotoon."""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "lora": asdict(self.lora),
            "training": asdict(self.training),
            "dataset_path": self.dataset_path,
            "dataset_format": self.dataset_format,
            "validation_split": self.validation_split,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "use_unsloth": self.use_unsloth,
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FullConfig':
        """Luo dict:stä."""
        config = cls()
        config.model_path = data.get("model_path", "")
        config.model_name = data.get("model_name", "")

        if "lora" in data:
            lora_data = data["lora"]
            config.lora = LoRAConfig(**lora_data)

        if "training" in data:
            training_data = data["training"]
            config.training = TrainingConfig(**training_data)

        config.dataset_path = data.get("dataset_path", "")
        config.dataset_format = data.get("dataset_format", "alpaca")
        config.validation_split = data.get("validation_split", 0.1)
        config.output_dir = data.get("output_dir", "")
        config.run_name = data.get("run_name", "")
        config.use_unsloth = data.get("use_unsloth", False)
        config.quantization = data.get("quantization")

        return config


# Model-specific target modules
TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama_efficient": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "v_proj"],
}

# Recommended configs by model size
PRESET_CONFIGS = {
    "small": {  # <3B parameters
        "rank": 32,
        "alpha": 64,
        "batch_size": 8,
        "learning_rate": 2e-4,
    },
    "medium": {  # 3B-13B parameters
        "rank": 16,
        "alpha": 32,
        "batch_size": 4,
        "learning_rate": 1e-4,
    },
    "large": {  # >13B parameters
        "rank": 8,
        "alpha": 16,
        "batch_size": 2,
        "learning_rate": 5e-5,
    },
}


class LoRATrainer:
    """LoRA fine-tuning työkalu."""

    def __init__(self):
        """Alusta trainer."""
        paths = get_paths()
        self.output_dir = paths.models_dir / "lora"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_dir = paths.root / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.configs_dir = paths.config_dir / "lora_configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # Tarkista riippuvuudet
        self._deps = self._check_dependencies()

        # Unsloth-yhteensopivuustarkistin
        self._unsloth_checker = UnslothCompatibilityChecker()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Tarkista riippuvuudet."""
        deps = {
            "torch": False,
            "transformers": False,
            "peft": False,
            "datasets": False,
            "trl": False,
            "bitsandbytes": False,
            "unsloth": False,
        }

        try:
            import torch
            deps["torch"] = True
        except ImportError:
            pass

        try:
            import transformers
            deps["transformers"] = True
        except ImportError:
            pass

        try:
            import peft
            deps["peft"] = True
        except ImportError:
            pass

        try:
            import datasets
            deps["datasets"] = True
        except ImportError:
            pass

        try:
            import trl
            deps["trl"] = True
        except ImportError:
            pass

        try:
            import bitsandbytes
            deps["bitsandbytes"] = True
        except ImportError:
            pass

        try:
            import unsloth
            deps["unsloth"] = True
        except ImportError:
            pass

        return deps

    def get_status(self) -> Dict[str, Any]:
        """Palauta trainerin status."""
        required = ["torch", "transformers", "peft", "datasets"]
        missing = [k for k in required if not self._deps[k]]

        # Unsloth-status
        unsloth_status = self._unsloth_checker.get_quick_status()

        return {
            "ready": len(missing) == 0,
            "dependencies": self._deps,
            "missing_required": missing,
            "output_dir": str(self.output_dir),
            "datasets_dir": str(self.datasets_dir),
            "has_unsloth": self._deps["unsloth"],
            "has_quantization": self._deps["bitsandbytes"],
            # Unsloth-laajennukset
            "unsloth_available": unsloth_status["available"],
            "unsloth_version": unsloth_status["version"],
            "unsloth_gpu_ok": unsloth_status["gpu_ok"],
            "unsloth_os_ok": unsloth_status["os_ok"],
        }

    def check_unsloth_compatibility(
        self,
        config: FullConfig,
    ) -> UnslothCompatibilityResult:
        """
        Tarkista Unsloth-yhteensopivuus annetulle konfiguraatiolle.

        Args:
            config: Täysi training-konfiguraatio

        Returns:
            UnslothCompatibilityResult: Yksityiskohtainen yhteensopivuusraportti
        """
        model_path = Path(config.model_path) if config.model_path else None

        return self._unsloth_checker.check_full_compatibility(
            model_path=model_path,
            lora_config=config.lora,
            training_config=config.training,
            quantization=config.quantization,
        )

    def _resolve_dtype_config(self, config: TrainingConfig) -> Dict[str, Any]:
        """
        Määritä optimaalinen dtype GPU:n perusteella.

        Palauttaa dict:n, jossa:
        - compute_dtype: torch dtype mallille ja quantizationille
        - fp16: bool SFTConfig:lle
        - bf16: bool SFTConfig:lle
        """
        import torch

        # Jos käyttäjä on eksplisiittisesti määrittänyt, käytetään sitä
        if config.fp16 is not None or config.bf16 is not None:
            use_fp16 = config.fp16 if config.fp16 is not None else False
            use_bf16 = config.bf16 if config.bf16 is not None else False

            if use_bf16:
                compute_dtype = torch.bfloat16
            elif use_fp16:
                compute_dtype = torch.float16
            else:
                compute_dtype = torch.float32

            return {
                "compute_dtype": compute_dtype,
                "fp16": use_fp16,
                "bf16": use_bf16,
            }

        # Auto-detect: tarkista GPU:n bf16-tuki
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Ampere (SM 8.0+) ja uudemmat tukevat bf16 tehokkaasti
            # RTX 30xx = SM 8.6, RTX 40xx = SM 8.9
            if capability[0] >= 8:
                return {
                    "compute_dtype": torch.bfloat16,
                    "fp16": False,
                    "bf16": True,
                }
            else:
                # Vanhemmat GPU:t (Turing, Volta, Pascal) - käytä fp16
                return {
                    "compute_dtype": torch.float16,
                    "fp16": True,
                    "bf16": False,
                }

        # Ei GPU:ta - käytä float32
        return {
            "compute_dtype": torch.float32,
            "fp16": False,
            "bf16": False,
        }

    def _detect_hardware_capabilities(self) -> Dict[str, Any]:
        """
        Tunnista laitteiston kyvyt automaattisesti.

        Palauttaa dict:n, jossa kaikki tiedot käytettävissä olevista optimoinneista.
        """
        import torch

        caps = {
            # GPU perustiedot
            "cuda_available": False,
            "gpu_name": None,
            "gpu_capability": None,
            "gpu_memory_gb": 0,
            "multi_gpu": False,
            "gpu_count": 0,

            # Tuetut ominaisuudet
            "supports_bf16": False,
            "supports_tf32": False,
            "supports_flash_attn": False,
            "supports_sdpa": False,
            "supports_fused_adamw": False,
            "supports_torch_compile": False,

            # Suositellut asetukset (täytetään alla)
            "recommended_dtype": "float32",
            "recommended_attn": "eager",
            "recommended_optim": "adamw_torch",
            "recommended_workers": 0,
        }

        # Tarkista CUDA
        caps["cuda_available"] = torch.cuda.is_available()

        if caps["cuda_available"]:
            caps["gpu_count"] = torch.cuda.device_count()
            caps["multi_gpu"] = caps["gpu_count"] > 1
            caps["gpu_name"] = torch.cuda.get_device_name(0)

            props = torch.cuda.get_device_properties(0)
            caps["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)

            capability = torch.cuda.get_device_capability()
            caps["gpu_capability"] = f"SM {capability[0]}.{capability[1]}"

            # SM 8.0+ = Ampere (RTX 30xx, A100, etc.)
            # SM 8.9 = Ada Lovelace (RTX 40xx)
            # SM 9.0 = Hopper (H100)
            is_ampere_or_newer = capability[0] >= 8

            # BF16 tuki - Ampere+
            caps["supports_bf16"] = is_ampere_or_newer

            # TF32 tuki - Ampere+
            caps["supports_tf32"] = is_ampere_or_newer

            # SDPA (Scaled Dot Product Attention) - PyTorch 2.0+
            caps["supports_sdpa"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Flash Attention 2 - tarkista saatavuus
        try:
            from transformers.utils import is_flash_attn_2_available
            caps["supports_flash_attn"] = is_flash_attn_2_available()
        except ImportError:
            caps["supports_flash_attn"] = False

        # torch.compile (PyTorch 2.0+)
        caps["supports_torch_compile"] = hasattr(torch, "compile")

        # Fused AdamW
        try:
            import inspect
            from torch.optim import AdamW
            caps["supports_fused_adamw"] = (
                "fused" in inspect.signature(AdamW).parameters
                and caps["cuda_available"]
            )
        except Exception:
            caps["supports_fused_adamw"] = False

        # Määritä suositellut asetukset automaattisesti
        if caps["cuda_available"]:
            # Dtype
            if caps["supports_bf16"]:
                caps["recommended_dtype"] = "bfloat16"
            else:
                caps["recommended_dtype"] = "float16"

            # Attention implementation
            if caps["supports_flash_attn"]:
                caps["recommended_attn"] = "flash_attention_2"
            elif caps["supports_sdpa"]:
                caps["recommended_attn"] = "sdpa"
            else:
                caps["recommended_attn"] = "eager"

            # Optimoija
            if caps["supports_fused_adamw"]:
                caps["recommended_optim"] = "adamw_torch_fused"
            else:
                caps["recommended_optim"] = "adamw_torch"

            # DataLoader workers - CPU ytimien mukaan, max 8
            import os
            cpu_count = os.cpu_count() or 4
            caps["recommended_workers"] = min(cpu_count // 2, 8)

        return caps

    def _apply_gpu_optimizations(self, hw_caps: Dict[str, Any], progress_callback: Optional[Callable] = None) -> None:
        """
        Sovella GPU-optimoinnit tunnistetun laitteiston perusteella.
        """
        import torch

        if not hw_caps["cuda_available"]:
            return

        # TF32 - Ampere+
        if hw_caps["supports_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # cuDNN benchmark - aina hyödyllinen
        torch.backends.cudnn.benchmark = True

        # Logita käytetyt optimoinnit
        if progress_callback:
            opts = []
            if hw_caps["supports_tf32"]:
                opts.append("TF32")
            if hw_caps["recommended_attn"] == "flash_attention_2":
                opts.append("FlashAttn2")
            elif hw_caps["recommended_attn"] == "sdpa":
                opts.append("SDPA")
            if hw_caps["supports_fused_adamw"]:
                opts.append("FusedAdamW")

            opts_str = ", ".join(opts) if opts else "basic"
            progress_callback(f"GPU: {hw_caps['gpu_name']} ({hw_caps['gpu_capability']}) | {opts_str}")

    def install_dependencies(self, include_optional: bool = False,
                            progress_callback: Optional[Callable] = None) -> bool:
        """Asenna riippuvuudet."""
        import subprocess

        packages = []

        # Required
        if not self._deps["torch"]:
            packages.append("torch")
        if not self._deps["transformers"]:
            packages.append("transformers")
        if not self._deps["peft"]:
            packages.append("peft")
        if not self._deps["datasets"]:
            packages.append("datasets")
        if not self._deps["trl"]:
            packages.append("trl")

        # Optional
        if include_optional:
            if not self._deps["bitsandbytes"]:
                packages.append("bitsandbytes")

        if not packages:
            return True

        try:
            for pkg in packages:
                if progress_callback:
                    progress_callback(f"Asennetaan {pkg}...")

                subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    check=True,
                    capture_output=True,
                )

            # Päivitä deps
            self._deps = self._check_dependencies()
            return True

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Asennus epäonnistui: {e}[/red]")
            return False

    # ==================== DATASET HANDLING ====================

    def detect_dataset_format(self, file_path: Path) -> Optional[str]:
        """Tunnista datasetin formaatti."""
        if not file_path.exists():
            return None

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return "csv"

        if suffix in [".json", ".jsonl"]:
            # Lue ensimmäinen rivi
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('['):
                        # JSON array
                        data = json.loads(first_line + f.read())
                        sample = data[0] if data else {}
                    else:
                        sample = json.loads(first_line)

                # Tunnista formaatti sisällön perusteella
                if "messages" in sample:
                    return "chat"
                elif "conversations" in sample:
                    return "sharegpt"
                elif "instruction" in sample:
                    if "input" in sample:
                        return "alpaca"
                    return "instruction"
                elif "text" in sample:
                    return "text"
                elif "prompt" in sample and "completion" in sample:
                    return "completion"

            except Exception:
                pass

        return "unknown"

    def validate_dataset(self, file_path: Path, format_type: str = "auto") -> Dict[str, Any]:
        """Validoi dataset."""
        result = {
            "valid": False,
            "format": None,
            "num_samples": 0,
            "sample": None,
            "errors": [],
            "warnings": [],
        }

        if not file_path.exists():
            result["errors"].append(f"Tiedostoa ei löydy: {file_path}")
            return result

        # Tunnista formaatti
        if format_type == "auto":
            format_type = self.detect_dataset_format(file_path)

        result["format"] = format_type

        if format_type == "unknown":
            result["errors"].append("Tuntematon formaatti")
            return result

        try:
            samples = []

            if format_type == "csv":
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        samples.append(row)
                        if i >= 100:
                            break

                # Tarkista sarakkeet
                if samples:
                    cols = set(samples[0].keys())
                    if not ("instruction" in cols or "text" in cols or "prompt" in cols):
                        result["warnings"].append("Dataset ei sisällä tunnettuja sarakkeita (instruction/text/prompt)")

            else:  # JSON/JSONL
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    if content.startswith('['):
                        # JSON array
                        data = json.loads(content)
                        samples = data[:100]
                    else:
                        # JSONL
                        for i, line in enumerate(content.split('\n')):
                            if line.strip():
                                samples.append(json.loads(line))
                            if i >= 100:
                                break

            result["num_samples"] = len(samples)
            result["sample"] = samples[0] if samples else None

            # Laske kokonaismäärä
            if file_path.suffix.lower() in [".json", ".jsonl"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('['):
                        result["num_samples"] = len(json.loads(content))
                    else:
                        result["num_samples"] = sum(1 for line in content.split('\n') if line.strip())

            # Varoitukset
            if result["num_samples"] < 100:
                result["warnings"].append(f"Pieni dataset ({result['num_samples']} näytettä). Suositus: >500")

            if result["num_samples"] > 100000:
                result["warnings"].append(f"Suuri dataset ({result['num_samples']} näytettä). Harkitse osaotantaa.")

            result["valid"] = True

        except Exception as e:
            result["errors"].append(f"Virhe lukiessa: {str(e)}")

        return result

    def create_sample_dataset(self, output_path: Path, format_type: str = "alpaca",
                             num_samples: int = 10) -> bool:
        """Luo esimerkkidataset."""
        samples = []

        if format_type == "alpaca":
            samples = [
                {
                    "instruction": "Selitä mitä tekoäly tarkoittaa.",
                    "input": "",
                    "output": "Tekoäly (AI) on tietojenkäsittelyn ala, joka keskittyy luomaan järjestelmiä, jotka voivat suorittaa tehtäviä, jotka normaalisti vaatisivat ihmisälyä."
                },
                {
                    "instruction": "Käännä seuraava lause englanniksi.",
                    "input": "Koira juoksee puistossa.",
                    "output": "The dog is running in the park."
                },
                {
                    "instruction": "Kirjoita lyhyt runo aiheesta luonto.",
                    "input": "",
                    "output": "Puut kuiskivat tuulessa,\nlinnut laulavat oksilla.\nAurinko paistaa kirkkaana,\nluonto herää uuteen päivään."
                },
            ] * (num_samples // 3 + 1)
            samples = samples[:num_samples]

        elif format_type == "chat":
            samples = [
                {
                    "messages": [
                        {"role": "user", "content": "Mikä on Pythonin paras ominaisuus?"},
                        {"role": "assistant", "content": "Pythonin paras ominaisuus on sen selkeä ja luettava syntaksi, joka tekee ohjelmoinnista helposti lähestyttävää."}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Kerro vitsi."},
                        {"role": "assistant", "content": "Miksi ohjelmoija käyttää tummaa teemaa? Koska valo houkuttelee bugeja!"}
                    ]
                },
            ] * (num_samples // 2 + 1)
            samples = samples[:num_samples]

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            return True
        except Exception:
            return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Listaa datasets-kansion datasetit."""
        datasets = []

        for file_path in self.datasets_dir.iterdir():
            if file_path.suffix.lower() in ['.json', '.jsonl', '.csv']:
                format_type = self.detect_dataset_format(file_path)
                size = file_path.stat().st_size

                datasets.append({
                    "path": file_path,
                    "name": file_path.name,
                    "format": format_type,
                    "size_bytes": size,
                })

        return datasets

    # ==================== MODEL HANDLING ====================

    def detect_model_type(self, model_path: Path) -> Optional[str]:
        """Tunnista mallin tyyppi config.json:sta."""
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            model_type = config.get("model_type", "").lower()

            # Normalisoi
            if "llama" in model_type:
                return "llama"
            elif "mistral" in model_type:
                return "mistral"
            elif "qwen" in model_type:
                return "qwen"
            elif "phi" in model_type:
                return "phi"
            elif "gemma" in model_type:
                return "gemma"

            return model_type or "unknown"

        except Exception:
            return None

    def get_recommended_config(self, model_path: Path) -> Dict[str, Any]:
        """Hae suositeltu konfiguraatio mallille."""
        model_type = self.detect_model_type(model_path)

        # Laske parametrit
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
        params_b = 7  # Oletus

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                vocab_size = config.get("vocab_size", 32000)

                # Karkea arvio
                params = 12 * num_layers * hidden_size * hidden_size + 2 * vocab_size * hidden_size
                params_b = params / 1e9
            except Exception:
                pass

        # Valitse preset
        if params_b < 3:
            preset = PRESET_CONFIGS["small"]
        elif params_b < 13:
            preset = PRESET_CONFIGS["medium"]
        else:
            preset = PRESET_CONFIGS["large"]

        # Valitse target_modules
        if model_type and model_type in TARGET_MODULES:
            target_modules = TARGET_MODULES[model_type]
        else:
            target_modules = TARGET_MODULES["default"]

        return {
            "model_type": model_type,
            "estimated_params_b": round(params_b, 1),
            "preset": preset,
            "target_modules": target_modules,
        }

    # ==================== TRAINING ====================

    def prepare_training(self, config: FullConfig) -> Dict[str, Any]:
        """Valmistele training."""
        if not self._deps["torch"] or not self._deps["transformers"] or not self._deps["peft"]:
            return {"success": False, "error": "Riippuvuudet puuttuvat"}

        import torch

        # Tarkista GPU
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else 0

        # Arvioi muistivaatimus
        # LoRA: ~2-4GB base + batch_size * seq_len * hidden_size * 4 bytes
        estimated_memory = 4 + config.training.batch_size * 2  # Karkea arvio GB

        return {
            "success": True,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_memory, 1),
            "estimated_memory_gb": estimated_memory,
            "fits_in_memory": gpu_memory > estimated_memory if gpu_available else False,
            "recommended_batch_size": max(1, int(gpu_memory / 3)) if gpu_available else 1,
        }

    def _format_dataset(self, dataset, format_type: str, tokenizer):
        """Formatoi dataset training-muotoon."""
        def format_alpaca(example):
            """Alpaca-formaatti."""
            if example.get("input"):
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            return {"text": text}

        def format_chat(example):
            """Chat-formaatti."""
            messages = example.get("messages", [])
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        def format_sharegpt(example):
            """ShareGPT-formaatti (conversations kenttä)."""
            conversations = example.get("conversations", [])
            # Muunna messages-formaattiin
            messages = []
            for conv in conversations:
                role = "user" if conv.get("from") in ["human", "user"] else "assistant"
                content = conv.get("value", "")
                messages.append({"role": role, "content": content})

            if not messages:
                return {"text": ""}

            # Käytä chat templatea jos saatavilla
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                # Fallback: manuaalinen formatointi
                text_parts = []
                for msg in messages:
                    if msg["role"] == "user":
                        text_parts.append(f"User: {msg['content']}")
                    else:
                        text_parts.append(f"Assistant: {msg['content']}")
                text = "\n\n".join(text_parts)

            return {"text": text}

        def format_completion(example):
            """Prompt-completion formaatti."""
            text = f"{example['prompt']}{example['completion']}"
            return {"text": text}

        def format_text(example):
            """Plain text."""
            return {"text": example.get("text", "")}

        formatters = {
            "alpaca": format_alpaca,
            "instruction": format_alpaca,
            "chat": format_chat,
            "sharegpt": format_sharegpt,
            "completion": format_completion,
            "text": format_text,
        }

        formatter = formatters.get(format_type, format_text)
        return dataset.map(formatter, remove_columns=dataset.column_names)

    def train(
        self,
        config: FullConfig,
        progress_callback: Optional[Callable] = None,
        use_unsloth: Optional[bool] = None,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """
        Suorita training.

        Args:
            config: Täysi training-konfiguraatio
            progress_callback: Callback edistymisen raportointiin
            use_unsloth: Käytä Unsloth-kiihdytystä
                - None: Auto-detect (käytä jos yhteensopiva)
                - True: Pakota Unsloth (fallback PEFT:iin jos epäonnistuu)
                - False: Käytä aina tavallista PEFT
            resume_from_checkpoint: Jatka viimeisimmästä checkpoint-* -kansiosta
                output-kansiossa (jos sellainen on)

        Returns:
            Dict[str, Any]: Tulokset sisältäen success, adapter_path, jne.
        """
        if not self.get_status()["ready"]:
            return {"success": False, "error": "Riippuvuudet puuttuvat"}

        # Määritä käytetäänkö Unslothia
        unsloth_result = None
        should_use_unsloth = False

        if use_unsloth is None:
            # Auto-detect
            unsloth_result = self.check_unsloth_compatibility(config)
            should_use_unsloth = unsloth_result.compatible

            if progress_callback:
                if unsloth_result.compatible:
                    progress_callback(f"Unsloth: {unsloth_result.get_summary()}")
                elif unsloth_result.unsloth_installed:
                    progress_callback(f"Unsloth: Ei yhteensopiva - käytetään PEFT")
        elif use_unsloth is True:
            # Käyttäjä haluaa Unslothin
            unsloth_result = self.check_unsloth_compatibility(config)
            should_use_unsloth = unsloth_result.compatible

            if not should_use_unsloth and progress_callback:
                progress_callback(f"Unsloth ei yhteensopiva: {unsloth_result.errors[0] if unsloth_result.errors else 'tuntematon syy'}")
                progress_callback("Fallback: käytetään tavallista PEFT")
        else:
            # use_unsloth = False
            should_use_unsloth = False
            if progress_callback:
                progress_callback("Unsloth: Ohitettu käyttäjän pyynnöstä")

        # Yritä Unsloth-koulutusta
        if should_use_unsloth:
            if progress_callback:
                savings = f"~{unsloth_result.vram_savings_percent:.0f}%" if unsloth_result else ""
                progress_callback(f"Käytetään Unsloth-kiihdytystä ({savings} VRAM-säästö)")

            try:
                result = self._train_with_unsloth(
                    config, progress_callback,
                    resume_from_checkpoint=resume_from_checkpoint,
                )
                if result["success"]:
                    result["backend"] = "unsloth"
                    return result
                else:
                    # Unsloth epäonnistui - kokeile fallbackia
                    if progress_callback:
                        progress_callback(f"Unsloth epäonnistui: {result.get('error', 'tuntematon')}")
                        progress_callback("Fallback: kokeillaan tavallista PEFT...")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Unsloth-virhe: {e}")
                    progress_callback("Fallback: kokeillaan tavallista PEFT...")

            # Vapauta epäonnistuneen Unsloth-yrityksen GPU-muisti ennen kuin
            # PEFT lataa mallin uudelleen - muuten fallback voi kaatua OOM:iin
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Tavallinen PEFT-koulutus
        if progress_callback and should_use_unsloth:
            progress_callback("Käytetään tavallista PEFT/transformers-koulutusta")
        elif progress_callback:
            progress_callback("Käytetään PEFT/transformers-koulutusta")

        result = self._train_with_peft(
            config, progress_callback,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        result["backend"] = "peft"
        return result

    @staticmethod
    def _resolve_resume_checkpoint(output_dir: Path, resume: bool) -> bool:
        """Palauta True vain jos jatkamista pyydettiin JA checkpoint loytyy.

        HF Trainer kaatuu resume_from_checkpoint=True jos checkpointtia ei ole,
        joten tarkistetaan olemassaolo etukateen.
        """
        if not resume:
            return False
        return any(output_dir.glob("checkpoint-*"))

    def _train_with_unsloth(
        self,
        config: FullConfig,
        progress_callback: Optional[Callable] = None,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """
        Suorita training Unslothilla.

        Unsloth tarjoaa:
        - 2-5x nopeampi training
        - 50-70% vähemmän VRAM
        - Optimoidut Triton-kernelit
        """
        try:
            import torch
            from unsloth import FastLanguageModel
            from datasets import load_dataset
            from trl import SFTTrainer, SFTConfig

            start_time = time.time()

            if progress_callback:
                progress_callback("[Unsloth] Ladataan malli...")

            # Määritä kvantisointityyppi
            load_in_4bit = config.quantization == "4bit"
            load_in_8bit = config.quantization == "8bit"

            # Lataa malli Unslothilla
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_path,
                max_seq_length=config.training.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

            if progress_callback:
                progress_callback("[Unsloth] Valmistellaan LoRA...")

            # Lisää LoRA-adapterit Unslothin optimoidulla metodilla
            # Käytä config.lora.target_modules jos määritelty, muuten Unslothin oletukset
            unsloth_default_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
            target_modules = config.lora.target_modules if config.lora.target_modules else unsloth_default_modules

            model = FastLanguageModel.get_peft_model(
                model,
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unslothin optimoitu GC
                random_state=42,
            )

            # Laske trainable params
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            if progress_callback:
                progress_callback(f"[Unsloth] Trainable: {trainable_params:,} / {total_params:,}")

            # Lataa dataset
            if progress_callback:
                progress_callback("[Unsloth] Ladataan dataset...")

            dataset = load_dataset(
                "json",
                data_files=config.dataset_path,
                split="train",
            )

            # Jaa train/eval
            if config.validation_split > 0:
                split = dataset.train_test_split(test_size=config.validation_split, seed=42)
                train_dataset = split["train"]
                eval_dataset = split["test"]
            else:
                train_dataset = dataset
                eval_dataset = None

            # Formatoi dataset
            train_dataset = self._format_dataset(train_dataset, config.dataset_format, tokenizer)
            if eval_dataset:
                eval_dataset = self._format_dataset(eval_dataset, config.dataset_format, tokenizer)

            # Output-kansio
            output_dir = Path(config.output_dir) if config.output_dir else self.output_dir / config.run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("[Unsloth] Konfiguroidaan trainer...")

            # SFT Config
            sft_config = SFTConfig(
                output_dir=str(output_dir),
                num_train_epochs=config.training.epochs,
                per_device_train_batch_size=config.training.batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                warmup_ratio=config.training.warmup_ratio,
                lr_scheduler_type=config.training.lr_scheduler_type,
                weight_decay=config.training.weight_decay,
                max_grad_norm=config.training.max_grad_norm,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                save_steps=config.training.save_steps,
                save_total_limit=config.training.save_total_limit,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=config.training.eval_steps if eval_dataset else None,
                logging_steps=config.training.logging_steps,
                report_to="none",
                run_name=config.run_name,
                optim="adamw_8bit",  # Unslothin optimoitu
                max_length=config.training.max_seq_length,
                packing=config.training.packing,
                dataset_text_field="text",
            )

            # Trainer
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            # Train!
            resume = self._resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
            if progress_callback:
                if resume:
                    progress_callback("[Unsloth] Jatketaan viimeisimmasta checkpointista...")
                else:
                    progress_callback("[Unsloth] Aloitetaan training...")

            trainer.train(resume_from_checkpoint=resume or None)

            # Tallenna
            if progress_callback:
                progress_callback("[Unsloth] Tallennetaan adapter...")

            adapter_path = output_dir / "adapter"
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            # Tallenna config
            config_path = output_dir / "training_config.json"
            training_info = config.to_dict()
            training_info["backend"] = "unsloth"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time

            return {
                "success": True,
                "output_dir": str(output_dir),
                "adapter_path": str(adapter_path),
                "trainable_params": trainable_params,
                "total_params": total_params,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": f"{int(elapsed//60)}m {int(elapsed%60)}s",
                "backend": "unsloth",
            }

        except Exception as e:
            return {"success": False, "error": str(e), "backend": "unsloth"}

    def _train_with_peft(
        self,
        config: FullConfig,
        progress_callback: Optional[Callable] = None,
        resume_from_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """Suorita training tavallisella PEFT:llä."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import load_dataset
            from trl import SFTTrainer, SFTConfig

            start_time = time.time()

            # 0. Tunnista laitteisto ja sovella optimoinnit automaattisesti
            hw_caps = self._detect_hardware_capabilities()
            self._apply_gpu_optimizations(hw_caps, progress_callback)

            # Määritä dtype - käytä hw_caps suositusta tai käyttäjän valintaa
            if config.training.fp16 is None and config.training.bf16 is None:
                # Auto-detect
                if hw_caps["recommended_dtype"] == "bfloat16":
                    compute_dtype = torch.bfloat16
                    use_fp16, use_bf16 = False, True
                elif hw_caps["recommended_dtype"] == "float16":
                    compute_dtype = torch.float16
                    use_fp16, use_bf16 = True, False
                else:
                    compute_dtype = torch.float32
                    use_fp16, use_bf16 = False, False
            else:
                # Käyttäjän valinta
                dtype_config = self._resolve_dtype_config(config.training)
                compute_dtype = dtype_config["compute_dtype"]
                use_fp16, use_bf16 = dtype_config["fp16"], dtype_config["bf16"]

            if progress_callback:
                dtype_name = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
                progress_callback(f"Dtype: {dtype_name}")

            # 1. Lataa tokenizer
            if progress_callback:
                progress_callback("Ladataan tokenizer...")

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
                if tokenizer.pad_token is None:
                    return {
                        "success": False,
                        "error": "Tokenizerilta puuttuu pad/eos/unk token - "
                                 "batchaus ei toimi ilman pad-tokenia",
                    }

            # 2. Lataa malli
            if progress_callback:
                progress_callback("Ladataan malli...")

            # Quantization config
            bnb_config = None
            if config.quantization == "4bit" and self._deps["bitsandbytes"]:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            elif config.quantization == "8bit" and self._deps["bitsandbytes"]:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            # Mallin latausparametrit - käytä automaattisesti tunnistettua attention-toteutusta
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": compute_dtype,
                "attn_implementation": hw_caps["recommended_attn"],
            }

            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                **model_kwargs,
            )

            # 3. Valmistele LoRA
            if progress_callback:
                progress_callback("Valmistellaan LoRA...")

            if config.quantization:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.dropout,
                bias=config.lora.bias,
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)

            # Laske trainable params
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            if progress_callback:
                progress_callback(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

            # 4. Lataa dataset
            if progress_callback:
                progress_callback("Ladataan dataset...")

            dataset = load_dataset(
                "json",
                data_files=config.dataset_path,
                split="train",
            )

            # Jaa train/eval
            if config.validation_split > 0:
                split = dataset.train_test_split(test_size=config.validation_split, seed=42)
                train_dataset = split["train"]
                eval_dataset = split["test"]
            else:
                train_dataset = dataset
                eval_dataset = None

            # Formatoi
            train_dataset = self._format_dataset(train_dataset, config.dataset_format, tokenizer)
            if eval_dataset:
                eval_dataset = self._format_dataset(eval_dataset, config.dataset_format, tokenizer)

            # 5. Training arguments
            if progress_callback:
                progress_callback("Konfiguroidaan training...")

            output_dir = Path(config.output_dir) if config.output_dir else self.output_dir / config.run_name
            output_dir.mkdir(parents=True, exist_ok=True)

            sft_config = SFTConfig(
                output_dir=str(output_dir),
                num_train_epochs=config.training.epochs,
                per_device_train_batch_size=config.training.batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                learning_rate=config.training.learning_rate,
                warmup_ratio=config.training.warmup_ratio,
                lr_scheduler_type=config.training.lr_scheduler_type,
                weight_decay=config.training.weight_decay,
                max_grad_norm=config.training.max_grad_norm,
                fp16=use_fp16,
                bf16=use_bf16,
                gradient_checkpointing=config.training.gradient_checkpointing,
                save_steps=config.training.save_steps,
                save_total_limit=config.training.save_total_limit,
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=config.training.eval_steps if eval_dataset else None,
                logging_steps=config.training.logging_steps,
                report_to="none",
                run_name=config.run_name,
                # Automaattisesti valittu optimoija
                optim=hw_caps["recommended_optim"],
                # DataLoader optimoinnit - automaattisesti tunnistettu workers-määrä
                dataloader_pin_memory=hw_caps["cuda_available"],
                dataloader_num_workers=hw_caps["recommended_workers"],
                # Gradient checkpointing optimointi
                # use_reentrant=False on suositeltava PyTorch 2.0+ ja nopeampi
                gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
                # SFTConfig-spesifiset parametrit
                max_length=config.training.max_seq_length,
                packing=config.training.packing,
                dataset_text_field="text",
            )

            # 6. Trainer
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            # 7. Train!
            resume = self._resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
            if progress_callback:
                if resume:
                    progress_callback("Jatketaan viimeisimmasta checkpointista...")
                else:
                    progress_callback("Aloitetaan training...")

            trainer.train(resume_from_checkpoint=resume or None)

            # 8. Tallenna
            if progress_callback:
                progress_callback("Tallennetaan adapter...")

            # Tallenna adapter
            adapter_path = output_dir / "adapter"
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            # Tallenna config
            config_path = output_dir / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time

            return {
                "success": True,
                "output_dir": str(output_dir),
                "adapter_path": str(adapter_path),
                "trainable_params": trainable_params,
                "total_params": total_params,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": f"{int(elapsed//60)}m {int(elapsed%60)}s",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_checkpoints(self, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Listaa checkpointit."""
        search_dir = output_dir or self.output_dir
        checkpoints = []

        for item in search_dir.iterdir():
            if item.is_dir():
                # Tarkista onko adapter
                adapter_config = item / "adapter_config.json"
                if adapter_config.exists():
                    checkpoints.append({
                        "path": item,
                        "name": item.name,
                        "type": "adapter",
                        "created": datetime.fromtimestamp(item.stat().st_mtime),
                    })

                # Tarkista onko checkpoint
                elif (item / "adapter").exists():
                    checkpoints.append({
                        "path": item,
                        "name": item.name,
                        "type": "training_output",
                        "created": datetime.fromtimestamp(item.stat().st_mtime),
                    })

        return sorted(checkpoints, key=lambda x: x["created"], reverse=True)

    # ==================== CHECKPOINT CLEANUP ====================

    def find_best_checkpoint(self, run_dir: Path) -> Optional[Path]:
        """
        Find the best checkpoint in a training run directory by eval_loss.

        Args:
            run_dir: Path to the training run directory

        Returns:
            Path to the best checkpoint, or None if no checkpoints found
        """
        checkpoints = list(run_dir.glob("checkpoint-*"))

        if not checkpoints:
            return None

        best_checkpoint = None
        best_loss = float('inf')

        for cp in checkpoints:
            # Try to find trainer_state.json for eval metrics
            trainer_state = cp / "trainer_state.json"
            if trainer_state.exists():
                try:
                    with open(trainer_state, 'r') as f:
                        state = json.load(f)

                    # Find best eval_loss in log history
                    for entry in state.get("log_history", []):
                        eval_loss = entry.get("eval_loss")
                        if eval_loss is not None and eval_loss < best_loss:
                            best_loss = eval_loss
                            best_checkpoint = cp
                except Exception:
                    pass

        # If no eval_loss found, fall back to the last checkpoint by step number
        if best_checkpoint is None and checkpoints:
            best_checkpoint = max(
                checkpoints,
                key=lambda p: int(p.name.split("-")[1]) if p.name.startswith("checkpoint-") else 0
            )

        return best_checkpoint

    def cleanup_checkpoints(
        self,
        run_dir: Path,
        keep_best: bool = True,
        keep_latest: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Clean up checkpoints in a training run, keeping only the best and latest.

        Args:
            run_dir: Path to the training run directory
            keep_best: Whether to keep the best checkpoint (by eval_loss)
            keep_latest: Whether to keep the latest checkpoint
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with cleanup results including deleted count and saved space
        """
        checkpoints = list(run_dir.glob("checkpoint-*"))

        if len(checkpoints) <= 2:
            return {
                "success": True,
                "deleted_count": 0,
                "saved_bytes": 0,
                "kept_checkpoints": [cp.name for cp in checkpoints],
                "message": "No cleanup needed (2 or fewer checkpoints)"
            }

        # Find checkpoints to keep
        checkpoints_to_keep = set()

        if keep_best:
            best_cp = self.find_best_checkpoint(run_dir)
            if best_cp:
                checkpoints_to_keep.add(best_cp)
                if progress_callback:
                    progress_callback(f"Best checkpoint: {best_cp.name}")

        if keep_latest:
            latest_cp = max(
                checkpoints,
                key=lambda p: int(p.name.split("-")[1]) if p.name.startswith("checkpoint-") else 0
            )
            checkpoints_to_keep.add(latest_cp)
            if progress_callback:
                progress_callback(f"Latest checkpoint: {latest_cp.name}")

        # Delete other checkpoints
        deleted_count = 0
        saved_bytes = 0

        for cp in checkpoints:
            if cp not in checkpoints_to_keep:
                try:
                    # Calculate size before deletion
                    cp_size = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
                    saved_bytes += cp_size

                    shutil.rmtree(cp)
                    deleted_count += 1

                    if progress_callback:
                        progress_callback(f"Deleted: {cp.name} ({cp_size / (1024**2):.1f} MB)")

                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Failed to delete {cp.name}: {e}")

        return {
            "success": True,
            "deleted_count": deleted_count,
            "saved_bytes": saved_bytes,
            "saved_mb": round(saved_bytes / (1024**2), 1),
            "saved_gb": round(saved_bytes / (1024**3), 2),
            "kept_checkpoints": [cp.name for cp in checkpoints_to_keep],
            "message": f"Cleaned up {deleted_count} checkpoint(s), saved {saved_bytes / (1024**3):.2f} GB"
        }

    def auto_cleanup_after_training(
        self,
        run_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Automatically clean up checkpoints after training completes.

        This keeps only:
        - The best checkpoint (by eval_loss)
        - The latest checkpoint

        Also moves the final adapter to the adapters directory.

        Args:
            run_dir: Path to the training run directory
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with cleanup results and final adapter path
        """
        paths = get_paths()
        result = {
            "success": True,
            "cleanup": None,
            "adapter_path": None,
        }

        # 1. Clean up checkpoints
        if progress_callback:
            progress_callback("Cleaning up checkpoints...")

        cleanup_result = self.cleanup_checkpoints(
            run_dir,
            keep_best=True,
            keep_latest=True,
            progress_callback=progress_callback
        )
        result["cleanup"] = cleanup_result

        # 2. Move final adapter to adapters directory
        adapter_source = run_dir / "adapter"
        if adapter_source.exists():
            # Create adapter name from run directory name
            adapter_name = run_dir.name

            # Ensure adapters directory exists
            adapters_dir = paths.adapters_dir
            adapters_dir.mkdir(parents=True, exist_ok=True)

            adapter_dest = adapters_dir / adapter_name

            # Handle existing adapter with same name
            if adapter_dest.exists():
                # Add timestamp suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                adapter_dest = adapters_dir / f"{adapter_name}_{timestamp}"

            try:
                shutil.copytree(adapter_source, adapter_dest)
                result["adapter_path"] = str(adapter_dest)

                if progress_callback:
                    progress_callback(f"Adapter saved to: {adapter_dest}")

            except Exception as e:
                result["success"] = False
                result["error"] = f"Failed to copy adapter: {e}"

        return result

    # ==================== INFERENCE / TESTING ====================

    def test_adapter(self, base_model_path: Path, adapter_path: Path,
                    prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
        """Testaa adapteria."""
        if not self.get_status()["ready"]:
            return {"success": False, "error": "Riippuvuudet puuttuvat"}

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            # Lataa base model (GPU + CPU, ei levylle siirtoa)
            tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                device_map="auto",
                torch_dtype=torch.float16,
            )

            # Lataa adapter
            model = PeftModel.from_pretrained(model, str(adapter_path))

            # Generoi
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "success": True,
                "prompt": prompt,
                "response": response,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def merge_adapter(self, base_model_path: Path, adapter_path: Path,
                     output_path: Path, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Yhdistä adapter base-malliin."""
        if not self.get_status()["ready"]:
            return {"success": False, "error": "Riippuvuudet puuttuvat"}

        try:
            import torch
            import gc
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            if progress_callback:
                progress_callback("Ladataan base model...")

            tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))

            # Käytä device_map="auto" ILMAN offload_folder parametria
            # Tämä käyttää GPU + CPU RAM yhdistelmää, ei levyä
            # Levylle siirto (disk offload) luo meta-tensoreita jotka eivät toimi LoRA:n kanssa
            model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            if progress_callback:
                progress_callback("Ladataan adapter...")

            model = PeftModel.from_pretrained(model, str(adapter_path))

            if progress_callback:
                progress_callback("Yhdistetään painot...")

            model = model.merge_and_unload()

            if progress_callback:
                progress_callback("Tallennetaan...")

            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_path), safe_serialization=True)
            tokenizer.save_pretrained(str(output_path))

            # Vapauta muisti
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "success": True,
                "output_path": str(output_path),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== CONFIG MANAGEMENT ====================

    def save_config(self, config: FullConfig, name: str) -> Path:
        """Tallenna konfiguraatio."""
        config_path = self.configs_dir / f"{name}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return config_path

    def load_config(self, name_or_path: str) -> Optional[FullConfig]:
        """Lataa konfiguraatio."""
        if Path(name_or_path).exists():
            config_path = Path(name_or_path)
        else:
            config_path = self.configs_dir / f"{name_or_path}.json"

        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return FullConfig.from_dict(data)
        except Exception:
            return None

    def list_configs(self) -> List[Dict[str, Any]]:
        """Listaa tallennetut konfiguraatiot."""
        configs = []
        for config_file in self.configs_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                configs.append({
                    "path": config_file,
                    "name": config_file.stem,
                    "model": data.get("model_name", "Unknown"),
                    "dataset": Path(data.get("dataset_path", "")).name,
                })
            except Exception:
                pass
        return configs
