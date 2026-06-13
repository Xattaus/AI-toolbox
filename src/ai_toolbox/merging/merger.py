"""
AI TOOLBOX - Model Merger
=========================

Yhdista malleja eri tekniikoilla: SLERP, TIES, Frankenmerge.
"""

import json
import shutil
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math

from rich.console import Console

from ..core.paths import get_paths

console = Console()


class MergeMethod(Enum):
    """Tuetut merge-metodit."""
    SLERP = "slerp"
    TIES = "ties"
    FRANKENMERGE = "frankenmerge"
    LINEAR = "linear"


@dataclass
class MergeConfig:
    """Merge-konfiguraatio."""
    method: MergeMethod
    models: List[Path]
    output_name: str

    # SLERP-parametrit
    slerp_ratio: float = 0.5  # 0.0 = model1, 1.0 = model2

    # TIES-parametrit
    ties_density: float = 0.5  # Kuinka suuri osa painoista sailytetaan (0.0-1.0)
    ties_majority_sign: bool = True  # Kayta enemmistoaanta etumerkeille

    # Frankenmerge-parametrit
    layer_ranges: Optional[Dict[str, Tuple[int, int]]] = None  # {"model_path": (start, end)}

    # Yleiset
    base_model: Optional[Path] = None  # TIES:lle base model
    normalize: bool = True
    dtype: str = "float16"  # "float16", "float32", "bfloat16"


@dataclass
class AdvancedMergeConfig:
    """
    Advanced merge configuration for handling model differences.

    Kasittelee mallien valisia eroja:
    - Eri vocab_size (embedding trimaus/padding)
    - Eri presisio (FP16/BF16/FP32 normalisointi)
    - Eri RoPE scaling konfiguraatiot
    """
    method: MergeMethod
    models: List[Path]
    output_name: str

    # Vocab handling
    vocab_strategy: str = "minimum"  # "minimum", "maximum", "first", "second"
    vocab_pad_value: float = 0.0     # Padding value for extended vocab

    # Precision handling
    target_dtype: str = "bfloat16"   # "float16", "bfloat16", "float32"
    normalize_precision: bool = True  # Convert all models to same precision

    # Merge parameters
    slerp_ratio: float = 0.5
    ties_density: float = 0.5

    # Config handling
    config_source: str = "first"      # "first", "second", "merge"
    merge_rope_scaling: bool = True   # Prefer model with RoPE scaling

    # Advanced options
    skip_embedding_merge: bool = False  # Skip merging embed_tokens
    skip_lm_head_merge: bool = False    # Skip merging lm_head
    embedding_merge_ratio: Optional[float] = None  # Different ratio for embeddings

    # Tokenizer source
    tokenizer_source: str = "first"  # "first", "second"


class ModelMerger:
    """Mallien yhdistamistyokalu."""

    def __init__(self):
        """Alusta merger."""
        paths = get_paths()
        self.output_dir = paths.models_dir / "merged"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = paths.root / "temp" / "merge"

        # Tarkista onko torch saatavilla
        self._torch_available = False
        self._safetensors_available = False
        self._check_dependencies()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Tarkista riippuvuudet."""
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False

        try:
            import safetensors
            self._safetensors_available = True
        except ImportError:
            self._safetensors_available = False

        return {
            "torch": self._torch_available,
            "safetensors": self._safetensors_available,
        }

    def get_status(self) -> Dict[str, Any]:
        """Palauta mergerin status."""
        deps = self._check_dependencies()
        return {
            "ready": deps["torch"] and deps["safetensors"],
            "dependencies": deps,
            "output_dir": str(self.output_dir),
            "missing": [k for k, v in deps.items() if not v],
        }

    def install_dependencies(self, progress_callback: Optional[Callable] = None) -> bool:
        """Asenna puuttuvat riippuvuudet."""
        import subprocess
        import sys

        packages = []
        if not self._torch_available:
            packages.append("torch")
        if not self._safetensors_available:
            packages.append("safetensors")

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

            # Paivita status
            self._check_dependencies()
            return True

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Asennus epaonnistui: {e}[/red]")
            return False

    def list_compatible_models(self, model_dir: Path) -> List[Dict[str, Any]]:
        """Listaa yhteensopivat mallit kansiosta."""
        models = []

        # Etsi safetensors-malleja
        for model_path in model_dir.rglob("*.safetensors"):
            if model_path.name.startswith("."):
                continue

            # Tarkista onko HF-tyylinen kansio
            parent = model_path.parent
            config_file = parent / "config.json"

            model_info = {
                "path": model_path,
                "name": parent.name if config_file.exists() else model_path.stem,
                "format": "safetensors",
                "is_sharded": "model-" in model_path.name and "-of-" in model_path.name,
                "has_config": config_file.exists(),
                "size_bytes": model_path.stat().st_size,
            }

            # Jos on konfiguraatio, lue mallin tiedot
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        model_info["architecture"] = config.get("architectures", ["Unknown"])[0]
                        model_info["hidden_size"] = config.get("hidden_size", 0)
                        model_info["num_layers"] = config.get("num_hidden_layers", 0)
                except Exception:
                    pass

            # Valta duplikaatteja (shard-tiedostot)
            if not model_info["is_sharded"] or "-00001-of-" in model_path.name:
                models.append(model_info)

        # Etsi myos PyTorch-malleja
        for model_path in model_dir.rglob("pytorch_model.bin"):
            parent = model_path.parent
            config_file = parent / "config.json"

            model_info = {
                "path": parent,
                "name": parent.name,
                "format": "pytorch",
                "is_sharded": False,
                "has_config": config_file.exists(),
                "size_bytes": model_path.stat().st_size,
            }
            models.append(model_info)

        return models

    def validate_models_compatible(
        self,
        models: List[Path],
        strict: bool = True,
        max_vocab_diff: int = 1000,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tarkista ovatko mallit yhteensopivia yhdistamiseen.

        Args:
            models: Lista malleista
            strict: True = esta vocab_size erot, False = salli pienet erot
            max_vocab_diff: Maksimi sallittu vocab_size ero (kun strict=False)

        Returns:
            (is_compatible, message, details)
        """
        details: Dict[str, Any] = {}

        if len(models) < 2:
            return False, "Tarvitaan vahintaan 2 mallia yhdistamiseen", details

        configs = []
        for model_path in models:
            config_file = None
            if model_path.is_dir():
                config_file = model_path / "config.json"
            else:
                config_file = model_path.parent / "config.json"

            if config_file and config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        configs.append(json.load(f))
                except json.JSONDecodeError as e:
                    return False, f"Virheellinen config.json: {model_path} - {e}", details
                except IOError as e:
                    return False, f"Ei voitu lukea config.json: {model_path} - {e}", details
            else:
                return False, f"Config.json puuttuu: {model_path}", details

        # Keraa tiedot
        details = {
            "architectures": [c.get("architectures", ["Unknown"])[0] for c in configs],
            "hidden_sizes": [c.get("hidden_size", 0) for c in configs],
            "num_layers": [c.get("num_hidden_layers", 0) for c in configs],
            "vocab_sizes": [c.get("vocab_size", 0) for c in configs],
            "model_types": [c.get("model_type", "unknown") for c in configs],
            "rope_scaling": [c.get("rope_scaling") for c in configs],
            "max_position_embeddings": [c.get("max_position_embeddings", 0) for c in configs],
        }

        # Kriittiset tarkistukset (eivat voi joustaa)
        if len(set(details["architectures"])) > 1:
            return False, f"Eri arkkitehtuurit: {', '.join(details['architectures'])}", details

        if len(set(details["hidden_sizes"])) > 1:
            return False, f"Eri hidden_size: {details['hidden_sizes']}", details

        if len(set(details["num_layers"])) > 1:
            return False, f"Eri kerrosmaarat: {details['num_layers']}", details

        # Joustava tarkistus (vocab_size)
        vocab_sizes = details["vocab_sizes"]
        if len(set(vocab_sizes)) > 1:
            vocab_diff = max(vocab_sizes) - min(vocab_sizes)
            details["vocab_diff"] = vocab_diff

            if strict:
                return False, f"Eri vocab_size: {vocab_sizes} (ero: {vocab_diff})", details
            else:
                if vocab_diff > max_vocab_diff:
                    return False, f"Vocab_size ero liian suuri ({vocab_diff} > {max_vocab_diff})", details
                # Pieni ero OK advanced mergessa - jatketaan varoituksella
                details["vocab_warning"] = f"Vocab_size ero: {vocab_diff} tokenia (kasitellaan automaattisesti)"

        return True, "Mallit ovat yhteensopivia", details

    def _load_safetensors(self, path: Path) -> Dict[str, Any]:
        """Lataa safetensors-tiedosto."""
        from safetensors import safe_open
        import torch

        tensors = {}

        # Tarkista onko sharded
        if path.is_dir():
            # Lataa kaikki shard-tiedostot
            shard_files = sorted(path.glob("model-*.safetensors"))
            if not shard_files:
                shard_files = [path / "model.safetensors"]
        else:
            shard_files = [path]

        for shard_file in shard_files:
            if shard_file.exists():
                with safe_open(str(shard_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)

        return tensors

    def _save_safetensors(self, tensors: Dict[str, Any], output_path: Path):
        """Tallenna safetensors-muotoon."""
        from safetensors.torch import save_file

        # Varmista output-kansio
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_file(tensors, str(output_path))

    def _slerp(self, t: float, v0: Any, v1: Any, DOT_THRESHOLD: float = 0.9995) -> Any:
        """
        Spherical linear interpolation (SLERP) kahdelle tensorille.

        Args:
            t: Interpolointikerroin (0.0 = v0, 1.0 = v1)
            v0: Ensimmainen tensori
            v1: Toinen tensori
            DOT_THRESHOLD: Kynnys lineaariselle interpoloinnille

        Returns:
            Interpoloitu tensori
        """
        import torch

        # Kopioi tensorit
        v0_copy = v0.clone().float()
        v1_copy = v1.clone().float()

        # Normalisoi
        v0_norm = torch.nn.functional.normalize(v0_copy.flatten(), dim=0)
        v1_norm = torch.nn.functional.normalize(v1_copy.flatten(), dim=0)

        # Laske pistetulo
        dot = torch.dot(v0_norm, v1_norm).item()

        # Jos vektorit ovat lahes samansuuntaisia, kayta lineaarista interpolointia
        if abs(dot) > DOT_THRESHOLD:
            result = (1 - t) * v0_copy + t * v1_copy
            return result.to(v0.dtype)

        # SLERP
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = math.sin(theta_t)

        s0 = math.cos(theta_t) - dot * sin_theta_t / sin_theta_0
        s1 = sin_theta_t / sin_theta_0

        result = s0 * v0_copy + s1 * v1_copy
        return result.to(v0.dtype)

    def _linear_interpolate(self, t: float, v0: Any, v1: Any) -> Any:
        """Lineaarinen interpolointi."""
        return (1 - t) * v0 + t * v1

    def merge_slerp(
        self,
        model1_path: Path,
        model2_path: Path,
        ratio: float = 0.5,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Yhdista kaksi mallia SLERP:lla.

        Args:
            model1_path: Ensimmainen malli
            model2_path: Toinen malli
            ratio: Interpolointikerroin (0.0 = model1, 1.0 = model2)
            output_name: Tulostiedoston nimi
            progress_callback: Edistymisen callback

        Returns:
            Tulos-dict
        """
        import torch

        if not self._torch_available or not self._safetensors_available:
            return {"success": False, "error": "Riippuvuudet puuttuvat. Asenna torch ja safetensors."}

        try:
            if progress_callback:
                progress_callback("Ladataan malli 1...")

            # Lataa mallit
            tensors1 = self._load_safetensors(model1_path)

            if progress_callback:
                progress_callback("Ladataan malli 2...")

            tensors2 = self._load_safetensors(model2_path)

            # Varmista samat avaimet
            if set(tensors1.keys()) != set(tensors2.keys()):
                # Kayta yhteisia avaimia
                common_keys = set(tensors1.keys()) & set(tensors2.keys())
                console.print(f"[yellow]Varoitus: {len(tensors1) - len(common_keys)} avainta puuttuu[/yellow]")
            else:
                common_keys = set(tensors1.keys())

            if progress_callback:
                progress_callback("Yhdistetaan SLERP:lla...")

            # SLERP jokaiselle tensorille
            merged_tensors = {}
            total_keys = len(common_keys)

            for i, key in enumerate(common_keys):
                t1 = tensors1[key]
                t2 = tensors2[key]

                # Varmista sama koko
                if t1.shape != t2.shape:
                    console.print(f"[yellow]Ohitetaan {key}: eri koot {t1.shape} vs {t2.shape}[/yellow]")
                    merged_tensors[key] = t1  # Kayta ensimmaista
                    continue

                # SLERP
                merged_tensors[key] = self._slerp(ratio, t1, t2)

                if progress_callback and i % 50 == 0:
                    progress_callback(f"Yhdistetaan... {i}/{total_keys}")

            # Maarita output-nimi
            if not output_name:
                m1_name = model1_path.stem if model1_path.is_file() else model1_path.name
                m2_name = model2_path.stem if model2_path.is_file() else model2_path.name
                output_name = f"slerp_{m1_name}_{m2_name}_{ratio:.2f}"

            # Luo output-kansio
            output_dir = self.output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "model.safetensors"

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_safetensors(merged_tensors, output_file)

            # Kopioi config.json
            for src_path in [model1_path, model2_path]:
                config_src = src_path / "config.json" if src_path.is_dir() else src_path.parent / "config.json"
                if config_src.exists():
                    config_dest = output_dir / "config.json"
                    shutil.copy2(config_src, config_dest)

                    # Paivita config
                    with open(config_dest, 'r') as f:
                        config = json.load(f)
                    config["_merge_info"] = {
                        "method": "slerp",
                        "ratio": ratio,
                        "model1": str(model1_path),
                        "model2": str(model2_path),
                    }
                    with open(config_dest, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                    break

            # Kopioi tokenizer-tiedostot
            for src_path in [model1_path, model2_path]:
                src_dir = src_path if src_path.is_dir() else src_path.parent
                for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                 "vocab.json", "merges.txt", "tokenizer.model"]:
                    tok_src = src_dir / tok_file
                    if tok_src.exists():
                        shutil.copy2(tok_src, output_dir / tok_file)

            output_size = output_file.stat().st_size / (1024**3)

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_file": str(output_file),
                "file_size_gb": output_size,
                "method": "slerp",
                "ratio": ratio,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def merge_ties(
        self,
        models: List[Path],
        base_model: Path,
        density: float = 0.5,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Yhdista malleja TIES-Merging -menetelmalla.

        TIES: Trim, Elect Sign, Disjoint Merge

        Args:
            models: Lista malleista yhdistettavaksi
            base_model: Perusmalli (esim. alkuperainen pretrained)
            density: Kuinka suuri osa delta-painoista sailytetaan (0.0-1.0)
            output_name: Tulostiedoston nimi
            progress_callback: Edistymisen callback

        Returns:
            Tulos-dict
        """
        import torch

        if not self._torch_available or not self._safetensors_available:
            return {"success": False, "error": "Riippuvuudet puuttuvat."}

        try:
            if progress_callback:
                progress_callback("Ladataan base model...")

            base_tensors = self._load_safetensors(base_model)

            # Lataa kaikki mallit ja laske deltat
            model_deltas = []
            for i, model_path in enumerate(models):
                if progress_callback:
                    progress_callback(f"Ladataan malli {i+1}/{len(models)}...")

                model_tensors = self._load_safetensors(model_path)

                # Laske delta (model - base)
                deltas = {}
                for key in base_tensors:
                    if key in model_tensors:
                        if base_tensors[key].shape == model_tensors[key].shape:
                            deltas[key] = model_tensors[key].float() - base_tensors[key].float()

                model_deltas.append(deltas)

            if progress_callback:
                progress_callback("Suoritetaan TIES merge...")

            common_keys = set(base_tensors.keys())
            for deltas in model_deltas:
                common_keys &= set(deltas.keys())

            merged_tensors = {}

            for key in common_keys:
                # Keraa deltat tasta kerroksesta
                layer_deltas = [d[key] for d in model_deltas if key in d]

                if not layer_deltas:
                    merged_tensors[key] = base_tensors[key]
                    continue

                # TRIM: Sailyta top-k% suurimmat arvot PER MALLI (TIES-paperin
                # mukaisesti jokainen task-vektori karsitaan erikseen -
                # globaali kynnys nollaisi pienempien deltojen mallin kokonaan)
                trimmed_list = []
                for delta in layer_deltas:
                    flat = delta.abs().flatten()
                    k = int(flat.numel() * density)
                    if 0 < k < flat.numel():
                        threshold = torch.topk(flat, k).values[-1]
                        trimmed_list.append(delta * (delta.abs() >= threshold))
                    else:
                        trimmed_list.append(delta)

                trimmed = torch.stack(trimmed_list, dim=0)

                # ELECT SIGN: Valitse etumerkki enemmistoaanestykella
                signs = torch.sign(trimmed)
                sum_signs = signs.sum(dim=0)
                majority_sign = torch.sign(sum_signs)
                majority_sign[majority_sign == 0] = 1  # Tasatilanne -> positiivinen

                # DISJOINT MERGE: Keskiarvo samanmerkkisista
                aligned = trimmed * (signs == majority_sign.unsqueeze(0))
                count = (signs == majority_sign.unsqueeze(0)).sum(dim=0).float()
                count[count == 0] = 1  # Valta jako nollalla

                merged_delta = aligned.sum(dim=0) / count

                # Lisaa base modeliin
                merged_tensors[key] = (base_tensors[key].float() + merged_delta).to(base_tensors[key].dtype)

            # Lisaa loput avaimet base modelista
            for key in base_tensors:
                if key not in merged_tensors:
                    merged_tensors[key] = base_tensors[key]

            # Maarita output-nimi
            if not output_name:
                output_name = f"ties_merge_{len(models)}models_d{density:.2f}"

            # Tallenna
            output_dir = self.output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "model.safetensors"

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_safetensors(merged_tensors, output_file)

            # Kopioi config ja tokenizer
            config_src = base_model / "config.json" if base_model.is_dir() else base_model.parent / "config.json"
            if config_src.exists():
                config_dest = output_dir / "config.json"
                shutil.copy2(config_src, config_dest)

                with open(config_dest, 'r') as f:
                    config = json.load(f)
                config["_merge_info"] = {
                    "method": "ties",
                    "density": density,
                    "base_model": str(base_model),
                    "models": [str(m) for m in models],
                }
                with open(config_dest, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

            # Kopioi tokenizer
            src_dir = base_model if base_model.is_dir() else base_model.parent
            for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                             "vocab.json", "merges.txt", "tokenizer.model"]:
                tok_src = src_dir / tok_file
                if tok_src.exists():
                    shutil.copy2(tok_src, output_dir / tok_file)

            output_size = output_file.stat().st_size / (1024**3)

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_file": str(output_file),
                "file_size_gb": output_size,
                "method": "ties",
                "density": density,
                "num_models": len(models),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def merge_frankenmerge(
        self,
        models: Dict[Path, Tuple[int, int]],
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Frankenmerge: Valitse kerroksia eri malleista.

        Args:
            models: Dict {model_path: (start_layer, end_layer)}
                    Kerrokset start_layer - end_layer (inklusiivinen) otetaan tasta mallista
            output_name: Tulostiedoston nimi
            progress_callback: Edistymisen callback

        Returns:
            Tulos-dict
        """
        import torch

        if not self._torch_available or not self._safetensors_available:
            return {"success": False, "error": "Riippuvuudet puuttuvat."}

        try:
            # Lataa kaikki mallit
            model_tensors = {}
            model_list = list(models.keys())

            for i, model_path in enumerate(model_list):
                if progress_callback:
                    progress_callback(f"Ladataan malli {i+1}/{len(model_list)}...")
                model_tensors[model_path] = self._load_safetensors(model_path)

            if progress_callback:
                progress_callback("Yhdistetaan kerroksia...")

            # Kayta ensimmaista mallia pohjana
            first_model = model_list[0]
            merged_tensors = dict(model_tensors[first_model])

            # Tunnista kerrosavaimet (esim. "model.layers.0.self_attn.q_proj.weight")
            # Yleinen kaava: layers.N. tai h.N. tai blocks.N.
            import re
            layer_pattern = re.compile(r'(layers|blocks|h)\.(\d+)\.')

            # Kay lapi jokainen malli ja sen kerrosalue
            for model_path, (start_layer, end_layer) in models.items():
                tensors = model_tensors[model_path]

                for key, tensor in tensors.items():
                    match = layer_pattern.search(key)
                    if match:
                        layer_idx = int(match.group(2))
                        if start_layer <= layer_idx <= end_layer:
                            merged_tensors[key] = tensor

            # Maarita output-nimi
            if not output_name:
                parts = []
                for model_path, (start, end) in models.items():
                    name = model_path.stem if model_path.is_file() else model_path.name
                    parts.append(f"{name[:10]}[{start}-{end}]")
                output_name = f"franken_{'_'.join(parts)}"

            # Tallenna
            output_dir = self.output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "model.safetensors"

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_safetensors(merged_tensors, output_file)

            # Kopioi config ensimmaisesta mallista
            config_src = first_model / "config.json" if first_model.is_dir() else first_model.parent / "config.json"
            if config_src.exists():
                config_dest = output_dir / "config.json"
                shutil.copy2(config_src, config_dest)

                with open(config_dest, 'r') as f:
                    config = json.load(f)
                config["_merge_info"] = {
                    "method": "frankenmerge",
                    "layer_sources": {str(k): v for k, v in models.items()},
                }
                with open(config_dest, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

            # Kopioi tokenizer
            src_dir = first_model if first_model.is_dir() else first_model.parent
            for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                             "vocab.json", "merges.txt", "tokenizer.model"]:
                tok_src = src_dir / tok_file
                if tok_src.exists():
                    shutil.copy2(tok_src, output_dir / tok_file)

            output_size = output_file.stat().st_size / (1024**3)

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_file": str(output_file),
                "file_size_gb": output_size,
                "method": "frankenmerge",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_info(self, model_path: Path) -> Dict[str, Any]:
        """Hae mallin tiedot."""
        info = {
            "path": str(model_path),
            "name": model_path.name if model_path.is_dir() else model_path.stem,
            "exists": model_path.exists(),
        }

        if not model_path.exists():
            return info

        # Etsi config.json
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                info["architecture"] = config.get("architectures", ["Unknown"])[0]
                info["hidden_size"] = config.get("hidden_size", 0)
                info["num_layers"] = config.get("num_hidden_layers", 0)
                info["vocab_size"] = config.get("vocab_size", 0)
                info["model_type"] = config.get("model_type", "unknown")

                # Laske parametrit (karkea arvio)
                h = info["hidden_size"]
                l = info["num_layers"]
                v = info["vocab_size"]
                if h and l:
                    # Transformer: ~12*l*h^2 + 2*v*h parametria
                    params = 12 * l * h * h + 2 * v * h
                    info["estimated_params_b"] = round(params / 1e9, 2)
            except Exception:
                pass

        # Laske koko
        if model_path.is_dir():
            total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        else:
            total_size = model_path.stat().st_size

        info["size_bytes"] = total_size
        info["size_gb"] = round(total_size / (1024**3), 2)

        return info

    def estimate_merge_requirements(self, models: List[Path]) -> Dict[str, Any]:
        """Arvioi merge-operaation vaatimukset."""
        if not models:
            return {
                "total_input_size_gb": 0,
                "estimated_ram_gb": 0,
                "estimated_output_size_gb": 0,
                "num_models": 0,
            }

        total_size = 0
        for model_path in models:
            info = self.get_model_info(model_path)
            total_size += info.get("size_bytes", 0)

        # RAM-vaatimus: noin 2-3x mallien koko (lataus + merge + tallennus)
        ram_required_gb = (total_size * 2.5) / (1024**3)

        return {
            "total_input_size_gb": round(total_size / (1024**3), 2),
            "estimated_ram_gb": round(ram_required_gb, 1),
            "estimated_output_size_gb": round(total_size / len(models) / (1024**3), 2),
            "num_models": len(models),
        }

    def merge_lora(
        self,
        base_model_path: Path,
        adapter_path: Path,
        output_name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Yhdista LoRA-adapter base-malliin.

        Kayttaa PEFT:n merge_and_unload() metodia, joka yhdistaa
        LoRA-painot suoraan base-mallin painoihin.

        Args:
            base_model_path: Polku base-malliin (SafeTensors)
            adapter_path: Polku LoRA-adapteriin (adapter_config.json + weights)
            output_name: Tuloksen nimi (oletus: base_lora-merged)
            progress_callback: Edistymisen callback

        Returns:
            Tulos-dict: success, output_path, error
        """
        if not self._torch_available:
            return {"success": False, "error": "PyTorch ei saatavilla"}

        # Tarkista PEFT
        try:
            from peft import PeftModel
        except ImportError:
            return {"success": False, "error": "PEFT ei asennettu. Asenna: pip install peft"}

        # Validoi adapter
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            return {"success": False, "error": f"adapter_config.json puuttuu: {adapter_path}"}

        try:
            import torch
            import gc
            import tempfile
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if progress_callback:
                progress_callback("Ladataan base model...")

            tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_path),
                trust_remote_code=True,
            )

            # Kayta GPU:ta maksimaalisesti + disk offload kun muisti loppuu
            # Tama on nopein tapa - GPU tekee laskennan, levy toimii varastona
            offload_dir = self.output_dir / "_offload_temp"
            offload_dir.mkdir(parents=True, exist_ok=True)

            if torch.cuda.is_available():
                if progress_callback:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    progress_callback(f"Ladataan base model (GPU {gpu_mem:.1f}GB + disk offload)...")

                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    offload_folder=str(offload_dir),
                    offload_state_dict=True,
                )
            else:
                # Pelkka CPU jos ei GPU:ta
                if progress_callback:
                    progress_callback("Ladataan base model (CPU)...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    device_map={"": "cpu"},
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )

            if progress_callback:
                progress_callback("Ladataan LoRA adapter...")

            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                offload_folder=str(offload_dir),
                is_trainable=False,
            )

            if progress_callback:
                progress_callback("Yhdistetaan painot (merge_and_unload)...")

            model = model.merge_and_unload()

            # Maarita output
            if not output_name:
                base_name = base_model_path.name if base_model_path.is_dir() else base_model_path.stem
                adapter_name = adapter_path.name
                output_name = f"{base_name}_{adapter_name}_merged"

            output_dir = self.output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback("Tallennetaan yhdistetty malli...")

            # Muunna FP16:ksi tallennusta varten (pienempi koko)
            if model.dtype == torch.float32:
                if progress_callback:
                    progress_callback("Muunnetaan FP16 tallennusta varten...")
                model = model.half()

            model.save_pretrained(str(output_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(output_dir))

            # Tallenna merge-info
            merge_info = {
                "method": "lora_merge",
                "base_model": str(base_model_path),
                "adapter": str(adapter_path),
            }

            # Lue ja paivita config.json
            config_file = output_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                config["_merge_info"] = merge_info
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)

            # Vapauta muisti
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Siivoa valiaikaiskansio
            if offload_dir and offload_dir.exists():
                import shutil
                try:
                    shutil.rmtree(offload_dir)
                except Exception:
                    pass  # Ei haittaa jos siivous epaonnistuu

            output_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_name": output_name,
                "file_size_gb": round(output_size / (1024**3), 2),
                "method": "lora_merge",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Advanced Merge Methods
    # =========================================================================

    def _load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """Lataa mallin config.json."""
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def _detect_actual_precision(self, model_path: Path) -> str:
        """
        Tunnista mallin todellinen presisio tiedostokoosta.

        8B Llama-malli:
        - FP32: ~32 GB
        - BF16/FP16: ~16 GB
        - INT8: ~8 GB
        """
        if model_path.is_dir():
            total_size = sum(f.stat().st_size for f in model_path.rglob("*.safetensors"))
        else:
            total_size = model_path.stat().st_size

        size_gb = total_size / (1024**3)

        # Arvioi parametrien maara config.json:sta
        config = self._load_model_config(model_path)
        hidden_size = config.get("hidden_size", 4096)
        num_layers = config.get("num_hidden_layers", 32)
        vocab_size = config.get("vocab_size", 128000)

        # Laske odotettu koko eri presisioille
        # Parametrit: ~12*L*H^2 + 2*V*H
        estimated_params = 12 * num_layers * hidden_size * hidden_size + 2 * vocab_size * hidden_size
        expected_fp16_gb = (estimated_params * 2) / (1024**3)  # 2 bytes per param

        # Vertaa toteutuneeseen kokoon
        if size_gb > expected_fp16_gb * 1.5:
            return "float32"
        elif size_gb > expected_fp16_gb * 0.6:
            return "bfloat16"  # tai float16
        else:
            return "int8"

    def _normalize_tensor_precision(
        self,
        tensor: Any,
        target_dtype: str = "bfloat16",
    ) -> Any:
        """
        Muunna tensori kohde-presisioon.

        Args:
            tensor: Lahtotensori
            target_dtype: "float16", "bfloat16", "float32"

        Returns:
            Muunnettu tensori
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        target = dtype_map.get(target_dtype, torch.bfloat16)

        if tensor.dtype != target:
            return tensor.to(target)
        return tensor

    def _harmonize_vocab_tensors(
        self,
        tensor1: Any,
        tensor2: Any,
        vocab_size1: int,
        vocab_size2: int,
        strategy: str = "minimum",
        pad_value: float = 0.0,
    ) -> Tuple[Any, Any, int]:
        """
        Harmonisoi embedding/lm_head tensorit samaan vocab_size:een.

        Args:
            tensor1: Ensimmainen tensori (vocab_size1 x hidden_size)
            tensor2: Toinen tensori (vocab_size2 x hidden_size)
            vocab_size1: Ensimmaisen mallin vocab_size
            vocab_size2: Toisen mallin vocab_size
            strategy: "minimum" = trim isompi, "maximum" = pad pienempi,
                      "first" = kayta 1:sta, "second" = kayta 2:sta
            pad_value: Arvo paddaukselle

        Returns:
            (harmonized_tensor1, harmonized_tensor2, final_vocab_size)
        """
        import torch

        if vocab_size1 == vocab_size2:
            return tensor1, tensor2, vocab_size1

        # Maarita kohde-koko
        if strategy == "minimum":
            target_size = min(vocab_size1, vocab_size2)
        elif strategy == "maximum":
            target_size = max(vocab_size1, vocab_size2)
        elif strategy == "first":
            target_size = vocab_size1
        else:  # "second"
            target_size = vocab_size2

        def adjust_tensor(t: Any, current_size: int, target: int) -> Any:
            """Saada tai paddaa tensori."""
            if current_size == target:
                return t
            elif current_size > target:
                # Trim
                return t[:target, :]
            else:
                # Pad
                hidden_size = t.shape[1]
                pad_tensor = torch.full(
                    (target - current_size, hidden_size),
                    pad_value,
                    dtype=t.dtype,
                    device=t.device,
                )
                return torch.cat([t, pad_tensor], dim=0)

        t1 = adjust_tensor(tensor1, vocab_size1, target_size)
        t2 = adjust_tensor(tensor2, vocab_size2, target_size)

        return t1, t2, target_size

    def _merge_configs(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        merge_config: AdvancedMergeConfig,
        final_vocab_size: int,
    ) -> Dict[str, Any]:
        """
        Yhdista mallien config.json-tiedostot.

        Args:
            config1: Ensimmaisen mallin config
            config2: Toisen mallin config
            merge_config: Advanced merge konfiguraatio
            final_vocab_size: Lopullinen vocab_size

        Returns:
            Yhdistetty config
        """
        # Valitse pohja
        if merge_config.config_source == "first":
            base = dict(config1)
        elif merge_config.config_source == "second":
            base = dict(config2)
        else:  # "merge"
            base = dict(config1)

        # Paivita vocab_size
        base["vocab_size"] = final_vocab_size

        # Kasittele RoPE scaling
        if merge_config.merge_rope_scaling:
            # Suosi mallia jolla on RoPE scaling
            if config1.get("rope_scaling") and not config2.get("rope_scaling"):
                base["rope_scaling"] = config1["rope_scaling"]
                base["max_position_embeddings"] = config1.get("max_position_embeddings", 8192)
            elif config2.get("rope_scaling") and not config1.get("rope_scaling"):
                base["rope_scaling"] = config2["rope_scaling"]
                base["max_position_embeddings"] = config2.get("max_position_embeddings", 8192)

        # Lisaa merge-info
        base["_merge_info"] = {
            "method": f"advanced_{merge_config.method.value}",
            "model1": str(merge_config.models[0]) if merge_config.models else "",
            "model2": str(merge_config.models[1]) if len(merge_config.models) > 1 else "",
            "vocab_strategy": merge_config.vocab_strategy,
            "original_vocab_sizes": [
                config1.get("vocab_size", 0),
                config2.get("vocab_size", 0),
            ],
            "final_vocab_size": final_vocab_size,
            "slerp_ratio": merge_config.slerp_ratio,
            "target_dtype": merge_config.target_dtype,
        }

        return base

    def _copy_advanced_tokenizer(
        self,
        model1_path: Path,
        model2_path: Path,
        output_dir: Path,
        config: AdvancedMergeConfig,
    ) -> None:
        """Kopioi tokenizer valitusta mallista."""
        source_path = model1_path if config.tokenizer_source == "first" else model2_path
        src_dir = source_path if source_path.is_dir() else source_path.parent

        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "added_tokens.json",
        ]

        for tok_file in tokenizer_files:
            tok_src = src_dir / tok_file
            if tok_src.exists():
                shutil.copy2(tok_src, output_dir / tok_file)

    def merge_advanced(
        self,
        model1_path: Path,
        model2_path: Path,
        config: AdvancedMergeConfig,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Edistynyt mallien yhdistaminen joka kasittelee yhteensopimattomuudet.

        Kasittelee automaattisesti:
        - Vocab_size erot (embedding trimaus/padding)
        - Presisio erot (FP16/BF16/FP32 normalisointi)
        - RoPE scaling erot

        Args:
            model1_path: Ensimmainen malli
            model2_path: Toinen malli
            config: AdvancedMergeConfig
            progress_callback: Edistymisen callback

        Returns:
            Tulos-dict
        """
        import torch
        import gc

        if not self._torch_available or not self._safetensors_available:
            return {"success": False, "error": "Riippuvuudet puuttuvat. Asenna torch ja safetensors."}

        try:
            # 1. Lataa mallit ja konfiguraatiot
            if progress_callback:
                progress_callback("Analysoidaan malleja...")

            config1 = self._load_model_config(model1_path)
            config2 = self._load_model_config(model2_path)

            vocab_size1 = config1.get("vocab_size", 128256)
            vocab_size2 = config2.get("vocab_size", 128256)

            # 2. Tarkista todellinen presisio
            actual_dtype1 = self._detect_actual_precision(model1_path)
            actual_dtype2 = self._detect_actual_precision(model2_path)

            if progress_callback:
                progress_callback(f"Malli 1: vocab={vocab_size1}, dtype={actual_dtype1}")
                progress_callback(f"Malli 2: vocab={vocab_size2}, dtype={actual_dtype2}")

            # 3. Lataa tensorit
            if progress_callback:
                progress_callback("Ladataan malli 1...")
            tensors1 = self._load_safetensors(model1_path)

            if progress_callback:
                progress_callback("Ladataan malli 2...")
            tensors2 = self._load_safetensors(model2_path)

            # 4. Normalisoi presisio
            if config.normalize_precision:
                if progress_callback:
                    progress_callback(f"Normalisoidaan presisioksi {config.target_dtype}...")

                for key in tensors1:
                    tensors1[key] = self._normalize_tensor_precision(tensors1[key], config.target_dtype)
                for key in tensors2:
                    tensors2[key] = self._normalize_tensor_precision(tensors2[key], config.target_dtype)

            # 5. Identifioi embedding-tensorit
            embedding_keys = ["model.embed_tokens.weight", "lm_head.weight"]

            # 6. Harmonisoi vocab_size embedding-tensoreille
            final_vocab_size = vocab_size1  # oletus

            for emb_key in embedding_keys:
                if emb_key in tensors1 and emb_key in tensors2:
                    if progress_callback:
                        progress_callback(f"Harmonisoidaan {emb_key}...")

                    t1, t2, final_vocab = self._harmonize_vocab_tensors(
                        tensors1[emb_key],
                        tensors2[emb_key],
                        vocab_size1,
                        vocab_size2,
                        strategy=config.vocab_strategy,
                        pad_value=config.vocab_pad_value,
                    )
                    tensors1[emb_key] = t1
                    tensors2[emb_key] = t2
                    final_vocab_size = final_vocab

            # 7. Suorita merge valitulla metodilla
            if progress_callback:
                progress_callback(f"Yhdistetaan {config.method.value}...")

            merged_tensors = {}
            common_keys = set(tensors1.keys()) & set(tensors2.keys())
            total_keys = len(common_keys)

            for i, key in enumerate(common_keys):
                t1 = tensors1[key]
                t2 = tensors2[key]

                # Tarkista onko embedding ja pitaako skipata
                is_embedding = key in embedding_keys

                if is_embedding:
                    if key == "model.embed_tokens.weight" and config.skip_embedding_merge:
                        merged_tensors[key] = t1  # Kayta ensimmaista
                        continue
                    if key == "lm_head.weight" and config.skip_lm_head_merge:
                        merged_tensors[key] = t1
                        continue

                # Tarkista sama koko
                if t1.shape != t2.shape:
                    if progress_callback:
                        progress_callback(f"[WARN] Ohitetaan {key}: {t1.shape} vs {t2.shape}")
                    merged_tensors[key] = t1
                    continue

                # Maarita ratio (eri embeddingille jos maaritelty)
                ratio = config.slerp_ratio
                if is_embedding and config.embedding_merge_ratio is not None:
                    ratio = config.embedding_merge_ratio

                # Merge
                if config.method == MergeMethod.SLERP:
                    merged_tensors[key] = self._slerp(ratio, t1, t2)
                else:
                    merged_tensors[key] = self._linear_interpolate(ratio, t1, t2)

                if progress_callback and i % 50 == 0:
                    progress_callback(f"Yhdistetaan... {i}/{total_keys}")

            # Lisaa avaimet jotka ovat vain toisessa mallissa
            for key in tensors1:
                if key not in merged_tensors:
                    merged_tensors[key] = tensors1[key]

            # 8. Luo output config
            output_config = self._merge_configs(config1, config2, config, final_vocab_size)

            # 9. Tallenna
            output_dir = self.output_dir / config.output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "model.safetensors"

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_safetensors(merged_tensors, output_file)

            # Tallenna config
            config_file = output_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(output_config, f, indent=2)

            # Kopioi tokenizer
            self._copy_advanced_tokenizer(model1_path, model2_path, output_dir, config)

            # Vapauta muisti
            del tensors1, tensors2, merged_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            output_size = output_file.stat().st_size / (1024**3)

            return {
                "success": True,
                "output_path": str(output_dir),
                "output_file": str(output_file),
                "file_size_gb": round(output_size, 2),
                "method": f"advanced_{config.method.value}",
                "final_vocab_size": final_vocab_size,
                "target_dtype": config.target_dtype,
                "original_vocab_sizes": [vocab_size1, vocab_size2],
            }

        except Exception as e:
            import traceback
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
