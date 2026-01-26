"""
AI TOOLBOX - Abliterator
========================

Main abliteration implementation for removing refusal behavior from models.

Abliteration works by:
1. Finding the "refusal direction" in activation space
2. Projecting this direction out of the model's weights
3. Resulting in a model that responds without censorship

WARNING: This tool removes safety features. Use responsibly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import json
import gc

from ..core.paths import get_paths
from .prompts import get_prompts, get_test_prompts
from .hooks import (
    register_activation_hooks,
    get_model_layer_count,
    get_recommended_layers,
    ActivationCache,
)


@dataclass
class AbliterationConfig:
    """Configuration for abliteration process."""

    model_path: str                           # Path to source model
    output_name: str                          # Name for abliterated model
    strength: float = 1.0                     # Abliteration strength (0.0-2.0)
    target_layers: Optional[List[int]] = None # Layers to modify (None = auto)
    method: str = "mean_diff"                 # "mean_diff" or "pca"
    num_harmful: int = 64                     # Number of harmful prompts (0 = all)
    num_harmless: int = 64                    # Number of harmless prompts (0 = all)
    include_llama31_prompts: bool = False     # Use Llama 3.1 specific prompts
    dtype: str = "float16"                    # Tensor dtype for computation

    # Custom prompt files (optional)
    harmful_prompts_file: Optional[str] = None   # Path to harmful prompts file
    harmless_prompts_file: Optional[str] = None  # Path to harmless prompts file

    # Performance options
    batch_size: int = 8                          # Prompt batch size for inference

    # Additional abliteration targets (optional, experimental)
    abliterate_embeddings: bool = False          # Also abliterate embed_tokens layer
    abliterate_lm_head: bool = False             # Also abliterate lm_head (output layer)


@dataclass
class AbliterationResult:
    """Result of abliteration process."""

    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    refusal_direction_norm: Optional[float] = None
    modified_layers: List[int] = field(default_factory=list)
    modified_weights: int = 0
    elapsed_seconds: float = 0.0
    method_used: str = ""
    strength_applied: float = 0.0


class Abliterator:
    """
    Abliteration tool for removing refusal behavior from language models.

    This works by finding and removing the "refusal direction" from model
    activations. The refusal direction is the vector that distinguishes
    harmful from harmless prompt processing.
    """

    def __init__(self):
        """Initialize the Abliterator."""
        self.paths = get_paths()
        self._torch = None
        self._safetensors = None

    def _clear_memory(self):
        """Clear GPU and CPU memory caches."""
        gc.collect()
        if self._torch and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Check for required dependencies."""
        deps = {
            "torch": False,
            "transformers": False,
            "safetensors": False,
            "sklearn": False,  # Optional, for PCA method
        }

        try:
            import torch
            self._torch = torch
            deps["torch"] = True
        except ImportError:
            pass

        try:
            import transformers
            deps["transformers"] = True
        except ImportError:
            pass

        try:
            import safetensors
            self._safetensors = safetensors
            deps["safetensors"] = True
        except ImportError:
            pass

        try:
            from sklearn.decomposition import PCA
            deps["sklearn"] = True
        except ImportError:
            pass

        return deps

    def get_status(self) -> Dict[str, Any]:
        """
        Get abliterator status.

        Returns:
            Status dictionary with readiness info
        """
        deps = self._check_dependencies()

        # Core requirements: torch, transformers, safetensors
        ready = deps["torch"] and deps["transformers"] and deps["safetensors"]

        missing = [k for k, v in deps.items() if not v and k != "sklearn"]

        return {
            "ready": ready,
            "dependencies": deps,
            "missing": missing,
            "output_dir": str(self.paths.root / "models" / "abliterated"),
            "pca_available": deps["sklearn"],
            "cuda_available": self._torch.cuda.is_available() if deps["torch"] else False,
        }

    def install_dependencies(
        self,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Install required dependencies.

        Args:
            progress_callback: Callback for progress messages

        Returns:
            True if successful
        """
        import subprocess
        import sys

        packages = [
            "torch",
            "transformers",
            "safetensors",
            "accelerate",
        ]

        try:
            for pkg in packages:
                if progress_callback:
                    progress_callback(f"Installing {pkg}...")

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            return True
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return False

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_path: Path to model directory or file

        Returns:
            Model information dictionary
        """
        model_path = Path(model_path)
        info = {
            "path": str(model_path),
            "name": model_path.name,
            "architecture": None,
            "num_layers": 0,
            "hidden_size": 0,
            "vocab_size": 0,
            "is_llama31": False,
        }

        # Try to read config.json
        config_file = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                info["architecture"] = config.get("architectures", [None])[0]
                info["num_layers"] = config.get("num_hidden_layers", 0)
                info["hidden_size"] = config.get("hidden_size", 0)
                info["vocab_size"] = config.get("vocab_size", 0)

                # Detect Llama 3.1
                model_type = config.get("model_type", "").lower()
                if "llama" in model_type:
                    # Llama 3.1 typically has specific vocab size and configs
                    if info["vocab_size"] >= 128000:
                        info["is_llama31"] = True
            except Exception:
                pass

        return info

    def extract_refusal_direction(
        self,
        config: AbliterationConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Extract the refusal direction vector from a model.

        This runs harmful and harmless prompts through the model,
        collects activations, and computes the difference vector.

        Args:
            config: Abliteration configuration
            progress_callback: Callback(message, progress_0_to_1)

        Returns:
            Result dict with refusal_direction tensor and metadata
        """
        if not self._torch:
            return {"success": False, "error": "PyTorch not available"}

        torch = self._torch

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return {"success": False, "error": "transformers not available"}

        model_path = Path(config.model_path)
        if not model_path.exists():
            return {"success": False, "error": f"Model not found: {model_path}"}

        try:
            if progress_callback:
                progress_callback("Loading model...", 0.0)

            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = getattr(torch, config.dtype, torch.float16)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model.eval()

            if progress_callback:
                progress_callback("Detecting layers...", 0.1)

            # Determine target layers
            num_layers = get_model_layer_count(model)
            target_layers = config.target_layers
            if target_layers is None:
                target_layers = get_recommended_layers(num_layers)

            if not target_layers:
                return {"success": False, "error": "Could not detect model layers"}

            if progress_callback:
                progress_callback(f"Targeting layers {target_layers[0]}-{target_layers[-1]}", 0.15)

            # Get prompts (from files if specified, else built-in)
            model_info = self.get_model_info(config.model_path)
            harmful, harmless = get_prompts(
                num_harmful=config.num_harmful,
                num_harmless=config.num_harmless,
                include_llama31=config.include_llama31_prompts or model_info.get("is_llama31", False),
                harmful_file=config.harmful_prompts_file,
                harmless_file=config.harmless_prompts_file,
            )

            # Log prompt source
            if config.harmful_prompts_file:
                source = f"custom files ({len(harmful)} harmful, {len(harmless)} harmless)"
            else:
                source = f"built-in ({len(harmful)} harmful, {len(harmless)} harmless)"

            if progress_callback:
                progress_callback(f"Using {source}...", 0.18)
                progress_callback(f"Collecting harmful activations (batch_size={config.batch_size})...", 0.2)

            # Collect harmful activations using batched processing
            def harmful_progress(msg, prog):
                if progress_callback:
                    progress_callback(f"Harmful: {msg}", 0.2 + prog * 0.3)

            harmful_acts = self._run_prompts_batched(
                model, tokenizer, harmful, device,
                batch_size=config.batch_size,
                progress_callback=harmful_progress
            )

            # Clear memory after harmful processing
            self._clear_memory()

            if progress_callback:
                progress_callback(f"Collecting harmless activations (batch_size={config.batch_size})...", 0.5)

            # Collect harmless activations using batched processing
            def harmless_progress(msg, prog):
                if progress_callback:
                    progress_callback(f"Harmless: {msg}", 0.5 + prog * 0.3)

            harmless_acts = self._run_prompts_batched(
                model, tokenizer, harmless, device,
                batch_size=config.batch_size,
                progress_callback=harmless_progress
            )

            # Clear memory after harmless processing
            self._clear_memory()

            if progress_callback:
                progress_callback("Computing refusal direction...", 0.85)

            # Compute refusal direction for each target layer
            refusal_directions = {}
            for layer in target_layers:
                if layer in harmful_acts and layer in harmless_acts:
                    direction = self._compute_refusal_vector(
                        harmful_acts[layer],
                        harmless_acts[layer],
                        config.method
                    )
                    refusal_directions[layer] = direction

            if progress_callback:
                progress_callback("Cleaning up...", 0.95)

            # Clean up model and memory
            del model, harmful_acts, harmless_acts
            self._clear_memory()

            return {
                "success": True,
                "refusal_directions": refusal_directions,
                "target_layers": target_layers,
                "num_layers": num_layers,
                "hidden_size": model_info.get("hidden_size", 0),
                "method": config.method,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_prompts_batched(
        self,
        model,
        tokenizer,
        prompts: List[str],
        device: str,
        batch_size: int = 8,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[int, Any]:
        """
        Run prompts through model in batches using Welford's algorithm for online mean.

        This is much faster than processing one prompt at a time, and memory efficient
        since we compute running means instead of storing all activations.

        Args:
            model: The loaded model
            tokenizer: The tokenizer
            prompts: List of prompts to process
            device: Device to use ('cuda' or 'cpu')
            batch_size: Number of prompts per batch
            progress_callback: Optional progress callback

        Returns:
            Dict of layer_idx -> mean activation tensor
        """
        torch = self._torch

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        means: Dict[int, Any] = {}
        counts: Dict[int, int] = {}

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(prompts), batch_size):
            batch = prompts[batch_idx:batch_idx + batch_size]
            current_batch_num = batch_idx // batch_size + 1

            if progress_callback:
                progress = current_batch_num / total_batches
                progress_callback(f"Batch {current_batch_num}/{total_batches}", progress)

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # Process hidden states from all layers
                # hidden_states is a tuple of (num_layers + 1) tensors:
                #   hidden_states[0] = embedding output (BEFORE layer 0)
                #   hidden_states[1] = layer 0 OUTPUT
                #   hidden_states[N+1] = layer N OUTPUT
                # Each tensor has shape [batch_size, seq_len, hidden_size]
                hidden_states = outputs.hidden_states

                for hs_idx, hidden in enumerate(hidden_states):
                    # Skip embedding layer output (index 0)
                    # We want layer N's OUTPUT which is at hidden_states[N+1]
                    if hs_idx == 0:
                        continue

                    # actual_layer is the transformer layer number (0, 1, 2, ...)
                    # hidden_states[1] = layer 0 output, hidden_states[2] = layer 1 output, etc.
                    actual_layer = hs_idx - 1

                    # Take last token position, convert to float32 for numerical stability
                    current = hidden[:, -1, :].float().cpu()
                    batch_mean = current.mean(dim=0)

                    if actual_layer not in means:
                        means[actual_layer] = batch_mean
                        counts[actual_layer] = len(batch)
                    else:
                        # Welford's online algorithm for computing mean
                        total = counts[actual_layer] + len(batch)
                        delta = batch_mean - means[actual_layer]
                        means[actual_layer] = means[actual_layer] + delta * len(batch) / total
                        counts[actual_layer] = total

                # Cleanup batch tensors
                del inputs, outputs, hidden_states
                self._clear_memory()

            except Exception as e:
                # Log but continue on errors
                if progress_callback:
                    progress_callback(f"Warning: batch failed - {str(e)[:50]}", 0)
                continue

        return means

    def _run_prompts(self, model, tokenizer, prompts: List[str], device: str):
        """Legacy single-prompt processing (kept for compatibility)."""
        torch = self._torch

        for prompt in prompts:
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )

                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    model(**inputs)

            except Exception:
                continue

    def _compute_refusal_vector(
        self,
        harmful_act,
        harmless_act,
        method: str = "mean_diff"
    ):
        """
        Compute refusal direction from activation differences.

        Args:
            harmful_act: Mean activation from harmful prompts
            harmless_act: Mean activation from harmless prompts
            method: "mean_diff" or "pca"

        Returns:
            Normalized refusal direction vector
        """
        torch = self._torch

        if method == "mean_diff":
            # Simple: difference of means
            refusal_dir = harmful_act - harmless_act
            refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-8)

        elif method == "pca":
            # Use PCA to find principal component
            try:
                from sklearn.decomposition import PCA
                import numpy as np

                diff = (harmful_act - harmless_act).cpu().numpy()
                if diff.ndim == 1:
                    diff = diff.reshape(1, -1)

                # Just use the normalized difference for single samples
                refusal_dir = torch.from_numpy(diff[0])
                refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-8)
            except ImportError:
                # Fallback to mean_diff
                refusal_dir = harmful_act - harmless_act
                refusal_dir = refusal_dir / (refusal_dir.norm() + 1e-8)
        else:
            raise ValueError(f"Unknown method: {method}")

        return refusal_dir

    def apply_abliteration(
        self,
        config: AbliterationConfig,
        refusal_directions: Dict[int, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AbliterationResult:
        """
        Apply abliteration to model weights.

        Args:
            config: Abliteration configuration
            refusal_directions: Dict of layer -> refusal direction vector
            progress_callback: Callback(message, progress_0_to_1)

        Returns:
            AbliterationResult
        """
        import time
        start_time = time.time()

        if not self._torch or not self._safetensors:
            return AbliterationResult(
                success=False,
                error="Required dependencies not available"
            )

        torch = self._torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        model_path = Path(config.model_path)
        output_dir = self.paths.root / "models" / "abliterated" / config.output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if progress_callback:
                progress_callback("Finding weight files...", 0.0)

            # Find safetensors files
            st_files = list(model_path.glob("*.safetensors"))
            if not st_files:
                return AbliterationResult(
                    success=False,
                    error="No safetensors files found"
                )

            modified_count = 0
            modified_layers = set()

            # Weight keys to target for abliteration
            # IMPORTANT: Only target OUTPUT projections (layers that output to residual stream)
            # The refusal direction lives in the output/residual space, so we only modify:
            # - down_proj: MLP output → residual stream
            # - o_proj: Attention output → residual stream
            # DO NOT ablate input projections (up_proj, gate_proj, q/k/v_proj) as this
            # modifies how the model interprets inputs and can break the model!
            target_patterns = [
                "mlp.down_proj.weight",
                "self_attn.o_proj.weight",
            ]

            # Compute combined refusal direction for non-layer-specific weights
            # (embed_tokens, lm_head) by averaging directions from all target layers
            combined_refusal_dir = None
            if config.abliterate_embeddings or config.abliterate_lm_head:
                all_dirs = list(refusal_directions.values())
                if all_dirs:
                    combined_refusal_dir = torch.stack(all_dirs).mean(dim=0)
                    combined_refusal_dir = torch.nn.functional.normalize(combined_refusal_dir, dim=0)

            for i, st_file in enumerate(st_files):
                if progress_callback:
                    progress_callback(
                        f"Processing {st_file.name}...",
                        0.1 + 0.8 * (i / len(st_files))
                    )

                modified_tensors = {}

                with safe_open(st_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)

                        # Check if this is a layer-specific weight that should be modified
                        layer_num = self._extract_layer_from_key(key)
                        should_modify_layer = (
                            layer_num is not None and
                            layer_num in refusal_directions and
                            any(p in key for p in target_patterns)
                        )

                        # Check for embed_tokens (if enabled)
                        is_embed_tokens = (
                            config.abliterate_embeddings and
                            combined_refusal_dir is not None and
                            "embed_tokens.weight" in key
                        )

                        # Check for lm_head (if enabled)
                        is_lm_head = (
                            config.abliterate_lm_head and
                            combined_refusal_dir is not None and
                            "lm_head.weight" in key
                        )

                        if should_modify_layer:
                            refusal_dir = refusal_directions[layer_num]
                            tensor = self._apply_to_weight(
                                tensor,
                                refusal_dir,
                                config.strength,
                                device="cuda"
                            )
                            modified_count += 1
                            modified_layers.add(layer_num)

                        elif is_embed_tokens:
                            # Abliterate embedding layer with combined direction
                            tensor = self._apply_to_weight(
                                tensor,
                                combined_refusal_dir,
                                config.strength,
                                device="cuda"
                            )
                            modified_count += 1

                        elif is_lm_head:
                            # Abliterate output layer with combined direction
                            tensor = self._apply_to_weight(
                                tensor,
                                combined_refusal_dir,
                                config.strength,
                                device="cuda"
                            )
                            modified_count += 1

                        modified_tensors[key] = tensor

                # Save modified shard and cleanup
                output_file = output_dir / st_file.name
                save_file(modified_tensors, str(output_file))

                # Clean up this shard's tensors from memory
                del modified_tensors
                self._clear_memory()

            if progress_callback:
                progress_callback("Copying config files...", 0.95)

            # Copy config files
            for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                               "special_tokens_map.json", "generation_config.json"]:
                src = model_path / config_file
                if src.exists():
                    dst = output_dir / config_file
                    dst.write_bytes(src.read_bytes())

            # Copy tokenizer model if exists
            for tokenizer_file in model_path.glob("*.model"):
                dst = output_dir / tokenizer_file.name
                dst.write_bytes(tokenizer_file.read_bytes())

            # Save abliteration metadata
            metadata = {
                "source_model": str(model_path),
                "abliteration_config": {
                    "strength": config.strength,
                    "method": config.method,
                    "target_layers": list(modified_layers),
                    "num_harmful": config.num_harmful,
                    "num_harmless": config.num_harmless,
                },
                "timestamp": datetime.now().isoformat(),
                "modified_weights": modified_count,
            }
            with open(output_dir / "abliteration_info.json", "w") as f:
                json.dump(metadata, f, indent=2)

            if progress_callback:
                progress_callback("Done!", 1.0)

            return AbliterationResult(
                success=True,
                output_path=str(output_dir),
                modified_layers=sorted(list(modified_layers)),
                modified_weights=modified_count,
                elapsed_seconds=time.time() - start_time,
                method_used=config.method,
                strength_applied=config.strength,
            )

        except Exception as e:
            return AbliterationResult(
                success=False,
                error=str(e),
                elapsed_seconds=time.time() - start_time,
            )

    def _extract_layer_from_key(self, key: str) -> Optional[int]:
        """Extract layer number from weight key."""
        import re
        match = re.search(r'layers\.(\d+)', key)
        if match:
            return int(match.group(1))
        return None

    def _apply_to_weight(self, weight, refusal_dir, strength: float, device: str = "cuda"):
        """
        Apply abliteration to a weight tensor using GPU acceleration.

        Projects out the refusal direction from the weight matrix.
        Uses optimized O(n²) algorithm instead of O(n³) matrix multiplication.

        Math: W_new = W - strength * (W @ r) ⊗ r^T  (for input projection)
              W_new = W - strength * r ⊗ (r^T @ W)  (for output projection)

        Features:
        - Moves tensors to GPU for fast computation
        - Uses float32 for numerical stability
        - Returns result back to CPU in original dtype

        Args:
            weight: Weight tensor [out_features, in_features] or [features]
            refusal_dir: Refusal direction vector [hidden_size]
            strength: Abliteration strength (0-2, 1 = full removal)
            device: Device to use for computation ("cuda" or "cpu")

        Returns:
            Modified weight tensor (on CPU)
        """
        torch = self._torch
        original_dtype = weight.dtype

        # Check if GPU is available
        use_gpu = device == "cuda" and torch.cuda.is_available()
        compute_device = "cuda" if use_gpu else "cpu"

        # Move to compute device and convert to float32 for numerical stability
        weight_gpu = weight.to(compute_device, dtype=torch.float32, non_blocking=True)
        refusal_gpu = refusal_dir.to(compute_device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is 1D (fix for PCA method returning 2D tensor)
        if refusal_gpu.dim() > 1:
            refusal_gpu = refusal_gpu.flatten()

        # Normalize refusal direction (matches miniagentti reference implementation)
        refusal_gpu = torch.nn.functional.normalize(refusal_gpu, dim=0)

        if weight_gpu.dim() == 2:
            # Matrix weight [out, in]
            out_features, in_features = weight_gpu.shape

            if in_features == refusal_gpu.shape[-1]:
                # Project refusal direction out of input space
                # Optimized: W @ (r ⊗ r^T) = (W @ r) ⊗ r^T
                # This is O(out * in) instead of O(out * in * in)
                dot = weight_gpu @ refusal_gpu  # [out]
                weight_gpu = weight_gpu - strength * torch.outer(dot, refusal_gpu)

            elif out_features == refusal_gpu.shape[-1]:
                # Project refusal direction out of output space
                # Optimized: (r ⊗ r^T) @ W = r ⊗ (r^T @ W)
                # This is O(out * in) instead of O(out * out * in)
                dot = refusal_gpu @ weight_gpu  # [in]
                weight_gpu = weight_gpu - strength * torch.outer(refusal_gpu, dot)

        elif weight_gpu.dim() == 1:
            # Bias vector
            if weight_gpu.shape[0] == refusal_gpu.shape[-1]:
                proj = (weight_gpu @ refusal_gpu) * refusal_gpu
                weight_gpu = weight_gpu - strength * proj

        # Move back to CPU with original dtype
        result = weight_gpu.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup GPU memory
        del weight_gpu, refusal_gpu

        return result

    def full_abliteration(
        self,
        config: AbliterationConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> AbliterationResult:
        """
        Perform full abliteration: extract refusal direction and apply.

        Args:
            config: Abliteration configuration
            progress_callback: Callback(message, progress_0_to_1)

        Returns:
            AbliterationResult
        """
        # Phase 1: Extract refusal direction
        if progress_callback:
            progress_callback("Phase 1: Extracting refusal direction...", 0.0)

        def extract_progress(msg, prog):
            if progress_callback:
                progress_callback(f"Extract: {msg}", prog * 0.5)

        extract_result = self.extract_refusal_direction(config, extract_progress)

        if not extract_result["success"]:
            return AbliterationResult(
                success=False,
                error=f"Extraction failed: {extract_result.get('error', 'Unknown')}"
            )

        # Phase 2: Apply to weights
        if progress_callback:
            progress_callback("Phase 2: Applying abliteration...", 0.5)

        def apply_progress(msg, prog):
            if progress_callback:
                progress_callback(f"Apply: {msg}", 0.5 + prog * 0.5)

        return self.apply_abliteration(
            config,
            extract_result["refusal_directions"],
            apply_progress
        )

    def estimate_requirements(self, model_path: str) -> Dict[str, Any]:
        """
        Estimate computational requirements for abliteration.

        Args:
            model_path: Path to model

        Returns:
            Requirements estimate
        """
        info = self.get_model_info(model_path)

        # Rough estimates based on model size
        hidden_size = info.get("hidden_size", 4096)
        num_layers = info.get("num_layers", 32)

        # Estimate model size (very rough)
        param_count = hidden_size * hidden_size * num_layers * 4  # Very approximate
        model_size_gb = param_count * 2 / (1024**3)  # FP16

        return {
            "model_info": info,
            "estimated_model_size_gb": round(model_size_gb, 1),
            "estimated_ram_gb": round(model_size_gb * 2.5, 1),  # Need headroom
            "estimated_vram_gb": round(model_size_gb * 1.2, 1),
            "recommended_device": "cuda" if model_size_gb < 12 else "cpu",
        }
