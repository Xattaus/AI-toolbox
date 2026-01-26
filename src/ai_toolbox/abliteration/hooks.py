"""
AI TOOLBOX - Activation Hooks
=============================

PyTorch hooks for collecting model activations during forward pass.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import re


@dataclass
class ActivationCache:
    """Cache for storing activations from model layers."""

    activations: Dict[int, List[Any]] = field(default_factory=dict)
    hooks: List[Any] = field(default_factory=list)
    target_layers: List[int] = field(default_factory=list)

    def clear(self):
        """Clear all cached activations."""
        self.activations = {layer: [] for layer in self.target_layers}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_mean_activations(self, device=None):
        """
        Get mean activations per layer.

        Args:
            device: Device to move tensors to

        Returns:
            Dict mapping layer number to mean activation tensor
        """
        try:
            import torch
        except ImportError:
            return {}

        result = {}
        for layer, acts in self.activations.items():
            if acts:
                stacked = torch.stack(acts)
                mean_act = stacked.mean(dim=0)
                if device is not None:
                    mean_act = mean_act.to(device)
                result[layer] = mean_act
        return result


def parse_layer_number(name: str) -> Optional[int]:
    """
    Extract layer number from module name.

    Supports patterns like:
    - model.layers.15.mlp
    - transformer.h.15.mlp
    - decoder.layers.15

    Args:
        name: Module name string

    Returns:
        Layer number or None if not found
    """
    patterns = [
        r'layers\.(\d+)',           # Llama, Mistral
        r'h\.(\d+)',                # GPT-2, GPT-Neo
        r'decoder\.layers\.(\d+)', # Some T5 variants
        r'encoder\.layers\.(\d+)', # BERT, etc
        r'block\.(\d+)',           # Some custom models
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def get_target_module_patterns() -> List[str]:
    """
    Get patterns for modules to hook for activation collection.

    The refusal direction is typically found in:
    - MLP output projections (down_proj)
    - Attention output projections (o_proj)
    - Residual stream

    Returns:
        List of pattern strings to match module names
    """
    return [
        "mlp.down_proj",      # Llama, Mistral, Qwen
        "mlp.c_proj",         # GPT-2, GPT-Neo
        "mlp.dense_4h_to_h",  # GPT-NeoX
        "mlp.wo",             # Some custom
        "feed_forward.w2",    # Alternative naming
    ]


def create_activation_hook(
    cache: ActivationCache,
    layer_num: int,
    extract_last_token: bool = True
) -> Callable:
    """
    Create a forward hook for capturing activations.

    Args:
        cache: ActivationCache to store activations
        layer_num: Layer number for this hook
        extract_last_token: If True, only capture last token activation

    Returns:
        Hook function
    """
    def hook(module, input, output):
        try:
            import torch

            # Handle different output formats
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            # Extract last token if requested (for causal LMs)
            if extract_last_token and activation.dim() >= 2:
                # Shape: [batch, seq_len, hidden] -> [batch, hidden]
                activation = activation[:, -1, :]

            # Detach and store
            cache.activations[layer_num].append(activation.detach().cpu())

        except Exception as e:
            # Silent fail to not interrupt inference
            pass

    return hook


def register_activation_hooks(
    model,
    target_layers: List[int],
    module_patterns: Optional[List[str]] = None,
    extract_last_token: bool = True
) -> ActivationCache:
    """
    Register forward hooks on specified layers.

    Args:
        model: PyTorch model
        target_layers: List of layer numbers to hook
        module_patterns: Patterns to match module names (default: MLP down_proj)
        extract_last_token: Only capture last token activation

    Returns:
        ActivationCache with registered hooks
    """
    if module_patterns is None:
        module_patterns = get_target_module_patterns()

    cache = ActivationCache(
        activations={layer: [] for layer in target_layers},
        target_layers=target_layers
    )

    for name, module in model.named_modules():
        layer_num = parse_layer_number(name)

        if layer_num is None or layer_num not in target_layers:
            continue

        # Check if module matches target patterns
        if not any(pattern in name for pattern in module_patterns):
            continue

        # Create and register hook
        hook_fn = create_activation_hook(cache, layer_num, extract_last_token)
        hook = module.register_forward_hook(hook_fn)
        cache.hooks.append(hook)

    return cache


def get_model_layer_count(model) -> int:
    """
    Detect the number of layers in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of layers detected
    """
    max_layer = -1

    for name, _ in model.named_modules():
        layer_num = parse_layer_number(name)
        if layer_num is not None:
            max_layer = max(max_layer, layer_num)

    return max_layer + 1 if max_layer >= 0 else 0


def get_recommended_layers(num_layers: int) -> List[int]:
    """
    Get recommended layers for refusal direction extraction.

    Typically the middle-to-late layers contain the strongest
    refusal direction signal.

    Args:
        num_layers: Total number of layers in model

    Returns:
        List of recommended layer indices
    """
    if num_layers <= 0:
        return []

    # Target middle 40% of layers (from 30% to 70%)
    start = int(num_layers * 0.3)
    end = int(num_layers * 0.7)

    # Ensure at least a few layers
    if end - start < 4:
        start = max(0, num_layers // 2 - 2)
        end = min(num_layers, num_layers // 2 + 2)

    return list(range(start, end))
