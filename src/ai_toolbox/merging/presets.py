"""
AI TOOLBOX - Merge Presets
==========================

Valmiit merge-konfiguraatiot yleisiin tilanteisiin.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from pathlib import Path


class PresetCategory(str, Enum):
    """Preset-kategoriat."""
    GENERAL = "general"              # Yleiset
    LANGUAGE = "language"            # Kielimallien yhdistaminen
    REASONING = "reasoning"          # Reasoning-kyvykkyydet
    INSTRUCTION = "instruction"      # Instruction-following
    CREATIVE = "creative"            # Luova kirjoittaminen
    CODING = "coding"                # Koodin generointi


@dataclass
class MergePreset:
    """Merge preset maarittely."""
    name: str
    description: str
    category: PresetCategory
    method: str

    # Oletusparametrit
    default_params: Dict[str, Any] = field(default_factory=dict)

    # Suositellut mallit (avainsanoja mallien nimissa)
    recommended_keywords: List[str] = field(default_factory=list)

    # Rajoitukset
    min_models: int = 2
    max_models: int = 10
    requires_base: bool = False

    def to_config_dict(
        self,
        models: List[str],
        base_model: Optional[str] = None,
        dtype: str = "bfloat16",
        **overrides,
    ) -> Dict[str, Any]:
        """
        Muunna mergekit config dictiksi.

        Args:
            models: Lista mallipoluista
            base_model: Base model polku
            dtype: Datan tyyppi
            **overrides: Parametrien ylikirjoitukset

        Returns:
            Mergekit-yhteensopiva config dict
        """
        # Yhdista oletusparametrit ja overridet
        params = {**self.default_params, **overrides}

        config: Dict[str, Any] = {
            "merge_method": self.method,
            "dtype": dtype,
        }

        # Base model
        if base_model:
            config["base_model"] = base_model
        elif self.requires_base and models:
            config["base_model"] = models[0]

        # Metodikohtainen rakenne
        if self.method == "slerp":
            config["models"] = [{"model": m} for m in models[:2]]
            config["parameters"] = {"t": params.get("t", 0.5)}

        elif self.method in {"dare_ties", "dare_linear", "ties"}:
            models_list = []
            density = params.get("density", 0.5)
            weights = params.get("weights", [1.0 / len(models)] * len(models))

            for i, model in enumerate(models):
                model_def: Dict[str, Any] = {"model": model}
                model_params: Dict[str, Any] = {}

                if i < len(weights):
                    model_params["weight"] = weights[i]
                if self.method in {"dare_ties", "dare_linear"}:
                    model_params["density"] = density

                if model_params:
                    model_def["parameters"] = model_params
                models_list.append(model_def)

            config["models"] = models_list
            config["parameters"] = {
                "normalize": params.get("normalize", True),
            }
            if self.method in {"dare_ties", "dare_linear"}:
                config["parameters"]["int8_mask"] = params.get("int8_mask", True)

        elif self.method == "task_arithmetic":
            models_list = []
            weights = params.get("weights", [1.0 / len(models)] * len(models))

            for i, model in enumerate(models):
                model_def: Dict[str, Any] = {"model": model}
                if i < len(weights):
                    model_def["parameters"] = {"weight": weights[i]}
                models_list.append(model_def)

            config["models"] = models_list
            config["parameters"] = {"normalize": params.get("normalize", False)}

        elif self.method == "linear":
            models_list = []
            weights = params.get("weights", [1.0 / len(models)] * len(models))

            for i, model in enumerate(models):
                model_def: Dict[str, Any] = {"model": model}
                if i < len(weights):
                    model_def["parameters"] = {"weight": weights[i]}
                models_list.append(model_def)

            config["models"] = models_list
            config["parameters"] = {"normalize": params.get("normalize", True)}

        elif self.method == "della":
            models_list = []
            density = params.get("density", 0.5)
            weights = params.get("weights", [1.0 / len(models)] * len(models))

            for i, model in enumerate(models):
                model_def: Dict[str, Any] = {"model": model}
                model_params: Dict[str, Any] = {}
                if i < len(weights):
                    model_params["weight"] = weights[i]
                model_params["density"] = density
                model_def["parameters"] = model_params
                models_list.append(model_def)

            config["models"] = models_list
            config["parameters"] = {
                "normalize": params.get("normalize", True),
                "rescale": params.get("rescale", True),
            }

        else:
            config["models"] = [{"model": m} for m in models]

        return config


# =============================================================================
# Valmiit presetit
# =============================================================================

PRESETS: Dict[str, MergePreset] = {

    # =========================================================================
    # SLERP Presets
    # =========================================================================

    "slerp_balanced": MergePreset(
        name="SLERP Balanced",
        description="Tasapainoinen 50/50 SLERP merge kahdelle mallille",
        category=PresetCategory.GENERAL,
        method="slerp",
        default_params={"t": 0.5},
        recommended_keywords=["*"],
        min_models=2,
        max_models=2,
    ),

    "slerp_light": MergePreset(
        name="SLERP Light Blend",
        description="Kevyt blend: 70% malli 1 / 30% malli 2",
        category=PresetCategory.GENERAL,
        method="slerp",
        default_params={"t": 0.3},
        recommended_keywords=["*"],
        min_models=2,
        max_models=2,
    ),

    "slerp_heavy": MergePreset(
        name="SLERP Heavy Blend",
        description="Voimakas blend: 30% malli 1 / 70% malli 2",
        category=PresetCategory.GENERAL,
        method="slerp",
        default_params={"t": 0.7},
        recommended_keywords=["*"],
        min_models=2,
        max_models=2,
    ),

    # =========================================================================
    # DARE-TIES Presets
    # =========================================================================

    "dare_ties_balanced": MergePreset(
        name="DARE-TIES Balanced",
        description="Tasapainoinen DARE-TIES 2+ mallille",
        category=PresetCategory.GENERAL,
        method="dare_ties",
        default_params={
            "density": 0.5,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    "dare_ties_language": MergePreset(
        name="DARE-TIES Language Transfer",
        description="Optimoitu kielten yhdistamiseen (esim. FI + EN)",
        category=PresetCategory.LANGUAGE,
        method="dare_ties",
        default_params={
            "density": 0.6,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["poro", "viking", "finllama", "llama", "mistral"],
        requires_base=True,
    ),

    "dare_ties_conservative": MergePreset(
        name="DARE-TIES Conservative",
        description="Varovainen merge, sailyttaa enemman alkuperaisia painoja",
        category=PresetCategory.GENERAL,
        method="dare_ties",
        default_params={
            "density": 0.7,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    "dare_ties_aggressive": MergePreset(
        name="DARE-TIES Aggressive",
        description="Aggressiivinen merge, enemman uusia ominaisuuksia",
        category=PresetCategory.GENERAL,
        method="dare_ties",
        default_params={
            "density": 0.3,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    # =========================================================================
    # TIES Presets
    # =========================================================================

    "ties_standard": MergePreset(
        name="TIES Standard",
        description="Vakio TIES merge 2+ mallille",
        category=PresetCategory.GENERAL,
        method="ties",
        default_params={
            "density": 0.5,
            "normalize": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    # =========================================================================
    # Task Arithmetic Presets
    # =========================================================================

    "task_arithmetic_additive": MergePreset(
        name="Task Arithmetic Additive",
        description="Additiivinen task vector merge",
        category=PresetCategory.INSTRUCTION,
        method="task_arithmetic",
        default_params={
            "normalize": False,
        },
        recommended_keywords=["instruct", "chat", "assistant"],
        requires_base=True,
    ),

    "task_arithmetic_normalized": MergePreset(
        name="Task Arithmetic Normalized",
        description="Normalisoitu task arithmetic merge",
        category=PresetCategory.INSTRUCTION,
        method="task_arithmetic",
        default_params={
            "normalize": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    # =========================================================================
    # LINEAR Presets
    # =========================================================================

    "linear_average": MergePreset(
        name="Linear Average",
        description="Yksinkertainen painotettu keskiarvo",
        category=PresetCategory.GENERAL,
        method="linear",
        default_params={
            "normalize": True,
        },
        recommended_keywords=["*"],
        requires_base=False,
    ),

    # =========================================================================
    # DELLA Presets
    # =========================================================================

    "della_efficient": MergePreset(
        name="DELLA Efficient",
        description="DELLA - tehokas pruning + merge",
        category=PresetCategory.GENERAL,
        method="della",
        default_params={
            "density": 0.5,
            "normalize": True,
            "rescale": True,
        },
        recommended_keywords=["*"],
        requires_base=True,
    ),

    # =========================================================================
    # Specialized Presets
    # =========================================================================

    "reasoning_boost": MergePreset(
        name="Reasoning Boost",
        description="Optimoitu reasoning-kyvykkyyden lisaamiseen",
        category=PresetCategory.REASONING,
        method="slerp",
        default_params={"t": 0.35},
        recommended_keywords=["deepseek", "r1", "qwq", "reason", "cot"],
        min_models=2,
        max_models=2,
    ),

    "coding_merge": MergePreset(
        name="Coding Specialist",
        description="Koodauskyvykkyyden yhdistaminen",
        category=PresetCategory.CODING,
        method="dare_ties",
        default_params={
            "density": 0.5,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["code", "coder", "deepseek", "starcoder", "codellama"],
        requires_base=True,
    ),

    "creative_writing": MergePreset(
        name="Creative Writing",
        description="Luovan kirjoittamisen optimointi",
        category=PresetCategory.CREATIVE,
        method="slerp",
        default_params={"t": 0.45},
        recommended_keywords=["writing", "story", "creative", "nous", "hermes"],
        min_models=2,
        max_models=2,
    ),

    "finnish_llama": MergePreset(
        name="Finnish LLaMA Merge",
        description="Suomenkielisen Llama-mallin luonti",
        category=PresetCategory.LANGUAGE,
        method="dare_ties",
        default_params={
            "density": 0.55,
            "normalize": True,
            "int8_mask": True,
        },
        recommended_keywords=["poro", "viking", "finllama", "llama"],
        requires_base=True,
    ),
}


def get_preset(name: str) -> Optional[MergePreset]:
    """Hae preset nimella."""
    return PRESETS.get(name)


def list_presets(category: Optional[PresetCategory] = None) -> List[MergePreset]:
    """
    Listaa presetit, optionaalisesti kategorialla.

    Args:
        category: Suodata kategorian mukaan

    Returns:
        Lista preseteista
    """
    if category:
        return [p for p in PRESETS.values() if p.category == category]
    return list(PRESETS.values())


def get_presets_by_category() -> Dict[PresetCategory, List[MergePreset]]:
    """
    Ryhmittele presetit kategorioittain.

    Returns:
        Dict[kategoria] = [presetit]
    """
    result: Dict[PresetCategory, List[MergePreset]] = {}
    for preset in PRESETS.values():
        if preset.category not in result:
            result[preset.category] = []
        result[preset.category].append(preset)
    return result


def get_recommended_preset(
    models: List[str],
    task: Optional[str] = None,
) -> Optional[MergePreset]:
    """
    Suosittele preset mallien ja tehtavan perusteella.

    Analysoi mallien nimia ja ehdottaa sopivaa presetia.

    Args:
        models: Lista mallien nimista/poluista
        task: Valinnainen tehtavakuvaus

    Returns:
        Suositeltu preset tai None
    """
    model_names_lower = [str(m).lower() for m in models]

    # Etsi avainsanojen perusteella
    best_match: Optional[MergePreset] = None
    best_score = 0

    for preset in PRESETS.values():
        if not preset.recommended_keywords or "*" in preset.recommended_keywords:
            continue

        score = 0
        for keyword in preset.recommended_keywords:
            for model_name in model_names_lower:
                if keyword.lower() in model_name:
                    score += 1

        if score > best_score:
            best_score = score
            best_match = preset

    # Jos ei loydy avainsanoilla, kayta oletusta mallien maaran mukaan
    if not best_match:
        if len(models) == 2:
            best_match = PRESETS.get("slerp_balanced")
        else:
            best_match = PRESETS.get("dare_ties_balanced")

    return best_match


def get_preset_choices_for_cli() -> List[Dict[str, Any]]:
    """
    Palauta presetit CLI-valikkoa varten.

    Returns:
        Lista: [{"title": str, "value": str, "category": str}]
    """
    choices = []
    by_category = get_presets_by_category()

    for category in PresetCategory:
        if category in by_category:
            for preset in by_category[category]:
                choices.append({
                    "title": f"{preset.name:<25} {preset.description}",
                    "value": list(PRESETS.keys())[list(PRESETS.values()).index(preset)],
                    "category": category.value,
                })

    return choices
