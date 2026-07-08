"""
AI TOOLBOX - HuggingFace Search Filters
=======================================

Filter constants and configurations for HuggingFace model search.
Includes task categories, libraries, applications, and metadata definitions.
"""

from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# TASK CATEGORIES - Organized by modality
# =============================================================================

TASK_CATEGORIES: Dict[str, Dict[str, str]] = {
    "NLP": {
        "text-generation": "Tekstin generointi (LLM)",
        "text2text-generation": "Teksti-tekstiksi",
        "text-classification": "Tekstin luokittelu",
        "token-classification": "Tokenien luokittelu (NER)",
        "question-answering": "Kysymys-vastaus",
        "translation": "Kaannos",
        "summarization": "Tiivistaminen",
        "fill-mask": "Taydennys (MLM)",
        "conversational": "Keskustelu",
        "table-question-answering": "Taulukko-QA",
        "sentence-similarity": "Lauseiden samankaltaisuus",
        "zero-shot-classification": "Zero-shot luokittelu",
    },
    "Audio": {
        "text-to-speech": "Teksti puheeksi (TTS)",
        "automatic-speech-recognition": "Puheentunnistus (ASR)",
        "audio-classification": "Audion luokittelu",
        "audio-to-audio": "Audio-audioksi",
        "voice-activity-detection": "Puheen tunnistus (VAD)",
    },
    "Vision": {
        "image-classification": "Kuvan luokittelu",
        "object-detection": "Kohteiden tunnistus",
        "image-segmentation": "Kuvan segmentointi",
        "text-to-image": "Teksti kuvaksi",
        "image-to-image": "Kuva kuvaksi",
        "image-to-text": "Kuva tekstiksi",
        "unconditional-image-generation": "Kuvan generointi",
        "depth-estimation": "Syvyysarviointi",
        "image-feature-extraction": "Kuvan piirteet",
        "video-classification": "Videon luokittelu",
        "text-to-video": "Teksti videoksi",
        "zero-shot-image-classification": "Zero-shot kuvaluokittelu",
        "mask-generation": "Maskin generointi",
        "zero-shot-object-detection": "Zero-shot tunnistus",
        "image-to-3d": "Kuva 3D-malliksi",
        "text-to-3d": "Teksti 3D-malliksi",
    },
    "Multimodal": {
        "image-text-to-text": "Kuva+teksti tekstiksi (VLM)",
        "visual-question-answering": "Visuaalinen QA",
        "document-question-answering": "Dokumentti-QA",
        "feature-extraction": "Piirteiden poiminta",
        "video-text-to-text": "Video+teksti tekstiksi",
        "any-to-any": "Universaali",
    },
    "Tabular": {
        "tabular-classification": "Taulukon luokittelu",
        "tabular-regression": "Taulukon regressio",
        "time-series-forecasting": "Aikasarjaennustus",
    },
    "Reinforcement Learning": {
        "reinforcement-learning": "Vahvistusoppiminen",
        "robotics": "Robotiikka",
    },
}

# Flat list of all tasks for validation
ALL_TASKS: List[str] = [task for category in TASK_CATEGORIES.values() for task in category.keys()]

# =============================================================================
# LIBRARIES AND FORMATS
# =============================================================================

LIBRARIES: Dict[str, str] = {
    # Core frameworks
    "transformers": "HuggingFace Transformers",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "jax": "JAX/Flax",
    # Model formats
    "gguf": "GGUF (llama.cpp)",
    "safetensors": "SafeTensors",
    "onnx": "ONNX",
    "openvino": "OpenVINO",
    "coreml": "Core ML (Apple)",
    "tflite": "TensorFlow Lite",
    # Specialized libraries
    "diffusers": "Diffusers (kuvagenerointi)",
    "peft": "PEFT/LoRA (adapterit)",
    "sentence-transformers": "Sentence Transformers",
    "timm": "PyTorch Image Models",
    "spacy": "spaCy",
    "flair": "Flair NLP",
    "fasttext": "FastText",
    "adapter-transformers": "Adapter Transformers",
    "espnet": "ESPnet (puhe)",
    "fairseq": "Fairseq",
    "speechbrain": "SpeechBrain",
    "paddlepaddle": "PaddlePaddle",
    "stanza": "Stanza NLP",
    "nemo": "NVIDIA NeMo",
    "setfit": "SetFit",
    "span-marker": "SpanMarker",
    "sklearn": "Scikit-learn",
    "keras": "Keras",
    # Quantization
    "bitsandbytes": "bitsandbytes (8-bit)",
    "gptq": "GPTQ",
    "awq": "AWQ",
    "exl2": "ExLlamaV2",
    # LLM inference
    "vllm": "vLLM",
    "text-generation-inference": "TGI (HF)",
    "mlx": "MLX (Apple)",
    "ctranslate2": "CTranslate2",
}

# Libraries that indicate GGUF compatibility
GGUF_COMPATIBLE_LIBRARIES = ["gguf", "llama.cpp"]

# Libraries that indicate transformer models
TRANSFORMER_LIBRARIES = ["transformers", "safetensors", "pytorch"]

# =============================================================================
# APPLICATIONS AND RUNNERS
# =============================================================================

APPS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "name": "Ollama",
        "description": "Paikallinen LLM-ajuri",
        "requires": ["gguf"],
    },
    "llama.cpp": {
        "name": "llama.cpp",
        "description": "Kevyt C++ inference",
        "requires": ["gguf"],
    },
    "lmstudio": {
        "name": "LM Studio",
        "description": "Tyopoyta-LLM",
        "requires": ["gguf"],
    },
    "jan": {
        "name": "Jan",
        "description": "Avoimen lahdekoodin ChatGPT",
        "requires": ["gguf"],
    },
    "gpt4all": {
        "name": "GPT4All",
        "description": "Paikallinen LLM",
        "requires": ["gguf"],
    },
    "localai": {
        "name": "LocalAI",
        "description": "OpenAI-yhteensopiva API",
        "requires": ["gguf"],
    },
    "koboldcpp": {
        "name": "KoboldCpp",
        "description": "Tarinankerrontaan",
        "requires": ["gguf"],
    },
    "text-generation-webui": {
        "name": "Text Generation WebUI",
        "description": "Oobaboogan UI",
        "requires": ["transformers", "gguf", "gptq", "awq"],
    },
    "vllm": {
        "name": "vLLM",
        "description": "Nopea tuotanto-inference",
        "requires": ["transformers", "safetensors", "awq", "gptq"],
    },
    "tgi": {
        "name": "Text Generation Inference",
        "description": "HuggingFace tuotantopalvelin",
        "requires": ["transformers", "safetensors"],
    },
    "mlx": {
        "name": "MLX",
        "description": "Apple Silicon inference",
        "requires": ["mlx"],
    },
    "exllamav2": {
        "name": "ExLlamaV2",
        "description": "GPU-optimoitu inference",
        "requires": ["exl2", "safetensors"],
    },
    "automatic1111": {
        "name": "Automatic1111",
        "description": "Stable Diffusion WebUI",
        "requires": ["diffusers", "safetensors"],
    },
    "comfyui": {
        "name": "ComfyUI",
        "description": "Node-pohjainen kuvagenerointi",
        "requires": ["diffusers", "safetensors"],
    },
}

# Simpler list for UI display
APP_CHOICES: List[Tuple[str, str]] = [
    ("ollama", "Ollama"),
    ("llama.cpp", "llama.cpp"),
    ("lmstudio", "LM Studio"),
    ("jan", "Jan"),
    ("vllm", "vLLM"),
    ("tgi", "Text Generation Inference"),
    ("text-generation-webui", "Text Generation WebUI"),
    ("automatic1111", "Automatic1111 (SD)"),
    ("comfyui", "ComfyUI"),
]

# =============================================================================
# MODEL SIZE RANGES (parameter count)
# =============================================================================

PARAMETER_RANGES: List[Tuple[str, Optional[int], Optional[int]]] = [
    ("< 1B", 0, 1_000_000_000),
    ("1B - 3B", 1_000_000_000, 3_000_000_000),
    ("3B - 7B", 3_000_000_000, 7_000_000_000),
    ("7B - 13B", 7_000_000_000, 13_000_000_000),
    ("13B - 30B", 13_000_000_000, 30_000_000_000),
    ("30B - 70B", 30_000_000_000, 70_000_000_000),
    ("70B+", 70_000_000_000, None),
]

# Size strings for display
SIZE_DISPLAY_MAP: Dict[str, str] = {
    "1b": "1B",
    "1.5b": "1.5B",
    "2b": "2B",
    "3b": "3B",
    "7b": "7B",
    "8b": "8B",
    "9b": "9B",
    "13b": "13B",
    "14b": "14B",
    "27b": "27B",
    "32b": "32B",
    "34b": "34B",
    "70b": "70B",
    "72b": "72B",
    "90b": "90B",
    "123b": "123B",
    "180b": "180B",
    "405b": "405B",
}

# =============================================================================
# LICENSES
# =============================================================================

LICENSES: Dict[str, Dict[str, Any]] = {
    # Open source
    "mit": {"name": "MIT", "open": True, "commercial": True},
    "apache-2.0": {"name": "Apache 2.0", "open": True, "commercial": True},
    "bsd-3-clause": {"name": "BSD 3-Clause", "open": True, "commercial": True},
    "gpl-3.0": {"name": "GPL 3.0", "open": True, "commercial": False},
    "lgpl-3.0": {"name": "LGPL 3.0", "open": True, "commercial": True},
    "cc-by-4.0": {"name": "CC BY 4.0", "open": True, "commercial": True},
    "cc-by-sa-4.0": {"name": "CC BY-SA 4.0", "open": True, "commercial": True},
    "cc-by-nc-4.0": {"name": "CC BY-NC 4.0", "open": True, "commercial": False},
    "cc-by-nc-sa-4.0": {"name": "CC BY-NC-SA 4.0", "open": True, "commercial": False},
    "cc0-1.0": {"name": "CC0 1.0", "open": True, "commercial": True},
    "openrail": {"name": "OpenRAIL", "open": True, "commercial": True},
    "openrail++": {"name": "OpenRAIL++", "open": True, "commercial": True},
    "bigscience-openrail-m": {"name": "BigScience OpenRAIL-M", "open": True, "commercial": True},
    "creativeml-openrail-m": {"name": "CreativeML OpenRAIL-M", "open": True, "commercial": True},
    "bigcode-openrail-m": {"name": "BigCode OpenRAIL-M", "open": True, "commercial": True},
    # Commercial/Restricted
    "llama2": {"name": "Llama 2 Community", "open": False, "commercial": True, "gated": True},
    "llama3": {"name": "Llama 3 Community", "open": False, "commercial": True, "gated": True},
    "llama3.1": {"name": "Llama 3.1 Community", "open": False, "commercial": True, "gated": True},
    "llama3.2": {"name": "Llama 3.2 Community", "open": False, "commercial": True, "gated": True},
    "llama3.3": {"name": "Llama 3.3 Community", "open": False, "commercial": True, "gated": True},
    "gemma": {"name": "Gemma Terms", "open": False, "commercial": True, "gated": True},
    "deepseek": {"name": "DeepSeek License", "open": False, "commercial": True},
    "qwen": {"name": "Qwen License", "open": False, "commercial": True},
    "mistral": {"name": "Mistral License", "open": False, "commercial": True},
    # Other
    "other": {"name": "Other", "open": None, "commercial": None},
    "unknown": {"name": "Unknown", "open": None, "commercial": None},
}

# License categories for filtering
LICENSE_CATEGORIES = {
    "permissive": ["mit", "apache-2.0", "bsd-3-clause", "cc0-1.0"],
    "copyleft": ["gpl-3.0", "lgpl-3.0", "cc-by-sa-4.0"],
    "non_commercial": ["cc-by-nc-4.0", "cc-by-nc-sa-4.0"],
    "model_specific": [
        "llama2",
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "gemma",
        "mistral",
        "qwen",
    ],
}

# =============================================================================
# QUANTIZATION QUALITY AND MEMORY ESTIMATES
# =============================================================================

# Quality scores based on perplexity tests (5.0 = original quality)
QUANTIZATION_QUALITY: Dict[str, float] = {
    "F32": 5.0,
    "F16": 5.0,
    "BF16": 5.0,
    "FP16": 5.0,
    "Q8_0": 4.95,
    "Q8_1": 4.95,
    "Q6_K": 4.85,
    "Q5_K_L": 4.7,
    "Q5_K_M": 4.6,
    "Q5_K_S": 4.5,
    "Q5_1": 4.5,
    "Q5_0": 4.4,
    "Q4_K_L": 4.3,
    "Q4_K_M": 4.2,  # Best quality/size ratio - recommended
    "Q4_K_S": 4.1,
    "Q4_1": 4.0,
    "Q4_0": 3.9,
    "Q3_K_L": 3.5,
    "Q3_K_M": 3.3,
    "Q3_K_S": 3.0,
    "Q2_K": 2.5,
    "Q2_K_S": 2.3,
    "IQ4_XS": 4.0,
    "IQ4_NL": 3.9,
    "IQ3_XXS": 2.8,
    "IQ3_XS": 3.0,
    "IQ3_S": 3.2,
    "IQ3_M": 3.4,
    "IQ2_XXS": 2.0,
    "IQ2_XS": 2.2,
    "IQ2_S": 2.3,
    "IQ2_M": 2.4,
    "IQ1_S": 1.5,
    "IQ1_M": 1.8,
}

# Bytes per parameter for different quantizations
BYTES_PER_PARAM: Dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "Q8_0": 1.1,
    "Q8_1": 1.2,
    "Q6_K": 0.85,
    "Q5_K_L": 0.72,
    "Q5_K_M": 0.68,
    "Q5_K_S": 0.64,
    "Q5_1": 0.7,
    "Q5_0": 0.65,
    "Q4_K_L": 0.6,
    "Q4_K_M": 0.55,
    "Q4_K_S": 0.5,
    "Q4_1": 0.56,
    "Q4_0": 0.5,
    "Q3_K_L": 0.45,
    "Q3_K_M": 0.42,
    "Q3_K_S": 0.38,
    "Q2_K": 0.35,
    "Q2_K_S": 0.32,
    "IQ4_XS": 0.48,
    "IQ4_NL": 0.5,
    "IQ3_XXS": 0.35,
    "IQ3_XS": 0.38,
    "IQ3_S": 0.4,
    "IQ3_M": 0.42,
    "IQ2_XXS": 0.28,
    "IQ2_XS": 0.3,
    "IQ2_S": 0.31,
    "IQ2_M": 0.32,
    "IQ1_S": 0.2,
    "IQ1_M": 0.22,
}

# Recommended quantizations (in order of recommendation)
RECOMMENDED_QUANTIZATIONS: List[str] = [
    "Q4_K_M",  # Best balance of quality and size
    "Q5_K_M",  # Higher quality
    "Q6_K",  # Near-original quality
    "Q3_K_M",  # Smaller, still usable
    "Q8_0",  # Highest quantized quality
]

# VRAM overhead factor (additional memory needed beyond model weights)
VRAM_OVERHEAD_FACTOR = 1.3  # 30% overhead for KV cache, etc.

# =============================================================================
# SEARCH PRESETS
# =============================================================================

SEARCH_PRESETS: Dict[str, Dict[str, any]] = {
    "llm_ollama": {
        "name": "LLM:t Ollamalle",
        "description": "Tekstingenerointi GGUF-muodossa Ollamalle",
        "filters": {
            "tasks": ["text-generation"],
            "libraries": ["gguf"],
        },
    },
    "llm_transformers": {
        "name": "LLM:t Transformersille",
        "description": "Tekstingenerointi SafeTensors-muodossa",
        "filters": {
            "tasks": ["text-generation"],
            "libraries": ["transformers", "safetensors"],
        },
    },
    "coding": {
        "name": "Koodausmallit",
        "description": "Ohjelmointiin optimoidut mallit",
        "filters": {
            "tasks": ["text-generation"],
            "query": "code coder coding",
        },
    },
    "vlm": {
        "name": "VLM (Vision-Language)",
        "description": "Kuvia ymartavat kielimallit",
        "filters": {
            "tasks": ["image-text-to-text", "visual-question-answering"],
        },
    },
    "translation": {
        "name": "Kaannosmallit",
        "description": "Monikieliset kaannosmallit",
        "filters": {
            "tasks": ["translation"],
        },
    },
    "tts": {
        "name": "Puhesynteesi (TTS)",
        "description": "Teksti puheeksi -mallit",
        "filters": {
            "tasks": ["text-to-speech"],
        },
    },
    "asr": {
        "name": "Puheentunnistus (ASR)",
        "description": "Puhe tekstiksi -mallit",
        "filters": {
            "tasks": ["automatic-speech-recognition"],
        },
    },
    "image_generation": {
        "name": "Kuvagenerointi",
        "description": "Teksti kuvaksi (Stable Diffusion, jne)",
        "filters": {
            "tasks": ["text-to-image"],
            "libraries": ["diffusers"],
        },
    },
    "embeddings": {
        "name": "Embedding-mallit",
        "description": "Tekstin vektorointi (RAG, haku)",
        "filters": {
            "tasks": ["feature-extraction", "sentence-similarity"],
            "libraries": ["sentence-transformers"],
        },
    },
    "lora": {
        "name": "LoRA-adapterit",
        "description": "Hienosaadetut PEFT-adapterit",
        "filters": {
            "libraries": ["peft"],
        },
    },
}

# =============================================================================
# INFERENCE PROVIDERS
# =============================================================================

INFERENCE_PROVIDERS: Dict[str, str] = {
    "hf-inference": "HuggingFace Inference API",
    "together": "Together AI",
    "fireworks": "Fireworks AI",
    "anyscale": "Anyscale",
    "deepinfra": "DeepInfra",
    "replicate": "Replicate",
    "groq": "Groq",
    "openrouter": "OpenRouter",
}

# =============================================================================
# SORT OPTIONS
# =============================================================================

SORT_OPTIONS: Dict[str, str] = {
    "downloads": "Lataukset (suosituin)",
    "likes": "Tykkäykset",
    "lastModified": "Viimeksi paivitetty",
    "created": "Lisayspaiva",
    "trending": "Trendaava",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_tasks_for_category(category: str) -> Dict[str, str]:
    """Get all tasks for a given category."""
    return TASK_CATEGORIES.get(category, {})


def get_all_task_choices() -> List[Tuple[str, str]]:
    """Get flat list of all tasks as (value, label) tuples."""
    choices = []
    for category, tasks in TASK_CATEGORIES.items():
        for task_id, task_name in tasks.items():
            choices.append((task_id, f"{task_name} [{category}]"))
    return choices


def get_quality_stars(quantization: str) -> str:
    """Get star rating for quantization quality (1-5 stars)."""
    quality = QUANTIZATION_QUALITY.get(quantization.upper(), 3.0)
    stars = round(quality)
    return "*" * stars + "." * (5 - stars)


def estimate_model_memory(param_count: int, quantization: str) -> int:
    """Estimate memory needed for a model in bytes."""
    bytes_per_param = BYTES_PER_PARAM.get(quantization.upper(), 0.5)
    base_size = param_count * bytes_per_param
    return int(base_size * VRAM_OVERHEAD_FACTOR)


def parse_model_size_from_name(name: str) -> Optional[str]:
    """Extract model size from name (e.g., 'Llama-2-7B' -> '7B')."""
    import re

    # Common patterns
    patterns = [
        r"(\d+\.?\d*)[Bb](?:illion)?",  # 7B, 7.5B, 7billion
        r"(\d+)[Mm](?:illion)?",  # 350M, 350million
        r"-(\d+\.?\d*)[Bb]-",  # -7B-
        r"_(\d+\.?\d*)[Bb]_",  # _7B_
    ]

    name_lower = name.lower()

    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            size = match.group(1)
            # Normalize
            if "b" in name_lower[match.start() : match.end()].lower():
                return f"{size}B"
            elif "m" in name_lower[match.start() : match.end()].lower():
                return f"{size}M"

    return None


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """Detect quantization type from GGUF filename."""
    import re

    filename_upper = filename.upper()

    # Check each known quantization
    for quant in QUANTIZATION_QUALITY.keys():
        # Create pattern that matches the quantization surrounded by non-alphanumeric
        pattern = rf"[^A-Z0-9]({re.escape(quant)})[^A-Z0-9]"
        if re.search(pattern, f"-{filename_upper}-"):
            return quant

    # Fallback patterns
    patterns = [
        (r"[._-]([QFBI][0-9]+(?:_[KXSML]+)?)[._-]", None),
        (r"[._-](F16|FP16|BF16|F32)[._-]", None),
    ]

    for pattern, default in patterns:
        match = re.search(pattern, f"-{filename_upper}-")
        if match:
            return match.group(1)

    return None


def get_app_compatibility(libraries: List[str], has_gguf: bool) -> List[str]:
    """Determine which apps are compatible based on libraries and file types."""
    compatible = []

    for app_id, app_info in APPS.items():
        required = app_info.get("requires", [])

        # Check if any required library/format is present
        if any(lib in libraries for lib in required):
            compatible.append(app_id)
        elif has_gguf and "gguf" in required:
            compatible.append(app_id)

    return compatible
