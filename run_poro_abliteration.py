"""
Poro-2-8B Abliteration Pipeline
===============================
Runs: Abliteration -> GGUF conversion -> Q8_0 quantization -> Ollama model -> Test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

def main():
    print("=" * 60)
    print("Poro-2-8B Abliteration Pipeline")
    print("=" * 60)
    print("1. Abliteration (Auto-tune + Reasoning Validation)")
    print("2. GGUF Conversion")
    print("3. Q8_0 Quantization")
    print("4. Ollama Model Creation")
    print("5. Test")
    print("=" * 60)

    # =========================================================================
    # STEP 1: ABLITERATION
    # =========================================================================
    print("\n[STEP 1] Abliteration")
    print("-" * 40)

    from ai_toolbox.abliteration.abliterator import Abliterator, AbliterationConfig
    from ai_toolbox.core.paths import get_paths

    paths = get_paths()
    model_path = paths.downloads_dir / "LumiOpen_Llama-Poro-2-8B-Instruct"
    output_name = "Poro-2-8B-abliterated-v2"

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    print(f"Source: {model_path.name}")
    print(f"Output: {output_name}")

    config = AbliterationConfig(
        model_path=str(model_path),
        output_name=output_name,
        strength=0.7,
        method="gradient",
        batch_size=4,  # Reduced for memory
        offload_mode="auto",  # Let transformers handle memory
        gradient_steps=30,  # Reduced from 50 to save memory
        use_smart_layers=True,
        layer_signal_threshold=0.5,
        use_dynamic_strength=True,
        use_linear_probe=True,
        probe_accuracy_threshold=0.85,
        use_auto_tune=True,
        auto_tune_max_iterations=4,
        auto_tune_test_prompts=4,
        use_capability_preservation=True,
        auto_scale_strength=True,
        use_reasoning_validation=True,
        reasoning_min_score=0.75,
        reasoning_strength_reduction=0.10,
        reasoning_min_strength=0.10,
        reasoning_max_retries=8,
    )

    abliterator = Abliterator()

    def progress_cb(msg: str, pct: float):
        print(f"  [{pct*100:.0f}%] {msg}")

    print("\nStarting abliteration...")
    result = abliterator.full_abliteration(config, progress_callback=progress_cb)

    if not result.success:
        print(f"ERROR: Abliteration failed: {result.error}")
        return

    print(f"\nAbliteration complete!")
    print(f"  Output: {result.output_path}")
    print(f"  Strength: {result.strength_applied:.2f}")
    if result.reasoning_score is not None:
        print(f"  Reasoning: {result.reasoning_score:.0%}")

    abliterated_path = Path(result.output_path)

    # =========================================================================
    # STEP 2: GGUF CONVERSION
    # =========================================================================
    print("\n[STEP 2] GGUF Conversion")
    print("-" * 40)

    from ai_toolbox.conversion.converter import GGUFConverter

    converter = GGUFConverter()
    gguf_output_dir = paths.gguf_dir
    gguf_output_dir.mkdir(parents=True, exist_ok=True)

    gguf_name = f"{output_name}-f16.gguf"
    gguf_path = gguf_output_dir / gguf_name

    print(f"Converting to: {gguf_path}")

    def gguf_progress_cb(msg: str, pct: float):
        print(f"  [{pct*100:.0f}%] {msg}")

    gguf_result = converter.convert_to_gguf(
        model_path=str(abliterated_path),
        output_path=str(gguf_path),
        progress_callback=gguf_progress_cb,
    )

    if not gguf_result.get("success"):
        print(f"ERROR: GGUF conversion failed: {gguf_result.get('error')}")
        return

    print("GGUF conversion complete!")
    actual_gguf_path = Path(gguf_result.get("output_path", gguf_path))

    # =========================================================================
    # STEP 3: Q8_0 QUANTIZATION
    # =========================================================================
    print("\n[STEP 3] Q8_0 Quantization")
    print("-" * 40)

    q8_name = f"{output_name}-q8_0.gguf"
    q8_path = gguf_output_dir / q8_name

    print(f"Quantizing to: {q8_path}")

    def quant_progress_cb(msg: str, pct: float):
        print(f"  [{pct*100:.0f}%] {msg}")

    quant_result = converter.quantize_gguf(
        input_path=str(actual_gguf_path),
        output_path=str(q8_path),
        quantization="q8_0",
        progress_callback=quant_progress_cb,
    )

    if not quant_result.get("success"):
        print(f"WARNING: Quantization issue: {quant_result.get('error')}")
        q8_path = actual_gguf_path
        print("Continuing with F16 model...")
    else:
        print("Quantization complete!")
        q8_path = Path(quant_result.get("output_path", q8_path))

    # =========================================================================
    # STEP 4: OLLAMA MODEL
    # =========================================================================
    print("\n[STEP 4] Create Ollama Model")
    print("-" * 40)

    from ai_toolbox.integrations.ollama import OllamaManager

    ollama = OllamaManager()

    ollama_name = "poro-2-8b-abliterated-v2:q8_0"

    print(f"Creating Ollama model: {ollama_name}")

    success, message = ollama.create_model(
        model_name=ollama_name,
        gguf_path=str(q8_path),
        system_prompt="Olet avulias suomenkielinen tekoalyassistentti. Vastaat kaikkiin kysymyksiin suoraan ja rehellisesti.",
    )

    if not success:
        print(f"ERROR: Ollama creation failed: {message}")
        print("You can create it manually:")
        print(f"  ollama create {ollama_name} -f Modelfile")
        return

    print(f"Ollama model created: {ollama_name}")

    # =========================================================================
    # STEP 5: TEST
    # =========================================================================
    print("\n[STEP 5] Test Model")
    print("-" * 40)

    test_prompts = [
        "Moikka! Kuka sina olet?",
        "Isani ainoan lapsen nimi on Ville. Kuka mina olen?",
        "Matti on Pekan isa. Pekka on Jonin isa. Miten Matti on sukua Jonille?",
    ]

    for prompt in test_prompts:
        print(f"\nQ: {prompt}")
        try:
            response = ollama.run_model(ollama_name, prompt, stream=False)
            print(f"A: {response}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Ollama model: {ollama_name}")
    print(f"Test with: ollama run {ollama_name}")


if __name__ == "__main__":
    main()
