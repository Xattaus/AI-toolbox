"""
Poro + DeepSeek-R1 Merge Pipeline
=================================
Yhdistää suomenkielisen abliteroidun Poron ja päättelykykyisen DeepSeek-R1:n.

Vaiheet:
1. Tarkista/lataa DeepSeek-R1-Distill-Llama-8B-abliterated
2. Suorita SLERP merge
3. Konvertoi GGUF:ksi
4. Kvantisoi Q8_0
5. Luo Ollama-malli
6. Testaa
"""

import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Fix Windows encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ai_toolbox.core.paths import get_paths

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {cmd[:100]}..." if len(cmd) > 100 else f"  Command: {cmd}")
    print()

    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Command returned {result.returncode}")
        return False
    return True


def main():
    print("="*60)
    print("Poro + DeepSeek-R1 Merge Pipeline")
    print("="*60)

    # Paths
    paths = get_paths()
    poro_path = paths.abliterated_dir / "Poro-2-8B-abliterated-v2"
    deepseek_model = "huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated"
    output_name = "Poro-R1-Merged"
    output_path = paths.merged_dir / output_name
    gguf_dir = paths.gguf_dir

    # SLERP parameter (0.0 = 100% Poro, 1.0 = 100% DeepSeek)
    # 0.3 = 70% Poro, 30% DeepSeek (recommended)
    slerp_t = 0.3

    print(f"\nConfiguration:")
    print(f"  Poro model:    {poro_path.name}")
    print(f"  DeepSeek:      {deepseek_model}")
    print(f"  SLERP t:       {slerp_t} ({int((1-slerp_t)*100)}% Poro, {int(slerp_t*100)}% DeepSeek)")
    print(f"  Output:        {output_path}")

    # Check Poro exists
    if not poro_path.exists():
        print(f"\nERROR: Poro model not found: {poro_path}")
        print("Run abliteration first or check the path.")
        return

    # =========================================================================
    # STEP 1: Check mergekit installation
    # =========================================================================
    print("\n[STEP 1] Checking mergekit...")

    try:
        import mergekit  # noqa: F401
        print("  mergekit is installed")
    except ImportError:
        print("  mergekit not found. Installing...")
        run_command("pip install mergekit", "Installing mergekit")

    # =========================================================================
    # STEP 2: Create merge config
    # =========================================================================
    print("\n[STEP 2] Creating merge config...")

    config_content = f'''models:
  - model: {poro_path}
    parameters:
      weight: 1.0
  - model: {deepseek_model}
    parameters:
      weight: 1.0

merge_method: slerp
base_model: {poro_path}

parameters:
  t: {slerp_t}

dtype: bfloat16
'''

    config_path = paths.root / "configs" / "merges" / "_current_merge.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    print(f"  Config written to: {config_path}")

    # =========================================================================
    # STEP 3: Run merge
    # =========================================================================
    print("\n[STEP 3] Running SLERP merge...")
    print("  This will download DeepSeek-R1 if not cached (~16GB)")
    print("  Merge process may take 10-30 minutes...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    merge_cmd = f'mergekit-yaml "{config_path}" "{output_path}" --cuda --copy-tokenizer'
    if not run_command(merge_cmd, "Merging models"):
        # Try without CUDA if it fails
        print("  Retrying without --cuda...")
        merge_cmd = f'mergekit-yaml "{config_path}" "{output_path}" --copy-tokenizer'
        run_command(merge_cmd, "Merging models (CPU)")

    if not output_path.exists():
        print(f"\nERROR: Merge output not found: {output_path}")
        return

    print(f"\n  Merge complete: {output_path}")

    # =========================================================================
    # STEP 4: Convert to GGUF
    # =========================================================================
    print("\n[STEP 4] Converting to GGUF...")

    from ai_toolbox.conversion.converter import GGUFConverter

    converter = GGUFConverter()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = gguf_dir / f"{output_name}-f16.gguf"

    print(f"  Output: {gguf_path}")

    gguf_result = converter.convert_to_gguf(
        model_path=str(output_path),
        output_path=str(gguf_path),
        progress_callback=lambda msg, pct: print(f"  [{pct*100:.0f}%] {msg}"),
    )

    if not gguf_result.get("success"):
        print(f"  ERROR: GGUF conversion failed: {gguf_result.get('error')}")
        return

    actual_gguf = Path(gguf_result.get("output_path", gguf_path))
    print(f"  GGUF created: {actual_gguf}")

    # =========================================================================
    # STEP 5: Quantize to Q8_0
    # =========================================================================
    print("\n[STEP 5] Quantizing to Q8_0...")

    q8_path = gguf_dir / f"{output_name}-q8_0.gguf"

    quant_result = converter.quantize_gguf(
        input_path=str(actual_gguf),
        output_path=str(q8_path),
        quantization="q8_0",
        progress_callback=lambda msg, pct: print(f"  [{pct*100:.0f}%] {msg}"),
    )

    if not quant_result.get("success"):
        print(f"  WARNING: Quantization failed, using F16")
        q8_path = actual_gguf
    else:
        q8_path = Path(quant_result.get("output_path", q8_path))

    print(f"  Quantized: {q8_path}")

    # =========================================================================
    # STEP 6: Create Ollama model
    # =========================================================================
    print("\n[STEP 6] Creating Ollama model...")

    ollama_name = "poro-r1-merged:q8_0"

    # Create Modelfile
    modelfile_content = f'''FROM {q8_path}
SYSTEM Olet avulias suomenkielinen tekoalyassistentti jolla on vahva päättelykyky. Ajattele askel askeleelta ja vastaa suomeksi.
TEMPLATE """{{{{- if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{- end }}}}{{{{- range .Messages }}}}<|start_header_id|>{{{{ .Role }}}}<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>{{{{- end }}}}<|start_header_id|>assistant<|end_header_id|>

"""
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
'''

    modelfile_path = gguf_dir / "Modelfile_poro_r1"
    modelfile_path.write_text(modelfile_content)

    ollama_cmd = f'ollama create {ollama_name} -f "{modelfile_path}"'
    run_command(ollama_cmd, f"Creating Ollama model: {ollama_name}")

    # =========================================================================
    # STEP 7: Test
    # =========================================================================
    print("\n[STEP 7] Testing model...")

    test_prompts = [
        ("Moikka! Kuka sinä olet?", "Tervehdys"),
        ("Isäni ainoan lapsen nimi on Ville. Kuka minä olen?", "Logiikka 1"),
        ("Matti on Pekan isä. Pekka on Jonin isä. Miten Matti on sukua Jonille?", "Sukulaisuus"),
        ("Milla on 5 vuotta vanhempi kuin Liisa. Liisa on 8-vuotias. Kuinka vanha Milla on? Ajattele askel askeleelta.", "Matematiikka"),
    ]

    for prompt, name in test_prompts:
        print(f"\n  [{name}]")
        print(f"  Q: {prompt}")
        result = subprocess.run(
            f'ollama run {ollama_name} "{prompt}"',
            shell=True, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Clean up ANSI codes
            answer = result.stdout.strip()
            print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        else:
            print(f"  ERROR: {result.stderr}")

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "="*60)
    print("Pipeline complete!")
    print(f"  Merged model: {output_path}")
    print(f"  GGUF: {q8_path}")
    print(f"  Ollama: {ollama_name}")
    print(f"  Test: ollama run {ollama_name}")
    print("="*60)


if __name__ == "__main__":
    main()
