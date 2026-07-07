"""
AI TOOLBOX - GGUF Converter
===========================

Converts HuggingFace models to GGUF format with quantization support.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import psutil
from ..core.ui import console
from rich.panel import Panel

from ..core.paths import get_gguf_dir, get_llama_cpp_dir
from .quantization import QuantizationType, QUANTIZATION_INFO
from .llama_cpp import LlamaCppManager



class GGUFConverter:
    """Converts HuggingFace models to GGUF format."""

    def __init__(
        self,
        llama_cpp_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the converter.

        Args:
            llama_cpp_path: Path to llama.cpp repository
            output_dir: Default output directory for converted models
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = get_gguf_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if llama_cpp_path:
            self.llama_cpp_path = Path(llama_cpp_path)
        else:
            self.llama_cpp_path = get_llama_cpp_dir()

        self._llama_cpp = LlamaCppManager(str(self.llama_cpp_path))
        self._convert_script: Optional[Path] = None

    def check_llama_cpp(self) -> Dict[str, Any]:
        """Check if llama.cpp is available and return status."""
        return self._llama_cpp.check_status()

    def setup_llama_cpp(self, force: bool = False) -> bool:
        """Set up llama.cpp by cloning the repository."""
        return self._llama_cpp.setup_from_git(force)

    def download_llama_cpp_binaries(self, cuda: bool = True) -> bool:
        """Download pre-built llama.cpp binaries from GitHub releases."""
        return self._llama_cpp.download_binaries(cuda)

    def _ensure_llama_cpp(self) -> None:
        """Ensure llama.cpp is available and set up."""
        if not self.llama_cpp_path.exists():
            if not self._llama_cpp.setup_from_git():
                raise RuntimeError("Failed to set up llama.cpp")

        self._convert_script = self._llama_cpp.find_convert_script()
        if not self._convert_script:
            raise RuntimeError(
                f"Could not find convert script in llama.cpp at {self.llama_cpp_path}. "
                "Please ensure llama.cpp is properly cloned."
            )

    @staticmethod
    def _is_valid_gguf(path: Path) -> bool:
        """Check that a file starts with the GGUF magic bytes."""
        try:
            with open(path, 'rb') as f:
                return f.read(4) == b'GGUF'
        except OSError:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for conversion recommendations."""
        memory = psutil.virtual_memory()

        return {
            "total_ram_gb": round(memory.total / (1024**3), 2),
            "available_ram_gb": round(memory.available / (1024**3), 2),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
        }

    def estimate_model_size(self, model_path: Path, quantization: str = "q4_k_m") -> Dict[str, float]:
        """Estimate the output model size for a given quantization."""
        model_path = Path(model_path)
        config_path = model_path / "config.json"

        if not config_path.exists():
            return {"estimated_size_gb": 0, "original_size_gb": 0, "error": "config.json not found"}

        with open(config_path, "r") as f:
            config = json.load(f)

        # Calculate parameter count
        hidden_size = config.get("hidden_size", config.get("d_model", 4096))
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
        vocab_size = config.get("vocab_size", 32000)
        intermediate_size = config.get("intermediate_size", hidden_size * 4)

        params_per_layer = (
            4 * hidden_size * hidden_size +
            2 * hidden_size * intermediate_size
        )
        embedding_params = vocab_size * hidden_size * 2
        total_params = num_layers * params_per_layer + embedding_params

        original_size_bytes = total_params * 2
        original_size_gb = original_size_bytes / (1024**3)

        # Get bits per weight for quantization
        try:
            quant_type = QuantizationType(quantization.lower())
            quant_info = QUANTIZATION_INFO.get(quant_type)
            bits_per_weight = quant_info.bits_per_weight if quant_info else 4.0
        except ValueError:
            bits_per_weight = 4.0

        quantized_size_bytes = total_params * (bits_per_weight / 8)
        quantized_size_gb = quantized_size_bytes / (1024**3)

        return {
            "estimated_size_gb": round(quantized_size_gb, 2),
            "original_size_gb": round(original_size_gb, 2),
            "total_params_billions": round(total_params / 1e9, 2),
            "compression_ratio": round(original_size_gb / quantized_size_gb, 2) if quantized_size_gb > 0 else 0,
        }

    def convert_to_gguf(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        output_type: str = "f16",
        vocab_type: Optional[str] = None,
        ctx_size: Optional[int] = None,
        pad_vocab: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Convert a HuggingFace model to GGUF format.

        Args:
            model_path: Path to the HuggingFace model directory
            output_path: Path for the output GGUF file
            output_type: Output type (f32, f16, bf16, q8_0, auto)
            vocab_type: Vocabulary type override
            ctx_size: Context size override
            pad_vocab: Pad vocab to multiple of 64
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with conversion result
        """
        try:
            self._ensure_llama_cpp()
        except Exception as e:
            return {"success": False, "error": str(e)}

        model_path = Path(model_path)
        if not model_path.exists():
            return {"success": False, "error": f"Model path does not exist: {model_path}"}

        # Determine output path
        if output_path is None:
            model_name = model_path.name
            out_path = self.output_dir / f"{model_name}-{output_type}.gguf"
        else:
            out_path = Path(output_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove any stale output up front so a leftover file from a previous
        # run can never be mistaken for a successful conversion.
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError as e:
                return {"success": False, "error": f"Could not remove stale output {out_path}: {e}"}

        console.print(Panel(
            f"[bold cyan]Converting model to GGUF[/bold cyan]\n\n"
            f"[white]Source:[/white] {model_path}\n"
            f"[white]Output:[/white] {out_path}\n"
            f"[white]Type:[/white] {output_type}",
            title="GGUF Conversion",
            border_style="cyan"
        ))

        # Build the conversion command
        cmd = [
            sys.executable,
            str(self._convert_script),
            str(model_path),
            "--outfile", str(out_path),
            "--outtype", output_type,
        ]

        if vocab_type:
            cmd.extend(["--vocab-type", vocab_type])

        if ctx_size:
            cmd.extend(["--ctx", str(ctx_size)])

        if pad_vocab:
            cmd.append("--pad-vocab")

        # Run conversion
        console.print("[cyan]Running conversion...[/cyan]")

        output_lines = []
        return_code = None

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
            )

            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                stripped = line.strip()
                if stripped:
                    if len(stripped) > 80:
                        stripped = stripped[:77] + "..."
                    try:
                        safe_line = stripped.replace("[", "\\[")
                        safe_line = safe_line.encode('ascii', 'replace').decode('ascii')
                        console.print(f"  [dim]{safe_line}[/dim]")
                    except Exception:
                        pass
                if progress_callback:
                    try:
                        progress_callback(stripped, -1)
                    except Exception:
                        pass

            process.wait()
            return_code = process.returncode

        except Exception as e:
            run_error = str(e)
        else:
            run_error = None

        # Any non-zero exit code - OR a failure to even launch the subprocess,
        # which leaves return_code as None - is a failure. The output file may
        # be missing or truncated, so never report it as success.
        if return_code != 0:
            error_output = '\n'.join(output_lines[-20:])
            if run_error:
                error_output += f"\nProcess error: {run_error}"
            if out_path.exists():
                error_output += f"\n\nHuom: keskeneräinen tiedosto voi olla levyllä: {out_path}"
            return {"success": False, "error": f"Conversion failed (code {return_code}):\n{error_output}"}

        if out_path.exists() and out_path.stat().st_size > 0:
            if not self._is_valid_gguf(out_path):
                return {
                    "success": False,
                    "error": f"Output file is not a valid GGUF (bad magic bytes): {out_path}",
                }

            file_size_gb = out_path.stat().st_size / (1024**3)
            try:
                console.print("[green]Conversion complete![/green]")
                console.print(f"[white]Output file:[/white] {out_path}")
                console.print(f"[white]File size:[/white] {file_size_gb:.2f} GB")
            except Exception:
                print(f"Conversion complete! Output: {out_path} ({file_size_gb:.2f} GB)")

            return {
                "success": True,
                "output_path": str(out_path),
                "file_size_gb": round(file_size_gb, 2),
            }

        error = f"Conversion completed but output file not found: {out_path}"
        if run_error:
            error += f"\nProcess error: {run_error}"
        return {"success": False, "error": error}

    def quantize_gguf(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        quantization: str = "q4_k_m",
        threads: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Quantize a GGUF model.

        Args:
            input_path: Path to the input GGUF file
            output_path: Path for the output quantized GGUF file
            quantization: Quantization type (string)
            threads: Number of threads to use
            progress_callback: Callback for progress updates

        Returns:
            Dictionary with quantization result
        """
        quantize_binary = self._llama_cpp.find_quantize_binary()

        if not quantize_binary:
            return {
                "success": False,
                "error": "llama-quantize binary not found. Please build llama.cpp first.",
                "hint": "Run: cd ~/.ai-toolbox/llama.cpp && make llama-quantize"
            }

        input_path = Path(input_path)
        if not input_path.exists():
            return {"success": False, "error": f"Input GGUF file not found: {input_path}"}

        # Determine output path
        if output_path is None:
            stem = input_path.stem.replace("-f16", "").replace("-f32", "")
            out_path = input_path.parent / f"{stem}-{quantization}.gguf"
        else:
            out_path = Path(output_path)

        # Get quantization info
        try:
            quant_type = QuantizationType(quantization.lower())
            quant_info = QUANTIZATION_INFO.get(quant_type)
            description = quant_info.description if quant_info else "N/A"
        except ValueError:
            description = "Custom quantization"

        console.print(Panel(
            f"[bold cyan]Quantizing GGUF model[/bold cyan]\n\n"
            f"[white]Input:[/white] {input_path}\n"
            f"[white]Output:[/white] {out_path}\n"
            f"[white]Quantization:[/white] {quantization}\n"
            f"[white]Description:[/white] {description}",
            title="GGUF Quantization",
            border_style="magenta"
        ))

        # Build quantization command
        # llama-quantize usage: llama-quantize input.gguf output.gguf TYPE [nthreads]
        # (nthreads is a positional argument, not a flag)
        cmd = [str(quantize_binary), str(input_path), str(out_path), quantization.upper()]

        if threads:
            cmd.append(str(threads))

        # Run quantization
        console.print("[cyan]Running quantization...[/cyan]")

        output_lines = []
        return_code = None

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
            )

            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                stripped = line.strip()
                if stripped:
                    try:
                        if len(stripped) > 70:
                            stripped = stripped[:67] + "..."
                        safe_line = stripped.encode('ascii', 'replace').decode('ascii')
                        console.print(f"  [dim]{safe_line}[/dim]")
                    except Exception:
                        pass
                if progress_callback:
                    try:
                        progress_callback(stripped, -1)
                    except Exception:
                        pass

            process.wait()
            return_code = process.returncode

        except Exception as e:
            run_error = str(e)
        else:
            run_error = None

        # Non-zero exit code = quantizer aborted; the output may be truncated
        if return_code is not None and return_code != 0:
            error_output = '\n'.join(output_lines[-20:])
            if out_path.exists():
                error_output += f"\n\nHuom: keskeneräinen tiedosto voi olla levyllä: {out_path}"
            return {"success": False, "error": f"Quantization failed (code {return_code}):\n{error_output}"}

        if out_path.exists() and out_path.stat().st_size > 0:
            if not self._is_valid_gguf(out_path):
                return {
                    "success": False,
                    "error": f"Output file is not a valid GGUF (bad magic bytes): {out_path}",
                }

            file_size_gb = out_path.stat().st_size / (1024**3)
            try:
                console.print("[green]Quantization complete![/green]")
                console.print(f"[white]Output file:[/white] {out_path}")
                console.print(f"[white]File size:[/white] {file_size_gb:.2f} GB")
            except Exception:
                print(f"Quantization complete! Output: {out_path} ({file_size_gb:.2f} GB)")

            return {
                "success": True,
                "output_path": str(out_path),
                "file_size_gb": round(file_size_gb, 2),
                "quantization": quantization,
            }

        error = f"Quantization completed but output file not found: {out_path}"
        if run_error:
            error += f"\nProcess error: {run_error}"
        return {"success": False, "error": error}

    def convert_and_quantize(
        self,
        model_path: str,
        quantization: str = "q4_k_m",
        output_path: Optional[str] = None,
        keep_f16: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert HuggingFace model to GGUF and quantize in one step.

        Args:
            model_path: Path to HuggingFace model
            quantization: Target quantization type
            output_path: Path for final output
            keep_f16: Keep intermediate F16 file

        Returns:
            Dictionary with result
        """
        model_path = Path(model_path)
        model_name = model_path.name

        # Step 1: Convert to F16 GGUF
        console.print("\n[bold cyan]Step 1/2: Converting to GGUF (F16)...[/bold cyan]\n")

        f16_result = self.convert_to_gguf(
            model_path=str(model_path),
            output_type="f16",
        )

        if not f16_result.get("success"):
            return f16_result

        f16_path = f16_result["output_path"]

        # Step 2: Quantize
        console.print(f"\n[bold cyan]Step 2/2: Quantizing to {quantization}...[/bold cyan]\n")

        if output_path:
            quant_output = output_path
        else:
            quant_output = str(self.output_dir / f"{model_name}-{quantization}.gguf")

        quant_result = self.quantize_gguf(
            input_path=f16_path,
            output_path=quant_output,
            quantization=quantization,
        )

        # Clean up F16 file if not keeping
        if not keep_f16 and quant_result.get("success"):
            try:
                Path(f16_path).unlink()
                console.print("[dim]Cleaned up intermediate F16 file[/dim]")
            except Exception:
                pass

        if quant_result.get("success"):
            quant_result["model_name"] = model_name
            quant_result["source_path"] = str(model_path)

        return quant_result

    @staticmethod
    def list_quantization_types() -> list:
        """Get list of available quantization types."""
        result = []
        for quant_type, info in QUANTIZATION_INFO.items():
            result.append({
                "type": quant_type.value,
                "bits_per_weight": info.bits_per_weight,
                "quality": info.quality,
                "description": info.description,
            })
        return result

    def recommend_quantization(
        self,
        model_params_billions: float,
        available_ram_gb: Optional[float] = None
    ) -> list:
        """Recommend quantization types based on model size and available RAM."""
        if available_ram_gb is None:
            available_ram_gb = self.get_system_info()["available_ram_gb"]

        recommendations = []

        for quant_type, info in QUANTIZATION_INFO.items():
            estimated_size_gb = (model_params_billions * 1e9 * info.bits_per_weight / 8) / (1024**3)
            required_ram = estimated_size_gb + 2

            if required_ram <= available_ram_gb * 0.8:
                recommendations.append({
                    "type": quant_type.value,
                    "bits_per_weight": info.bits_per_weight,
                    "quality": info.quality,
                    "estimated_size_gb": round(estimated_size_gb, 2),
                    "required_ram_gb": round(required_ram, 2),
                    "fits_in_ram": True,
                })

        recommendations.sort(key=lambda x: x["bits_per_weight"], reverse=True)

        return recommendations[:5]
