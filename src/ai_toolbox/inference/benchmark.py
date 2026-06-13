"""
AI TOOLBOX - Benchmark Runner
=============================

Mallien nopeuden ja laadun vertailutyökalu.
Mittaa tokens/second, latenssia ja muistinkäyttöä.
"""

import json
import csv
import time
import gc
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..core.paths import get_paths

console = Console()


# =============================================================================
# DEFAULT BENCHMARK PROMPTS
# =============================================================================

DEFAULT_PROMPTS = {
    "short": "What is 2+2?",
    "medium": "Explain the concept of machine learning in 3 sentences.",
    "long": "Write a detailed explanation of how neural networks work, including the concepts of layers, weights, biases, activation functions, and backpropagation. Be thorough but concise.",
    "code": "Write a Python function that calculates the nth Fibonacci number using recursion with memoization.",
    "reasoning": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
    "creative": "Write a short poem (4-6 lines) about artificial intelligence and its impact on humanity.",
    "finnish": "Selitä lyhyesti mitä koneoppiminen tarkoittaa ja anna yksi käytännön esimerkki.",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark-konfiguraatio."""
    prompt: str = ""
    prompt_name: str = "custom"
    max_tokens: int = 128
    temperature: float = 0.7
    num_runs: int = 3
    warmup_runs: int = 1
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Yksittäisen benchmarkin tulos."""
    id: str = ""
    model_name: str = ""
    model_path: str = ""
    quantization: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_used_mb: float = 0.0
    timestamp: str = ""
    prompt_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    response_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Muunna dict-muotoon."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Luo dict:stä."""
        return cls(**data)


@dataclass
class ComparisonReport:
    """Mallien vertailuraportti."""
    results: List[BenchmarkResult] = field(default_factory=list)
    fastest_model: str = ""
    slowest_model: str = ""
    rankings: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Muunna dict-muotoon."""
        return {
            "results": [r.to_dict() for r in self.results],
            "fastest_model": self.fastest_model,
            "slowest_model": self.slowest_model,
            "rankings": self.rankings,
            "timestamp": self.timestamp,
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Benchmark-työkalut mallien vertailuun."""

    def __init__(self):
        """Alusta benchmark runner."""
        paths = get_paths()
        self.benchmarks_dir = paths.benchmarks_dir
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.benchmarks_dir / "results.json"
        self.exports_dir = self.benchmarks_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        # llama_cpp backend (lazy load)
        self._llm = None
        self._model_path = None
        self._llama_available = None

    def _check_llama_cpp(self) -> bool:
        """Tarkista onko llama-cpp-python asennettu."""
        if self._llama_available is None:
            try:
                from llama_cpp import Llama
                self._llama_available = True
            except ImportError:
                self._llama_available = False
        return self._llama_available

    def get_status(self) -> Dict[str, Any]:
        """Palauta benchmark runnerin status."""
        results = self.load_results()
        return {
            "llama_cpp_available": self._check_llama_cpp(),
            "results_count": len(results),
            "benchmarks_dir": str(self.benchmarks_dir),
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Palauta järjestelmätiedot."""
        import multiprocessing

        mem = psutil.virtual_memory()

        info = {
            "cpu_count": multiprocessing.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False) or multiprocessing.cpu_count(),
            "total_ram_gb": round(mem.total / (1024**3), 1),
            "available_ram_gb": round(mem.available / (1024**3), 1),
            "used_ram_gb": round(mem.used / (1024**3), 1),
            "ram_percent": mem.percent,
        }

        # Yritä tunnistaa GPU (CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            else:
                info["gpu_available"] = False
        except ImportError:
            info["gpu_available"] = False

        return info

    def get_default_prompts(self) -> Dict[str, str]:
        """Palauta oletuspromptikokoelma."""
        return DEFAULT_PROMPTS.copy()

    # ==================== MODEL LOADING ====================

    def _load_model(self, model_path: str, config: BenchmarkConfig) -> bool:
        """Lataa malli benchmarkia varten (hiljainen lataus)."""
        if not self._check_llama_cpp():
            return False

        try:
            from llama_cpp import Llama
            import multiprocessing

            # Vapauta edellinen malli
            self._unload_model()

            n_threads = config.n_threads
            if n_threads is None:
                n_threads = max(1, multiprocessing.cpu_count() - 1)

            self._llm = Llama(
                model_path=model_path,
                n_ctx=config.n_ctx,
                n_threads=n_threads,
                n_gpu_layers=config.n_gpu_layers,
                seed=42,  # Kiinteä siemen - benchmark-ajot toistettavia
                verbose=False,
            )

            self._model_path = model_path
            return True

        except Exception as e:
            console.print(f"[red]Mallin lataus epäonnistui: {e}[/red]")
            return False

    def _unload_model(self):
        """Vapauta malli muistista."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._model_path = None
            gc.collect()

    def _detect_quantization(self, model_path: str) -> Optional[str]:
        """Tunnista kvantisointityyppi tiedostonimestä."""
        name = Path(model_path).name.lower()

        quant_types = [
            "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0", "q5_1",
            "q4_k_m", "q4_k_s", "q4_0", "q4_1",
            "q3_k_m", "q3_k_s", "q3_k_l",
            "q2_k", "iq4_xs", "iq3_m", "iq2_xs", "iq1_s",
            "f32", "f16", "bf16",
        ]

        for qt in quant_types:
            if qt in name or qt.upper() in name:
                return qt.upper()

        return None

    # ==================== BENCHMARKING ====================

    def run_benchmark(
        self,
        model_path: str,
        config: BenchmarkConfig,
        progress_callback: Optional[Any] = None,
    ) -> Optional[BenchmarkResult]:
        """
        Suorita benchmark yhdelle mallille.

        Args:
            model_path: Polku GGUF-malliin
            config: Benchmark-konfiguraatio
            progress_callback: Kutsutaan edistymän päivityksiin

        Returns:
            BenchmarkResult tai None jos epäonnistui
        """
        if not self._check_llama_cpp():
            console.print("[red]llama-cpp-python ei ole asennettu![/red]")
            return None

        model_name = Path(model_path).name
        quantization = self._detect_quantization(model_path)

        # Mittaa muisti ennen latausta
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        if progress_callback:
            progress_callback(f"Ladataan mallia: {model_name}...")

        # Lataa malli
        if not self._load_model(model_path, config):
            return None

        mem_after_load = process.memory_info().rss / (1024 * 1024)

        if progress_callback:
            progress_callback("Lämmittelyajot...")

        # Warmup-ajot
        for _ in range(config.warmup_runs):
            try:
                self._llm(
                    config.prompt,
                    max_tokens=min(32, config.max_tokens),
                    temperature=config.temperature,
                    echo=False,
                )
            except Exception:
                pass

        if progress_callback:
            progress_callback(f"Benchmark-ajot ({config.num_runs} kpl)...")

        # Benchmark-ajot
        run_times = []
        ttft_times = []
        completion_tokens_list = []
        prompt_tokens = 0
        last_response = ""

        for run_idx in range(config.num_runs):
            try:
                # Ajanotto
                start_time = time.perf_counter()

                response = self._llm(
                    config.prompt,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    echo=False,
                )

                end_time = time.perf_counter()
                run_time_ms = (end_time - start_time) * 1000
                run_times.append(run_time_ms)

                # Hae token-tiedot
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                comp_tokens = usage.get("completion_tokens", 0)
                completion_tokens_list.append(comp_tokens)

                # Vastaus
                last_response = response.get("choices", [{}])[0].get("text", "").strip()

                # TTFT arvio - laske jokaiselle ajolle
                # HUOM: Todellinen TTFT vaatisi streaming-API:a. Tämä on arvio
                # perustuen prompt-prosessointiin (prompt_eval_time)
                if comp_tokens > 0 and prompt_tokens > 0:
                    # Arvio: prompt-prosessointi vie suurimman osan TTFT:stä
                    # Käytämme kaavaa: total_time * (prompt_tokens / (prompt_tokens + comp_tokens))
                    total_tokens = prompt_tokens + comp_tokens
                    ttft_estimate = run_time_ms * (prompt_tokens / total_tokens)
                    ttft_times.append(ttft_estimate)

            except Exception as e:
                console.print(f"[red]Ajo {run_idx + 1} epäonnistui: {e}[/red]")

        # Vapauta malli
        self._unload_model()

        if not run_times:
            return None

        # Laske keskiarvot
        avg_time_ms = sum(run_times) / len(run_times)
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list) if completion_tokens_list else 0

        # Tokens/second
        tokens_per_second = 0
        if avg_time_ms > 0 and avg_completion_tokens > 0:
            tokens_per_second = (avg_completion_tokens / avg_time_ms) * 1000

        # TTFT
        avg_ttft = sum(ttft_times) / len(ttft_times) if ttft_times else 0

        # Luo tulos
        result = BenchmarkResult(
            id=str(uuid.uuid4())[:8],
            model_name=model_name,
            model_path=model_path,
            quantization=quantization,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(avg_completion_tokens),
            total_time_ms=round(avg_time_ms, 2),
            tokens_per_second=round(tokens_per_second, 2),
            time_to_first_token_ms=round(avg_ttft, 2),
            memory_before_mb=round(mem_before, 1),
            memory_after_mb=round(mem_after_load, 1),
            memory_used_mb=round(mem_after_load - mem_before, 1),
            timestamp=datetime.now().isoformat(),
            prompt_name=config.prompt_name,
            config={
                "prompt": config.prompt[:100] + "..." if len(config.prompt) > 100 else config.prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "num_runs": config.num_runs,
                "n_ctx": config.n_ctx,
                "n_gpu_layers": config.n_gpu_layers,
            },
            response_preview=last_response[:200] + "..." if len(last_response) > 200 else last_response,
        )

        return result

    def run_throughput_test(
        self,
        model_path: str,
        prompt_sizes: List[str] = None,
        max_tokens: int = 128,
        progress_callback: Optional[Any] = None,
    ) -> List[BenchmarkResult]:
        """
        Suorita throughput-testi eri prompt-pituuksilla.

        Args:
            model_path: Polku malliin
            prompt_sizes: Lista prompttien nimistä (short, medium, long)
            max_tokens: Generoitavien tokenien määrä

        Returns:
            Lista BenchmarkResult-tuloksia
        """
        if prompt_sizes is None:
            prompt_sizes = ["short", "medium", "long"]

        results = []

        for prompt_name in prompt_sizes:
            if prompt_name not in DEFAULT_PROMPTS:
                continue

            if progress_callback:
                progress_callback(f"Testataan: {prompt_name}...")

            config = BenchmarkConfig(
                prompt=DEFAULT_PROMPTS[prompt_name],
                prompt_name=prompt_name,
                max_tokens=max_tokens,
                num_runs=3,
                warmup_runs=1,
            )

            result = self.run_benchmark(model_path, config, progress_callback)
            if result:
                results.append(result)

        return results

    def measure_memory_usage(
        self,
        model_path: str,
        config: Optional[BenchmarkConfig] = None,
    ) -> Dict[str, Any]:
        """
        Mittaa mallin muistinkäyttö.

        Returns:
            Dict muistitiedoilla
        """
        if config is None:
            config = BenchmarkConfig()

        process = psutil.Process()

        # Muisti ennen
        gc.collect()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Lataa malli
        if not self._load_model(model_path, config):
            return {"error": "Mallin lataus epäonnistui"}

        # Muisti latauksen jälkeen
        mem_after_load = process.memory_info().rss / (1024 * 1024)

        # Aja yksi inferenssi
        try:
            self._llm("Test", max_tokens=32, echo=False)
        except Exception:
            pass

        mem_after_inference = process.memory_info().rss / (1024 * 1024)

        # Vapauta
        self._unload_model()
        gc.collect()

        mem_after_unload = process.memory_info().rss / (1024 * 1024)

        return {
            "model_name": Path(model_path).name,
            "memory_before_mb": round(mem_before, 1),
            "memory_after_load_mb": round(mem_after_load, 1),
            "memory_after_inference_mb": round(mem_after_inference, 1),
            "memory_after_unload_mb": round(mem_after_unload, 1),
            "model_memory_mb": round(mem_after_load - mem_before, 1),
            "inference_overhead_mb": round(mem_after_inference - mem_after_load, 1),
        }

    # ==================== COMPARISON ====================

    def compare_models(
        self,
        model_paths: List[str],
        config: BenchmarkConfig,
        progress_callback: Optional[Any] = None,
    ) -> ComparisonReport:
        """
        Vertaile useita malleja keskenään.

        Args:
            model_paths: Lista mallipolkuihin
            config: Yhteinen benchmark-konfiguraatio

        Returns:
            ComparisonReport
        """
        results = []

        for i, model_path in enumerate(model_paths):
            if progress_callback:
                progress_callback(f"Testataan {i+1}/{len(model_paths)}: {Path(model_path).name}")

            result = self.run_benchmark(model_path, config, progress_callback)
            if result:
                results.append(result)

        if not results:
            return ComparisonReport(timestamp=datetime.now().isoformat())

        # Järjestä nopeuden mukaan
        sorted_results = sorted(results, key=lambda r: r.tokens_per_second, reverse=True)

        # Luo rankings
        rankings = [(r.model_name, r.tokens_per_second) for r in sorted_results]

        return ComparisonReport(
            results=sorted_results,
            fastest_model=sorted_results[0].model_name if sorted_results else "",
            slowest_model=sorted_results[-1].model_name if sorted_results else "",
            rankings=rankings,
            timestamp=datetime.now().isoformat(),
        )

    # ==================== RESULTS MANAGEMENT ====================

    def save_result(self, result: BenchmarkResult) -> Path:
        """Tallenna tulos atomaarisesti (temp-tiedosto + rename)."""
        results = self.load_results()
        results.append(result)

        # Tallenna JSON atomaarisesti korruption estämiseksi
        data = {
            "version": "1.0",
            "results": [r.to_dict() for r in results],
        }

        # Kirjoita ensin temp-tiedostoon
        temp_file = self.results_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # Korvaa alkuperäinen vasta kun kirjoitus onnistui
            temp_file.replace(self.results_file)
        except Exception as e:
            # Siivoa temp-tiedosto virhetilanteessa
            if temp_file.exists():
                temp_file.unlink()
            raise e

        return self.results_file

    def load_results(self) -> List[BenchmarkResult]:
        """Lataa aiemmat tulokset."""
        if not self.results_file.exists():
            return []

        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = []
            for r in data.get("results", []):
                try:
                    results.append(BenchmarkResult.from_dict(r))
                except Exception:
                    pass

            return results
        except Exception:
            return []

    def clear_results(self) -> bool:
        """Tyhjennä kaikki tulokset."""
        try:
            if self.results_file.exists():
                self.results_file.unlink()
            return True
        except Exception:
            return False

    def export_csv(self, results: List[BenchmarkResult], output_path: Path) -> bool:
        """Vie tulokset CSV-muodossa."""
        if not results:
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "timestamp", "model_name", "quantization",
                "tokens_per_second", "total_time_ms", "time_to_first_token_ms",
                "prompt_tokens", "completion_tokens", "memory_used_mb",
                "prompt_name",
            ]

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for r in results:
                    writer.writerow({
                        "timestamp": r.timestamp,
                        "model_name": r.model_name,
                        "quantization": r.quantization or "",
                        "tokens_per_second": r.tokens_per_second,
                        "total_time_ms": r.total_time_ms,
                        "time_to_first_token_ms": r.time_to_first_token_ms,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "memory_used_mb": r.memory_used_mb,
                        "prompt_name": r.prompt_name,
                    })

            return True
        except Exception:
            return False

    def export_json(self, results: List[BenchmarkResult], output_path: Path) -> bool:
        """Vie tulokset JSON-muodossa."""
        if not results:
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "exported": datetime.now().isoformat(),
                "count": len(results),
                "results": [r.to_dict() for r in results],
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return True
        except Exception:
            return False

    # ==================== DISPLAY HELPERS ====================

    def format_result_table(self, result: BenchmarkResult) -> Table:
        """Luo Rich-taulukko yksittäiselle tulokselle."""
        table = Table(title=f"Benchmark: {result.model_name}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Tokens/second", f"{result.tokens_per_second:.1f} t/s")
        table.add_row("Time to first token", f"{result.time_to_first_token_ms:.1f} ms")
        table.add_row("Total time", f"{result.total_time_ms:.1f} ms")
        table.add_row("Prompt tokens", str(result.prompt_tokens))
        table.add_row("Completion tokens", str(result.completion_tokens))
        table.add_row("Memory used", f"{result.memory_used_mb:.1f} MB")
        table.add_row("Quantization", result.quantization or "N/A")
        table.add_row("Prompt", result.prompt_name)

        return table

    def format_comparison_table(self, report: ComparisonReport) -> Table:
        """Luo Rich-taulukko vertailuraportille."""
        table = Table(title="Model Comparison", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Tokens/s", style="green", justify="right")
        table.add_column("TTFT", style="yellow", justify="right")
        table.add_column("Time", style="white", justify="right")
        table.add_column("Memory", style="magenta", justify="right")
        table.add_column("Quant", style="dim")

        for result in report.results:
            # Merkitse nopein
            name = result.model_name
            if name == report.fastest_model:
                name = f"[bold green]{name}[/bold green]"

            table.add_row(
                name[:30],
                f"{result.tokens_per_second:.1f}",
                f"{result.time_to_first_token_ms:.0f} ms",
                f"{result.total_time_ms:.0f} ms",
                f"{result.memory_used_mb:.0f} MB",
                result.quantization or "-",
            )

        return table
