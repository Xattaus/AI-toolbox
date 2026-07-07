"""
AI TOOLBOX - llama.cpp Management
=================================

Handles llama.cpp setup, binary downloads, and tool location.
"""

import json
import shutil
import subprocess
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console

from ..core.paths import get_llama_cpp_dir

console = Console()

# Pinned llama.cpp release for reproducible builds. Both the source clone and
# the prebuilt Windows binaries are fetched at this tag. Bump to upgrade.
LLAMA_CPP_TAG = "b9902"


class LlamaCppManager:
    """Manages llama.cpp installation and binaries."""

    def __init__(self, llama_cpp_path: Optional[str] = None):
        """Initialize the manager."""
        if llama_cpp_path:
            self.llama_cpp_path = Path(llama_cpp_path)
        else:
            self.llama_cpp_path = get_llama_cpp_dir()

    def check_status(self) -> Dict[str, Any]:
        """Check if llama.cpp is available and return status."""
        status = {
            "installed": False,
            "path": str(self.llama_cpp_path),
            "convert_script": None,
            "quantize_binary": None,
        }

        if self.llama_cpp_path.exists():
            # Find convert script
            for script_name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py", "convert.py"]:
                script = self.llama_cpp_path / script_name
                if script.exists():
                    status["convert_script"] = str(script)
                    status["installed"] = True
                    break

            # Find quantize binary
            quantize = self.find_quantize_binary()
            if quantize:
                status["quantize_binary"] = str(quantize)

        return status

    def setup_from_git(self, force: bool = False) -> bool:
        """
        Set up llama.cpp by cloning the repository.

        Args:
            force: Force re-clone even if exists

        Returns:
            True if successful
        """
        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        is_properly_installed = self.llama_cpp_path.exists() and convert_script.exists()

        if is_properly_installed and not force:
            console.print(f"[green]llama.cpp already installed: {self.llama_cpp_path}[/green]")
            return True

        # Remove old/empty directory
        if self.llama_cpp_path.exists():
            console.print("[dim]Removing old llama.cpp directory...[/dim]")
            shutil.rmtree(self.llama_cpp_path)

        console.print(f"[cyan]Cloning llama.cpp repository (pinned {LLAMA_CPP_TAG})...[/cyan]")
        console.print("[dim]This may take a moment (~100 MB)...[/dim]")
        self.llama_cpp_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG,
                 "https://github.com/ggerganov/llama.cpp.git",
                 str(self.llama_cpp_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
            )

            if result.returncode != 0:
                console.print(f"[red]Clone failed: {result.stderr}[/red]")
                return False

            console.print("[green]llama.cpp cloned successfully![/green]")
            return True

        except subprocess.TimeoutExpired:
            console.print("[red]Clone timed out (5 min)[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Clone error: {e}[/red]")
            return False

    @staticmethod
    def _safe_extractall(zip_ref: "zipfile.ZipFile", dest: Path) -> None:
        """
        Extract a zip archive, rejecting members that would write outside
        `dest` (Zip Slip protection). Raises RuntimeError on a hostile path.
        """
        dest_resolved = Path(dest).resolve()
        for member in zip_ref.namelist():
            target = (dest_resolved / member).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError:
                raise RuntimeError(
                    f"Unsafe path in archive blocked (Zip Slip): {member}"
                )
        zip_ref.extractall(dest_resolved)

    def download_binaries(self, cuda: bool = True) -> bool:
        """
        Download pre-built llama.cpp binaries from GitHub releases.

        Args:
            cuda: Download CUDA version (True) or CPU-only version (False)

        Returns:
            True if successful
        """
        console.print("[cyan]Fetching llama.cpp binaries...[/cyan]")

        try:
            # Get pinned release info from GitHub API (reproducible builds)
            api_url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_TAG}"
            req = urllib.request.Request(api_url, headers={"User-Agent": "AI-Toolbox"})

            with urllib.request.urlopen(req, timeout=30) as response:
                release_data = json.loads(response.read().decode())

            tag_name = release_data.get("tag_name", LLAMA_CPP_TAG)
            console.print(f"[dim]Pinned version: {tag_name}[/dim]")

            # Find the right asset for Windows
            assets = release_data.get("assets", [])
            target_asset = None

            for asset in assets:
                name = asset.get("name", "")
                name_lower = name.lower()

                if not name.startswith("llama-b"):
                    continue
                if "win" not in name_lower or not name.endswith(".zip"):
                    continue

                if cuda and "cuda-12" in name_lower and "x64" in name_lower:
                    target_asset = asset
                    break
                elif not cuda and "cpu" in name_lower and "x64" in name_lower:
                    target_asset = asset
                    break

            # Fallback: any llama-b Windows x64 zip
            if not target_asset:
                for asset in assets:
                    name = asset.get("name", "")
                    if (name.startswith("llama-b") and "win" in name.lower()
                            and "x64" in name.lower() and name.endswith(".zip")):
                        target_asset = asset
                        break

            if not target_asset:
                console.print("[red]Windows binaries not found in release![/red]")
                console.print("[yellow]Download manually: https://github.com/ggml-org/llama.cpp/releases[/yellow]")
                return False

            download_url = target_asset.get("browser_download_url")
            file_name = target_asset.get("name")
            file_size = target_asset.get("size", 0)

            console.print(f"[dim]Downloading: {file_name} ({file_size / 1024 / 1024:.1f} MB)[/dim]")

            # Create directory and download
            self.llama_cpp_path.mkdir(parents=True, exist_ok=True)
            zip_path = self.llama_cpp_path / file_name

            # Download with progress
            req = urllib.request.Request(download_url, headers={"User-Agent": "AI-Toolbox"})
            with urllib.request.urlopen(req, timeout=300) as response:
                with open(zip_path, 'wb') as out_file:
                    total = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    block_size = 8192

                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if total > 0:
                            percent = (downloaded / total) * 100
                            console.print(f"\r[dim]Downloading... {percent:.1f}%[/dim]", end="")

            console.print()  # New line after progress

            # Extract zip (with Zip Slip protection)
            console.print("[dim]Extracting...[/dim]")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                self._safe_extractall(zip_ref, self.llama_cpp_path)

            # Remove zip file
            zip_path.unlink()

            # Check if llama-quantize exists now
            quantize_binary = self.find_quantize_binary()
            if quantize_binary:
                console.print("[green]llama.cpp binaries installed![/green]")
                console.print(f"[dim]llama-quantize: {quantize_binary}[/dim]")
                return True
            else:
                console.print("[yellow]Binaries downloaded but llama-quantize not found.[/yellow]")
                console.print(f"[dim]Check directory: {self.llama_cpp_path}[/dim]")
                return False

        except urllib.error.URLError as e:
            console.print(f"[red]Network error: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Download error: {e}[/red]")
            return False

    def find_convert_script(self) -> Optional[Path]:
        """Find the HuggingFace to GGUF convert script."""
        convert_script_candidates = [
            self.llama_cpp_path / "convert_hf_to_gguf.py",
            self.llama_cpp_path / "convert-hf-to-gguf.py",
            self.llama_cpp_path / "convert.py",
        ]

        for script in convert_script_candidates:
            if script.exists():
                return script

        return None

    def find_quantize_binary(self) -> Optional[Path]:
        """Find the llama-quantize binary."""
        binary_names = ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]
        search_paths = [
            self.llama_cpp_path,
            self.llama_cpp_path / "build" / "bin",
            self.llama_cpp_path / "build",
            Path.cwd(),
        ]

        # Also check subdirectories (for downloaded binaries)
        if self.llama_cpp_path.exists():
            for subdir in self.llama_cpp_path.iterdir():
                if subdir.is_dir():
                    search_paths.append(subdir)
                    if (subdir / "bin").exists():
                        search_paths.append(subdir / "bin")

        for search_path in search_paths:
            if not search_path.exists():
                continue
            for binary_name in binary_names:
                binary_path = search_path / binary_name
                if binary_path.exists():
                    return binary_path

        return None

    def ensure_available(self) -> bool:
        """Ensure llama.cpp is available, set up if needed."""
        status = self.check_status()
        if status["installed"]:
            return True

        # Try to set up from git first
        if self.setup_from_git():
            return True

        # Fall back to downloading binaries
        return self.download_binaries()
