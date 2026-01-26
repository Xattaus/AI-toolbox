"""
AI TOOLBOX - Dataset Prep
=========================

Datasetin valmistelu ja esikäsittely työkalut.
Tukee formaattimuunnoksia, puhdistusta, jakamista ja yhdistämistä.
"""

import json
import csv
import random
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

from rich.console import Console
from rich.table import Table
from rich import box

from ..core.paths import get_paths

console = Console()


class DatasetFormat(Enum):
    """Tuetut dataset-formaatit."""
    JSONL = "jsonl"
    JSON_ARRAY = "json_array"
    ALPACA = "alpaca"
    CHAT = "chat"
    SHAREGPT = "sharegpt"
    CSV = "csv"
    TEXT = "text"
    COMPLETION = "completion"
    UNKNOWN = "unknown"


class CleaningOperation(Enum):
    """Siivoustoimenpiteet."""
    REMOVE_EMPTY = "remove_empty"
    FIX_ENCODING = "fix_encoding"
    NORMALIZE_WHITESPACE = "normalize_whitespace"
    TRIM_TEXT = "trim_text"
    REMOVE_HTML = "remove_html"


@dataclass
class DatasetStats:
    """Datasetin tilastot."""
    total_samples: int = 0
    total_characters: int = 0
    avg_chars_per_sample: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    total_tokens: int = 0
    avg_tokens_per_sample: float = 0.0
    format_detected: Optional[str] = None
    schema: Dict[str, str] = field(default_factory=dict)
    field_fill_rates: Dict[str, float] = field(default_factory=dict)
    empty_rows: int = 0
    duplicate_count: int = 0


@dataclass
class SplitConfig:
    """Train/test/validation split konfiguraatio."""
    train_ratio: float = 0.8
    test_ratio: float = 0.1
    validation_ratio: float = 0.1
    shuffle: bool = True
    seed: int = 42


@dataclass
class FilterConfig:
    """Suodatuskonfiguraatio."""
    min_chars: Optional[int] = None
    max_chars: Optional[int] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    required_fields: List[str] = field(default_factory=list)


class DatasetPrep:
    """Dataset-valmistelutyökalu."""

    def __init__(self):
        """Alusta dataset prep."""
        paths = get_paths()
        self.datasets_dir = paths.datasets_dir
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = paths.processed_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tarkista riippuvuudet
        self._deps = self._check_dependencies()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Tarkista riippuvuudet."""
        deps = {
            "transformers": False,
            "rapidfuzz": False,
        }

        try:
            import transformers
            deps["transformers"] = True
        except ImportError:
            pass

        try:
            import rapidfuzz
            deps["rapidfuzz"] = True
        except ImportError:
            pass

        return deps

    def get_status(self) -> Dict[str, Any]:
        """Palauta dataset prepin status."""
        datasets = self.list_datasets()
        return {
            "datasets_count": len(datasets),
            "datasets_dir": str(self.datasets_dir),
            "output_dir": str(self.output_dir),
            "has_transformers": self._deps["transformers"],
            "has_rapidfuzz": self._deps["rapidfuzz"],
        }

    def list_datasets(self, include_processed: bool = False) -> List[Dict[str, Any]]:
        """Listaa datasets-kansion datasetit."""
        datasets = []
        dirs_to_scan = [self.datasets_dir]
        if include_processed:
            dirs_to_scan.append(self.output_dir)

        for scan_dir in dirs_to_scan:
            if not scan_dir.exists():
                continue
            for file_path in scan_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.json', '.jsonl', '.csv', '.txt']:
                    format_type = self.detect_format(file_path)
                    size = file_path.stat().st_size

                    datasets.append({
                        "path": file_path,
                        "name": file_path.name,
                        "format": format_type.value if format_type else "unknown",
                        "size_bytes": size,
                        "is_processed": scan_dir == self.output_dir,
                    })

        return sorted(datasets, key=lambda x: x["name"])

    # ==================== FORMAT DETECTION ====================

    def detect_format(self, file_path: Path) -> DatasetFormat:
        """Tunnista datasetin formaatti automaattisesti."""
        if not file_path.exists():
            return DatasetFormat.UNKNOWN

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return DatasetFormat.CSV

        if suffix == ".txt":
            return DatasetFormat.TEXT

        if suffix in [".json", ".jsonl"]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()

                    if first_line.startswith('['):
                        # JSON array - lue koko tiedosto
                        f.seek(0)
                        data = json.loads(f.read())
                        sample = data[0] if data else {}
                        is_array = True
                    else:
                        sample = json.loads(first_line)
                        is_array = False

                # Tunnista formaatti sisällön perusteella
                if "messages" in sample:
                    return DatasetFormat.CHAT
                elif "conversations" in sample:
                    return DatasetFormat.SHAREGPT
                elif "instruction" in sample:
                    return DatasetFormat.ALPACA
                elif "prompt" in sample and "completion" in sample:
                    return DatasetFormat.COMPLETION
                elif "text" in sample:
                    return DatasetFormat.TEXT
                else:
                    return DatasetFormat.JSON_ARRAY if is_array else DatasetFormat.JSONL

            except Exception:
                pass

        return DatasetFormat.UNKNOWN

    # ==================== INSPECTION ====================

    def inspect_dataset(self, file_path: Path,
                       sample_size: int = 1000) -> DatasetStats:
        """Analysoi dataset ja palauta tilastot."""
        stats = DatasetStats()

        if not file_path.exists():
            return stats

        format_type = self.detect_format(file_path)
        stats.format_detected = format_type.value

        try:
            data = self._load_dataset(file_path, limit=sample_size)
            stats.total_samples = len(data)

            if not data:
                return stats

            # Skeema ja field fill rates
            all_fields: Set[str] = set()
            field_counts: Counter = Counter()

            for item in data:
                if isinstance(item, dict):
                    all_fields.update(item.keys())
                    for key in item.keys():
                        if item[key]:  # Non-empty value
                            field_counts[key] += 1

            stats.schema = {k: "string" for k in sorted(all_fields)}
            stats.field_fill_rates = {
                k: round(v / len(data) * 100, 1)
                for k, v in field_counts.items()
            }

            # Merkkimäärät
            char_counts = []
            empty_count = 0

            for item in data:
                text = self._extract_text(item, format_type)
                char_count = len(text)
                char_counts.append(char_count)
                if char_count == 0:
                    empty_count += 1

            stats.total_characters = sum(char_counts)
            stats.avg_chars_per_sample = round(stats.total_characters / len(data), 1) if data else 0
            stats.min_chars = min(char_counts) if char_counts else 0
            stats.max_chars = max(char_counts) if char_counts else 0
            stats.empty_rows = empty_count

            # Duplikaattitarkistus (hash-pohjainen)
            seen_hashes: Set[str] = set()
            duplicate_count = 0
            for item in data:
                text = self._extract_text(item, format_type)
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    duplicate_count += 1
                seen_hashes.add(text_hash)

            stats.duplicate_count = duplicate_count

            # Laske todellinen kokonaismäärä (jos sample)
            if sample_size and file_path.suffix.lower() in [".json", ".jsonl"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content.startswith('['):
                        stats.total_samples = len(json.loads(content))
                    else:
                        stats.total_samples = sum(1 for line in content.split('\n') if line.strip())

        except Exception as e:
            console.print(f"[red]Virhe analysoinnissa: {e}[/red]")

        return stats

    def preview_samples(self, file_path: Path, n: int = 5) -> List[Dict[str, Any]]:
        """Palauta n esimerkkiriviä datasetista."""
        return self._load_dataset(file_path, limit=n)

    def get_schema(self, file_path: Path) -> Dict[str, str]:
        """Palauta datasetin skeema (kenttänimet ja tyypit)."""
        data = self._load_dataset(file_path, limit=100)
        if not data:
            return {}

        all_fields: Set[str] = set()
        for item in data:
            if isinstance(item, dict):
                all_fields.update(item.keys())

        # Tunnista tyypit
        schema = {}
        for field_name in sorted(all_fields):
            sample_values = [item.get(field_name) for item in data if field_name in item]
            if sample_values:
                first_val = sample_values[0]
                if isinstance(first_val, str):
                    schema[field_name] = "string"
                elif isinstance(first_val, (int, float)):
                    schema[field_name] = "number"
                elif isinstance(first_val, list):
                    schema[field_name] = "array"
                elif isinstance(first_val, dict):
                    schema[field_name] = "object"
                else:
                    schema[field_name] = "unknown"

        return schema

    # ==================== FORMAT CONVERSION ====================

    def convert_format(self,
                      input_path: Path,
                      output_path: Path,
                      target_format: DatasetFormat,
                      field_mapping: Optional[Dict[str, str]] = None,
                      progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Muunna dataset toiseen formaattiin."""
        result = {
            "success": False,
            "num_samples": 0,
            "output_path": str(output_path),
            "errors": [],
        }

        if not input_path.exists():
            result["errors"].append(f"Tiedostoa ei löydy: {input_path}")
            return result

        try:
            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(input_path)
            source_format = self.detect_format(input_path)

            if progress_callback:
                progress_callback(f"Muunnetaan {len(data)} näytettä...")

            converted = []
            for i, item in enumerate(data):
                try:
                    converted_item = self._convert_item(item, source_format, target_format, field_mapping)
                    if converted_item:
                        converted.append(converted_item)
                except Exception as e:
                    result["errors"].append(f"Rivi {i+1}: {str(e)}")

            if progress_callback:
                progress_callback("Tallennetaan...")

            # Tallenna kohdeformaatissa
            self._save_dataset(converted, output_path, target_format)

            result["success"] = True
            result["num_samples"] = len(converted)

        except Exception as e:
            result["errors"].append(str(e))

        return result

    def _convert_item(self, item: Dict[str, Any],
                     source_format: DatasetFormat,
                     target_format: DatasetFormat,
                     field_mapping: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Muunna yksittäinen item formaatista toiseen."""

        # Käytä kenttämäppäystä jos annettu
        if field_mapping:
            mapped_item = {}
            for target_key, source_key in field_mapping.items():
                if source_key in item:
                    mapped_item[target_key] = item[source_key]
            item = mapped_item

        # Muunna kohdeformaattiin
        if target_format == DatasetFormat.ALPACA:
            return self._to_alpaca(item, source_format)
        elif target_format == DatasetFormat.CHAT:
            return self._to_chat(item, source_format)
        elif target_format == DatasetFormat.SHAREGPT:
            return self._to_sharegpt(item, source_format)
        elif target_format == DatasetFormat.COMPLETION:
            return self._to_completion(item, source_format)
        elif target_format in [DatasetFormat.JSONL, DatasetFormat.JSON_ARRAY]:
            return item
        else:
            return item

    def _to_alpaca(self, item: Dict[str, Any], source_format: DatasetFormat) -> Dict[str, Any]:
        """Muunna Alpaca-formaattiin."""
        if source_format == DatasetFormat.ALPACA:
            return item

        if source_format == DatasetFormat.CHAT:
            messages = item.get("messages", [])
            instruction = ""
            output = ""
            for msg in messages:
                if msg.get("role") == "user":
                    instruction = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    output = msg.get("content", "")
            return {"instruction": instruction, "input": "", "output": output}

        if source_format == DatasetFormat.SHAREGPT:
            conversations = item.get("conversations", [])
            instruction = ""
            output = ""
            for conv in conversations:
                if conv.get("from") in ["human", "user"]:
                    instruction = conv.get("value", "")
                elif conv.get("from") in ["gpt", "assistant"]:
                    output = conv.get("value", "")
            return {"instruction": instruction, "input": "", "output": output}

        if source_format == DatasetFormat.COMPLETION:
            return {
                "instruction": item.get("prompt", ""),
                "input": "",
                "output": item.get("completion", ""),
            }

        # Fallback
        return {
            "instruction": item.get("instruction", item.get("prompt", item.get("text", ""))),
            "input": item.get("input", ""),
            "output": item.get("output", item.get("response", item.get("completion", ""))),
        }

    def _to_chat(self, item: Dict[str, Any], source_format: DatasetFormat) -> Dict[str, Any]:
        """Muunna Chat/Messages-formaattiin."""
        if source_format == DatasetFormat.CHAT:
            return item

        messages = []

        if source_format == DatasetFormat.ALPACA:
            user_content = item.get("instruction", "")
            if item.get("input"):
                user_content += f"\n\n{item['input']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": item.get("output", "")})

        elif source_format == DatasetFormat.SHAREGPT:
            conversations = item.get("conversations", [])
            for conv in conversations:
                role = "user" if conv.get("from") in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": conv.get("value", "")})

        elif source_format == DatasetFormat.COMPLETION:
            messages.append({"role": "user", "content": item.get("prompt", "")})
            messages.append({"role": "assistant", "content": item.get("completion", "")})

        else:
            # Fallback
            messages.append({"role": "user", "content": item.get("instruction", item.get("text", ""))})
            messages.append({"role": "assistant", "content": item.get("output", item.get("response", ""))})

        return {"messages": messages}

    def _to_sharegpt(self, item: Dict[str, Any], source_format: DatasetFormat) -> Dict[str, Any]:
        """Muunna ShareGPT-formaattiin."""
        if source_format == DatasetFormat.SHAREGPT:
            return item

        conversations = []

        if source_format == DatasetFormat.CHAT:
            messages = item.get("messages", [])
            for msg in messages:
                from_val = "human" if msg.get("role") == "user" else "gpt"
                conversations.append({"from": from_val, "value": msg.get("content", "")})

        elif source_format == DatasetFormat.ALPACA:
            user_content = item.get("instruction", "")
            if item.get("input"):
                user_content += f"\n\n{item['input']}"
            conversations.append({"from": "human", "value": user_content})
            conversations.append({"from": "gpt", "value": item.get("output", "")})

        else:
            conversations.append({"from": "human", "value": item.get("prompt", item.get("text", ""))})
            conversations.append({"from": "gpt", "value": item.get("completion", item.get("response", ""))})

        return {"conversations": conversations}

    def _to_completion(self, item: Dict[str, Any], source_format: DatasetFormat) -> Dict[str, Any]:
        """Muunna prompt-completion formaattiin."""
        if source_format == DatasetFormat.COMPLETION:
            return item

        if source_format == DatasetFormat.ALPACA:
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt += f"\n{item['input']}"
            return {"prompt": prompt, "completion": item.get("output", "")}

        if source_format == DatasetFormat.CHAT:
            messages = item.get("messages", [])
            prompt = ""
            completion = ""
            for msg in messages:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    completion = msg.get("content", "")
            return {"prompt": prompt, "completion": completion}

        return {"prompt": item.get("text", ""), "completion": item.get("response", "")}

    # ==================== DATA CLEANING ====================

    def clean_dataset(self,
                     input_path: Path,
                     output_path: Path,
                     operations: List[CleaningOperation],
                     progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Siivoa dataset valituilla operaatioilla."""
        result = {
            "success": False,
            "original_count": 0,
            "final_count": 0,
            "removed_count": 0,
            "operations_applied": [],
        }

        try:
            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(input_path)
            result["original_count"] = len(data)
            format_type = self.detect_format(input_path)

            for op in operations:
                if progress_callback:
                    progress_callback(f"Suoritetaan: {op.value}...")

                if op == CleaningOperation.REMOVE_EMPTY:
                    data = self._remove_empty_rows(data, format_type)
                elif op == CleaningOperation.FIX_ENCODING:
                    data = self._fix_encoding(data)
                elif op == CleaningOperation.NORMALIZE_WHITESPACE:
                    data = self._normalize_whitespace(data)
                elif op == CleaningOperation.TRIM_TEXT:
                    data = self._trim_text(data)
                elif op == CleaningOperation.REMOVE_HTML:
                    data = self._remove_html(data)

                result["operations_applied"].append(op.value)

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_dataset(data, output_path, format_type)

            result["success"] = True
            result["final_count"] = len(data)
            result["removed_count"] = result["original_count"] - result["final_count"]

        except Exception as e:
            result["error"] = str(e)

        return result

    def _remove_empty_rows(self, data: List[Dict], format_type: DatasetFormat) -> List[Dict]:
        """Poista tyhjät rivit."""
        cleaned = []
        for item in data:
            text = self._extract_text(item, format_type)
            if text.strip():
                cleaned.append(item)
        return cleaned

    def _fix_encoding(self, data: List[Dict]) -> List[Dict]:
        """Korjaa merkistöongelmat."""
        def fix_text(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Korjaa yleisiä ongelmia
            text = text.replace('\x00', '')  # Null bytes
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            return text

        return self._apply_to_strings(data, fix_text)

    def _normalize_whitespace(self, data: List[Dict]) -> List[Dict]:
        """Normalisoi välilyönnit ja rivinvaihdot."""
        def normalize(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Korvaa monet välilyönnit yhdellä
            text = re.sub(r'[ \t]+', ' ', text)
            # Korvaa monet rivinvaihdot kahdella
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()

        return self._apply_to_strings(data, normalize)

    def _trim_text(self, data: List[Dict]) -> List[Dict]:
        """Trimmaa tekstit."""
        def trim(text: str) -> str:
            if not isinstance(text, str):
                return text
            return text.strip()

        return self._apply_to_strings(data, trim)

    def _remove_html(self, data: List[Dict]) -> List[Dict]:
        """Poista HTML-tagit."""
        def remove_html(text: str) -> str:
            if not isinstance(text, str):
                return text
            return re.sub(r'<[^>]+>', '', text)

        return self._apply_to_strings(data, remove_html)

    def _apply_to_strings(self, data: List[Dict], func: Callable[[str], str]) -> List[Dict]:
        """Sovella funktiota kaikkiin string-kenttiin."""
        result = []
        for item in data:
            new_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    new_item[key] = func(value)
                elif isinstance(value, list):
                    new_item[key] = [
                        {k: func(v) if isinstance(v, str) else v for k, v in elem.items()}
                        if isinstance(elem, dict) else elem
                        for elem in value
                    ]
                else:
                    new_item[key] = value
            result.append(new_item)
        return result

    # ==================== DEDUPLICATION ====================

    def deduplicate(self,
                   input_path: Path,
                   output_path: Path,
                   method: str = "exact",
                   threshold: float = 0.9,
                   progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Poista duplikaatit datasetista."""
        result = {
            "success": False,
            "original_count": 0,
            "final_count": 0,
            "duplicates_removed": 0,
        }

        try:
            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(input_path)
            result["original_count"] = len(data)
            format_type = self.detect_format(input_path)

            if progress_callback:
                progress_callback(f"Poistetaan duplikaatit ({method})...")

            if method == "exact":
                data = self._exact_dedupe(data, format_type)
            elif method == "fuzzy":
                if not self._deps["rapidfuzz"]:
                    result["error"] = "Fuzzy-dedupe vaatii rapidfuzz-kirjaston"
                    return result
                data = self._fuzzy_dedupe(data, format_type, threshold)

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_dataset(data, output_path, format_type)

            result["success"] = True
            result["final_count"] = len(data)
            result["duplicates_removed"] = result["original_count"] - result["final_count"]

        except Exception as e:
            result["error"] = str(e)

        return result

    def _exact_dedupe(self, data: List[Dict], format_type: DatasetFormat) -> List[Dict]:
        """Tarkka duplikaattien poisto."""
        seen: Set[str] = set()
        unique = []

        for item in data:
            text = self._extract_text(item, format_type)
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(item)

        return unique

    def _fuzzy_dedupe(self, data: List[Dict], format_type: DatasetFormat, threshold: float) -> List[Dict]:
        """Sumea duplikaattien poisto."""
        from rapidfuzz import fuzz

        unique = []
        unique_texts = []

        for item in data:
            text = self._extract_text(item, format_type)
            is_duplicate = False

            for existing_text in unique_texts:
                similarity = fuzz.ratio(text, existing_text) / 100
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(item)
                unique_texts.append(text)

        return unique

    # ==================== SPLITTING ====================

    def split_dataset(self,
                     input_path: Path,
                     output_dir: Path,
                     config: SplitConfig,
                     progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Jaa dataset train/test/validation osiin."""
        result = {
            "success": False,
            "total_samples": 0,
            "splits": {},
        }

        try:
            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(input_path)
            result["total_samples"] = len(data)
            format_type = self.detect_format(input_path)

            # Sekoita jos halutaan
            if config.shuffle:
                random.seed(config.seed)
                random.shuffle(data)

            # Laske indeksit
            n = len(data)
            train_end = int(n * config.train_ratio)
            test_end = train_end + int(n * config.test_ratio)

            splits = {
                "train": data[:train_end],
                "test": data[train_end:test_end],
            }

            if config.validation_ratio > 0:
                splits["validation"] = data[test_end:]

            # Tallenna splitit
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = input_path.stem

            for split_name, split_data in splits.items():
                if not split_data:
                    continue

                if progress_callback:
                    progress_callback(f"Tallennetaan {split_name}...")

                output_path = output_dir / f"{base_name}_{split_name}.jsonl"
                self._save_dataset(split_data, output_path, format_type)

                result["splits"][split_name] = {
                    "count": len(split_data),
                    "path": str(output_path),
                }

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    # ==================== FILTERING ====================

    def filter_dataset(self,
                      input_path: Path,
                      output_path: Path,
                      config: FilterConfig,
                      tokenizer_name: Optional[str] = None,
                      progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Suodata dataset kriteerien mukaan."""
        result = {
            "success": False,
            "original_count": 0,
            "filtered_count": 0,
            "removed_count": 0,
            "removed_too_short": 0,
            "removed_too_long": 0,
            "removed_missing_fields": 0,
        }

        try:
            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(input_path)
            result["original_count"] = len(data)
            format_type = self.detect_format(input_path)

            # Lataa tokenizer jos tarvitaan
            tokenizer = None
            if (config.min_tokens or config.max_tokens) and tokenizer_name:
                if self._deps["transformers"]:
                    if progress_callback:
                        progress_callback("Ladataan tokenizer...")
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if progress_callback:
                progress_callback("Suodatetaan...")

            filtered = []
            for item in data:
                text = self._extract_text(item, format_type)

                # Merkkimääräsuodatus
                if config.min_chars and len(text) < config.min_chars:
                    result["removed_too_short"] += 1
                    continue
                if config.max_chars and len(text) > config.max_chars:
                    result["removed_too_long"] += 1
                    continue

                # Tokenimääräsuodatus
                if tokenizer and (config.min_tokens or config.max_tokens):
                    tokens = tokenizer.encode(text)
                    if config.min_tokens and len(tokens) < config.min_tokens:
                        result["removed_too_short"] += 1
                        continue
                    if config.max_tokens and len(tokens) > config.max_tokens:
                        result["removed_too_long"] += 1
                        continue

                # Pakolliset kentät
                if config.required_fields:
                    missing = False
                    for field in config.required_fields:
                        if field not in item or not item[field]:
                            missing = True
                            break
                    if missing:
                        result["removed_missing_fields"] += 1
                        continue

                filtered.append(item)

            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_dataset(filtered, output_path, format_type)

            result["success"] = True
            result["filtered_count"] = len(filtered)
            result["removed_count"] = result["original_count"] - result["filtered_count"]

        except Exception as e:
            result["error"] = str(e)

        return result

    # ==================== MERGING ====================

    def merge_datasets(self,
                      input_paths: List[Path],
                      output_path: Path,
                      deduplicate_result: bool = True,
                      shuffle: bool = True,
                      progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Yhdistä useita datasettejä."""
        result = {
            "success": False,
            "files_merged": 0,
            "total_samples": 0,
            "final_count": 0,
            "duplicates_removed": 0,
        }

        try:
            all_data = []
            target_format = None

            for i, path in enumerate(input_paths):
                if progress_callback:
                    progress_callback(f"Ladataan {path.name}...")

                if not path.exists():
                    continue

                data = self._load_dataset(path)
                all_data.extend(data)
                result["files_merged"] += 1

                # Käytä ensimmäisen tiedoston formaattia
                if target_format is None:
                    target_format = self.detect_format(path)

            result["total_samples"] = len(all_data)

            # Deduplikointi
            if deduplicate_result and target_format:
                if progress_callback:
                    progress_callback("Poistetaan duplikaatit...")
                all_data = self._exact_dedupe(all_data, target_format)
                result["duplicates_removed"] = result["total_samples"] - len(all_data)

            # Sekoitus
            if shuffle:
                if progress_callback:
                    progress_callback("Sekoitetaan...")
                random.shuffle(all_data)

            # Tallenna
            if progress_callback:
                progress_callback("Tallennetaan...")

            self._save_dataset(all_data, output_path, target_format or DatasetFormat.JSONL)

            result["success"] = True
            result["final_count"] = len(all_data)

        except Exception as e:
            result["error"] = str(e)

        return result

    # ==================== TOKEN COUNTING ====================

    def count_tokens(self,
                    file_path: Path,
                    tokenizer_name: str = "gpt2",
                    sample_size: Optional[int] = None,
                    progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Laske tokenimäärät datasetille."""
        result = {
            "success": False,
            "total_tokens": 0,
            "avg_tokens_per_sample": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "samples_counted": 0,
        }

        if not self._deps["transformers"]:
            result["error"] = "Token counting vaatii transformers-kirjaston"
            return result

        try:
            if progress_callback:
                progress_callback("Ladataan tokenizer...")

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if progress_callback:
                progress_callback("Ladataan dataa...")

            data = self._load_dataset(file_path, limit=sample_size)
            format_type = self.detect_format(file_path)

            if progress_callback:
                progress_callback("Lasketaan tokeneita...")

            token_counts = []
            for item in data:
                text = self._extract_text(item, format_type)
                tokens = tokenizer.encode(text)
                token_counts.append(len(tokens))

            if token_counts:
                result["success"] = True
                result["total_tokens"] = sum(token_counts)
                result["avg_tokens_per_sample"] = round(sum(token_counts) / len(token_counts), 1)
                result["min_tokens"] = min(token_counts)
                result["max_tokens"] = max(token_counts)
                result["samples_counted"] = len(token_counts)

        except Exception as e:
            result["error"] = str(e)

        return result

    # ==================== HELPER METHODS ====================

    def _load_dataset(self, file_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lataa dataset tiedostosta."""
        data = []
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".csv":
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        data.append(dict(row))
                        if limit and i >= limit - 1:
                            break

            elif suffix == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            data.append({"text": line.strip()})
                        if limit and i >= limit - 1:
                            break

            elif suffix in [".json", ".jsonl"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                    if content.startswith('['):
                        # JSON array
                        all_data = json.loads(content)
                        data = all_data[:limit] if limit else all_data
                    else:
                        # JSONL
                        for i, line in enumerate(content.split('\n')):
                            if line.strip():
                                data.append(json.loads(line))
                            if limit and i >= limit - 1:
                                break

        except Exception as e:
            console.print(f"[red]Virhe ladattaessa: {e}[/red]")

        return data

    def _save_dataset(self, data: List[Dict[str, Any]], output_path: Path,
                     format_type: DatasetFormat) -> None:
        """Tallenna dataset tiedostoon."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == DatasetFormat.CSV:
            if data:
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        elif format_type == DatasetFormat.JSON_ARRAY:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # JSONL (default)
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _extract_text(self, item: Dict[str, Any], format_type: DatasetFormat) -> str:
        """Pura teksti itemista formaatin mukaan."""
        if format_type == DatasetFormat.ALPACA:
            parts = [
                item.get("instruction", ""),
                item.get("input", ""),
                item.get("output", ""),
            ]
            return " ".join(p for p in parts if p)

        elif format_type == DatasetFormat.CHAT:
            messages = item.get("messages", [])
            return " ".join(m.get("content", "") for m in messages)

        elif format_type == DatasetFormat.SHAREGPT:
            conversations = item.get("conversations", [])
            return " ".join(c.get("value", "") for c in conversations)

        elif format_type == DatasetFormat.COMPLETION:
            return f"{item.get('prompt', '')} {item.get('completion', '')}"

        elif format_type == DatasetFormat.TEXT:
            return item.get("text", "")

        else:
            # Yritä löytää mikä tahansa teksti
            for key in ["text", "content", "instruction", "prompt", "input", "output"]:
                if key in item and item[key]:
                    return str(item[key])
            return str(item)
