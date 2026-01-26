"""
AI TOOLBOX - Tekoäly-chat
=========================

Interaktiivinen chat-käyttöliittymä, joka käyttää paikallista mallia
ja voi suorittaa toolbox-työkaluja käyttäjän puolesta.

Tuetut backendit:
- Sisäänrakennettu (llama-cpp-python) - Suora GGUF-mallin ajo
- Ollama - Ulkoinen Ollama-palvelu
"""

import json
import re
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Generator
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich import box

import questionary
from questionary import Style

from ..core.ui import (
    print_mini_banner,
    print_success,
    print_error,
    print_warning,
    print_info,
    format_size,
    console,
)

# Questionary-tyyli
custom_style = Style([
    ('qmark', 'fg:#ff9d00 bold'),
    ('question', 'fg:white bold'),
    ('answer', 'fg:#00d7ff bold'),
    ('pointer', 'fg:#ff9d00 bold'),
    ('highlighted', 'fg:#ff9d00 bold'),
    ('selected', 'fg:#00ff00'),
])


@dataclass
class ChatMessage:
    """Yksittäinen chat-viesti."""
    role: str  # "user", "assistant", "system"
    content: str


# =============================================================================
# BACKEND-ABSTRAKTIO
# =============================================================================

class ChatBackend(ABC):
    """Abstrakti pohjaluokka chat-backendeille."""

    @abstractmethod
    def is_available(self) -> bool:
        """Tarkista onko backend käytettävissä."""
        pass

    @abstractmethod
    def get_models(self) -> List[Dict[str, Any]]:
        """Listaa saatavilla olevat mallit."""
        pass

    @abstractmethod
    def load_model(self, model_path_or_name: str) -> bool:
        """Lataa malli."""
        pass

    @abstractmethod
    def unload_model(self):
        """Vapauta malli muistista."""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None]:
        """Lähetä chat-pyyntö ja palauta vastaus generaattorina."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Palauta backendin nimi."""
        pass


# =============================================================================
# LLAMA-CPP-PYTHON BACKEND (Sisäänrakennettu)
# =============================================================================

class LlamaCppBackend(ChatBackend):
    """Sisäänrakennettu backend käyttäen llama-cpp-python -kirjastoa."""

    def __init__(self):
        self.llm = None
        self.model_path: Optional[str] = None
        self._llama_cpp_available = None

    def _check_llama_cpp(self) -> bool:
        """Tarkista onko llama-cpp-python asennettu."""
        if self._llama_cpp_available is None:
            try:
                from llama_cpp import Llama
                self._llama_cpp_available = True
            except ImportError:
                self._llama_cpp_available = False
        return self._llama_cpp_available

    def is_available(self) -> bool:
        return self._check_llama_cpp()

    def get_name(self) -> str:
        return "Built-in (llama-cpp-python)"

    def get_models(self) -> List[Dict[str, Any]]:
        """Palauta tyhjä lista - mallit valitaan kirjastosta/levyltä."""
        return []

    def load_model(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> bool:
        """
        Lataa GGUF-malli.

        Args:
            model_path: Polku GGUF-tiedostoon
            n_ctx: Kontekstin pituus (oletus 4096)
            n_threads: CPU-säikeiden määrä (oletus: automaattinen)
            n_gpu_layers: GPU:lle siirrettävien kerrosten määrä (0 = vain CPU)
            verbose: Näytä latausviestit
        """
        if not self._check_llama_cpp():
            print_error("llama-cpp-python ei ole asennettu!")
            print_info("Asenna: pip install llama-cpp-python")
            return False

        if not Path(model_path).exists():
            print_error(f"Mallitiedostoa ei löydy: {model_path}")
            return False

        try:
            from llama_cpp import Llama

            # Vapauta vanha malli
            self.unload_model()

            # Määritä säikeiden määrä
            if n_threads is None:
                import multiprocessing
                n_threads = max(1, multiprocessing.cpu_count() - 1)

            console.print(Panel(
                f"[cyan]Ladataan mallia...[/cyan]\n\n"
                f"[dim]Konteksti: {n_ctx} tokenia\n"
                f"Säikeet: {n_threads}\n"
                f"GPU-kerrokset: {n_gpu_layers}[/dim]",
                title="[bold]Mallin lataus[/bold]",
                border_style="cyan",
                box=box.ROUNDED,
            ))

            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

            console.print("[green]Malli ladattu onnistuneesti![/green]")

            self.model_path = model_path
            return True

        except Exception as e:
            print_error(f"Mallin lataus epäonnistui: {e}")
            return False

    def unload_model(self):
        """Vapauta malli muistista."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.model_path = None
            # Yritä vapauttaa muistia
            import gc
            gc.collect()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        format: Optional[str] = None,
    ) -> str:
        """
        Generoi vastaus säädettävillä parametreilla (ei-streaming).

        Käytetään Two-Pass arkkitehtuurissa:
        - Router pass: temperature=0, max_tokens=256, format="json"
        - Response pass: temperature=0.7, max_tokens=2048

        Args:
            format: "json" ohjaa mallia tuottamaan JSONia (llama.cpp ei pakota)
        """
        if self.llm is None:
            return "Virhe: Mallia ei ole ladattu."

        try:
            prompt = self._format_chat_prompt(messages)

            # llama-cpp-python ei tue format-parametria natiivisti,
            # mutta prompti ohjaa mallia tuottamaan JSONia
            response = self.llm(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["<|im_end|>", "</s>", "<|end|>", "[/INST]", "Human:", "User:"],
                echo=False,
            )

            return response.get("choices", [{}])[0].get("text", "").strip()

        except Exception as e:
            return f"Virhe: {e}"

    def chat(self, messages: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None]:
        """Lähetä chat-pyyntö (streaming, oletus temperature=0.7)."""
        if self.llm is None:
            yield "Virhe: Mallia ei ole ladattu."
            return

        try:
            # Muodosta prompt chat-viesteistä
            prompt = self._format_chat_prompt(messages)

            if stream:
                # Streaming-vastaus
                response = self.llm(
                    prompt,
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=["<|im_end|>", "</s>", "<|end|>", "[/INST]", "Human:", "User:"],
                    stream=True,
                    echo=False,
                )

                for chunk in response:
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
            else:
                # Ei-streaming vastaus
                response = self.llm(
                    prompt,
                    temperature=0.7,
                    max_tokens=2048,
                    stop=["<|im_end|>", "</s>", "<|end|>", "[/INST]", "Human:", "User:"],
                    echo=False,
                )
                yield response.get("choices", [{}])[0].get("text", "")

        except Exception as e:
            yield f"Virhe: {e}"

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Muotoile viestit promptiksi.
        Käyttää ChatML-formaattia, joka toimii useimmilla malleilla.
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Lisää assistant-aloitus
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)


# =============================================================================
# OLLAMA BACKEND
# =============================================================================

class OllamaBackend(ChatBackend):
    """Backend Ollama-palvelulle."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model: Optional[str] = None

    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_name(self) -> str:
        return "Ollama"

    def get_models(self) -> List[Dict[str, Any]]:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get("models", [])
        except Exception:
            pass
        return []

    def load_model(self, model_name: str) -> bool:
        self.model = model_name
        return True

    def unload_model(self):
        self.model = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        format: Optional[str] = None,
    ) -> str:
        """
        Generoi vastaus säädettävillä parametreilla (ei-streaming).

        Käytetään Two-Pass arkkitehtuurissa.

        Args:
            format: "json" pakottaa Ollaman tuottamaan validia JSONia
        """
        if not self.model:
            return "Virhe: Mallia ei ole valittu."

        try:
            import requests

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "num_predict": max_tokens,
                }
            }

            # Lisää format jos määritelty (esim. "json")
            if format:
                payload["format"] = format

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "").strip()
            else:
                return f"Virhe: HTTP {response.status_code}"

        except Exception as e:
            return f"Virhe: {e}"

    def chat(self, messages: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None]:
        if not self.model:
            yield "Virhe: Mallia ei ole valittu."
            return

        try:
            import requests

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            }

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
                timeout=300,
            )

            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                text = data["message"].get("content", "")
                                if text:
                                    yield text
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                yield data.get("message", {}).get("content", "")

        except Exception as e:
            yield f"Virhe: {e}"


# =============================================================================
# TYÖKALU-REKISTERI
# =============================================================================

class ToolRegistry:
    """Rekisteri käytettävissä olevista työkaluista."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable):
        """Rekisteröi uusi työkalu."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self.handlers[name] = handler

    def get_tools_description(self) -> str:
        """Palauttaa kuvauksen kaikista työkaluista promptia varten."""
        if not self.tools:
            return "Ei työkaluja käytettävissä."

        # Esimerkkiarvot parametreille
        example_values = {
            "query": '"llama"',
            "format": '"gguf"',
            "limit": '5',
            "model_id": '"meta-llama/Llama-2-7b"',
            "parameters_b": '7',
        }

        lines = []
        for name, tool in self.tools.items():
            lines.append(f"**{name}**: {tool['description']}")

            # Näytä parametrit selkeästi esimerkkiarvoilla
            if tool['parameters']:
                params_list = []
                for param_name, param_info in tool['parameters'].items():
                    example = example_values.get(param_name, '"arvo"')
                    if param_info.get('required'):
                        params_list.append(f'"{param_name}": {example}')
                    else:
                        params_list.append(f'"{param_name}": {example}')

                params_str = ", ".join(params_list)
                lines.append(f'  Esimerkki: {{"tool": "{name}", "params": {{{params_str}}}}}')
            else:
                lines.append(f'  Esimerkki: {{"tool": "{name}", "params": {{}}}}')

            lines.append("")

        return "\n".join(lines)

    def execute(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Suorita työkalu."""
        if name not in self.handlers:
            return {"error": f"Tuntematon työkalu: {name}"}

        try:
            result = self.handlers[name](**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# AI CHAT -PÄÄLUOKKA
# =============================================================================

class AIChat:
    """Interaktiivinen AI-chat Two-Pass arkkitehtuurilla."""

    # =========================================================================
    # TWO-PASS PROMPTS
    # =========================================================================

    # PASS 1: ROUTER - Tiukka, looginen (temperature=0)
    # Päättää vain: tarvitaanko työkalua?
    ROUTER_PROMPT = """Analysoi kayttajan viesti ja paata tarvitaanko tyokalua.

TYOKALUT:
{tools_json}

OHJEET:
- Jos kayttaja kysyy tietoa jarjestelmasta (mallit, kirjasto, VRAM) -> kayta tyokalua
- Jos kayttaja kysyy yleista tietoa (mita on kvantisointi?) -> EI tyokalua
- Jos kayttaja tervehtii tai juttelee -> EI tyokalua

VASTAA AINA TASSA JSON-MUODOSSA:
{{"tool": "tyokalun_nimi", "params": {{"param": "arvo"}}}}

TAI jos tyokalua ei tarvita:
{{"tool": null}}

ALA KIRJOITA MITAAN MUUTA. VAIN JSON."""

    # PASS 2: RESPONSE - Selkeä, tarkka (ei liian luova)
    RESPONSE_PROMPT = """Olet AI Toolboxin avustaja.

Tehtavasi: Muotoile tyokalun data selkeaksi suomenkieliseksi vastaukseksi.

# KRIITTISET SAANNOT

1. PERUSTA vastaus VAIN alla olevaan dataan. Ala keksi mitaan.
2. ALA kirjoita ajatusprosessiasi ("Let me think", "I should").
3. Kayta LUONNOLLISTA suomea:
   - EI: "tekstidominoilla", "puhenyminen", "luovittaa"
   - KYLLA: "tekstiaineistolla", "puheentunnistus", "luovuutta"
4. Tekniset termit: Kayta englantia `koodilohkossa` jos et tieda suomenkielista termia.
   - Esim: `text-generation`, `safetensors`, `pipeline_tag`
5. Mallin nimet ja ID:t: Sailyta sellaisenaan (`meta-llama/Llama-2-7b`).

# MUOTOILU

- Kayta Markdown-muotoilua (## otsikot, **lihavointi**, taulukot)
- Ole ytimekas, mutta informatiivinen
- Kaytetyt emojit: ✅ ❌ 📦 🔧 💾 (max 2-3 per vastaus)

{context}

# VASTAA SUOMEKSI. VAIN LOPULLINEN VASTAUS."""

    # Vanha SYSTEM_PROMPT yhteensopivuuden vuoksi (Ollama)
    SYSTEM_PROMPT = RESPONSE_PROMPT

    def __init__(self, library=None, downloader=None, converter=None):
        self.backend: Optional[ChatBackend] = None
        self.tools = ToolRegistry()
        self.messages: List[ChatMessage] = []
        self.library = library
        self.downloader = downloader
        self.converter = converter

        # Rekisteröi työkalut
        self._register_tools()

    def _register_tools(self):
        """Rekisteröi käytettävissä olevat työkalut."""

        # Mallikirjaston työkalut
        self.tools.register(
            name="listaa_mallit",
            description="Listaa kaikki mallikirjaston mallit",
            parameters={
                "format": {
                    "description": "Suodata formaatin mukaan (gguf, safetensors, pytorch)",
                    "required": False,
                }
            },
            handler=self._tool_list_models,
        )

        self.tools.register(
            name="etsi_malleja",
            description="Etsi malleja kirjastosta nimen tai tagien perusteella",
            parameters={
                "query": {
                    "description": "Hakusana",
                    "required": True,
                }
            },
            handler=self._tool_search_models,
        )

        self.tools.register(
            name="kirjaston_tilastot",
            description="Näytä mallikirjaston tilastot",
            parameters={},
            handler=self._tool_library_stats,
        )

        # HuggingFace-työkalut
        self.tools.register(
            name="hae_huggingface",
            description="Hae malleja HuggingFace-palvelusta",
            parameters={
                "query": {
                    "description": "Hakusana (esim. 'llama', 'mistral', 'qwen')",
                    "required": True,
                },
                "limit": {
                    "description": "Tulosten enimmäismäärä (oletus 5)",
                    "required": False,
                }
            },
            handler=self._tool_search_huggingface,
        )

        self.tools.register(
            name="mallin_tiedot",
            description="Hae mallin yksityiskohtaiset tiedot HuggingFace-palvelusta",
            parameters={
                "model_id": {
                    "description": "Mallin ID (esim. 'meta-llama/Llama-2-7b-hf')",
                    "required": True,
                }
            },
            handler=self._tool_model_details,
        )

        # VRAM-laskin
        self.tools.register(
            name="laske_vram",
            description="Laske mallin VRAM-vaatimukset eri kvantisoinneilla",
            parameters={
                "parameters_b": {
                    "description": "Mallin parametrien määrä miljardeina (esim. 7, 13, 70)",
                    "required": True,
                }
            },
            handler=self._tool_calculate_vram,
        )

        # Kvantisointi-info
        self.tools.register(
            name="kvantisointityypit",
            description="Näytä saatavilla olevat kvantisointityypit ja niiden kuvaukset",
            parameters={},
            handler=self._tool_quantization_types,
        )

        # =========================================================================
        # TOIMINNALLISET TYÖKALUT (lataus, muunnos, kvantisointi)
        # =========================================================================

        # Mallin lataus
        self.tools.register(
            name="lataa_malli",
            description="Lataa malli HuggingFacesta paikalliseen kansioon",
            parameters={
                "model_id": {
                    "description": "Mallin HuggingFace ID (esim. 'meta-llama/Llama-2-7b-hf')",
                    "required": True,
                }
            },
            handler=self._tool_download_model,
        )

        # GGUF-muunnos
        self.tools.register(
            name="muunna_gguf",
            description="Muunna ladattu HuggingFace-malli GGUF-muotoon",
            parameters={
                "model_path": {
                    "description": "Polku mallin kansioon tai mallin HF ID",
                    "required": True,
                },
                "output_type": {
                    "description": "Tulosmuoto: f16 (oletus), f32, bf16",
                    "required": False,
                }
            },
            handler=self._tool_convert_to_gguf,
        )

        # GGUF-kvantisointi
        self.tools.register(
            name="kvantisoi_malli",
            description="Kvantisoi GGUF-malli pienemmäksi (esim. Q4_K_M)",
            parameters={
                "input_file": {
                    "description": "GGUF-tiedoston nimi tai polku",
                    "required": True,
                },
                "quantization": {
                    "description": "Kvantisointityyppi: q4_k_m (oletus), q8_0, q5_k_m, jne.",
                    "required": False,
                }
            },
            handler=self._tool_quantize_model,
        )

        # Ladattujen mallien listaus
        self.tools.register(
            name="listaa_lataukset",
            description="Listaa kaikki ladatut HuggingFace-mallit",
            parameters={},
            handler=self._tool_list_downloads,
        )

        # GGUF-mallien listaus
        self.tools.register(
            name="listaa_gguf",
            description="Listaa kaikki GGUF-mallit ja niiden kvantisointityypit",
            parameters={},
            handler=self._tool_list_gguf,
        )

    # =========================================================================
    # TYÖKALU-HANDLERIT
    # =========================================================================

    def _tool_list_models(self, format: str = None) -> str:
        if not self.library:
            return "Mallikirjasto ei ole kaytettavissa."

        models = self.library.list_models(format_filter=format)
        if not models:
            return "Kirjastossa ei ole malleja."

        filter_text = f" ({format.upper()})" if format else ""
        lines = [
            f"## Mallikirjasto{filter_text}",
            f"Loytyi **{len(models)}** mallia.",
            "",
            "| Malli | Formaatti | Kvantisointi | Koko |",
            "|-------|-----------|--------------|------|"
        ]

        for m in models[:10]:
            size = format_size(m.size_bytes)
            quant = m.quantization or "-"
            name = m.name[:30] + "..." if len(m.name) > 30 else m.name
            lines.append(f"| {name} | {m.format.upper()} | {quant} | {size} |")

        if len(models) > 10:
            lines.append(f"\n*...ja {len(models) - 10} muuta mallia.*")

        return "\n".join(lines)

    def _tool_search_models(self, query: str) -> str:
        if not self.library:
            return "Mallikirjasto ei ole kaytettavissa."

        results = self.library.search_models(query)
        if not results:
            return f"Haulla **{query}** ei loytynyt malleja."

        lines = [
            f"## Hakutulokset: \"{query}\"",
            f"Loytyi **{len(results)}** mallia.",
            "",
            "| Malli | Formaatti | Koko |",
            "|-------|-----------|------|"
        ]

        for m in results[:10]:
            size = format_size(m.size_bytes)
            name = m.name[:35] + "..." if len(m.name) > 35 else m.name
            lines.append(f"| {name} | {m.format.upper()} | {size} |")

        return "\n".join(lines)

    def _tool_library_stats(self) -> str:
        if not self.library:
            return "Mallikirjasto ei ole kaytettavissa."

        stats = self.library.get_stats()

        lines = [
            "## Mallikirjaston tilastot",
            "",
            "### Yhteenveto",
            f"- **Malleja yhteensa:** {stats['total_models']}",
            f"- **Kokonaiskoko:** {stats['total_size_gb']:.1f} GB",
            "",
            "### Formaatit",
            "| Formaatti | Maara |",
            "|-----------|-------|"
        ]

        for fmt, count in stats.get('format_counts', {}).items():
            lines.append(f"| {fmt.upper()} | {count} |")

        lines.append("")
        lines.append("### Lahteet")
        lines.append("| Lahde | Maara |")
        lines.append("|-------|-------|")

        for src, count in stats.get('source_counts', {}).items():
            lines.append(f"| {src} | {count} |")

        return "\n".join(lines)

    def _tool_search_huggingface(self, query: str, limit: int = 5) -> str:
        if not self.downloader:
            return "HuggingFace-lataustyokalu ei ole kaytettavissa."

        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 5

        results = self.downloader.search_models(query, limit=limit)
        if not results:
            return f"Haulla **{query}** ei loytynyt malleja HuggingFacesta."

        lines = [
            f"## HuggingFace-haku: \"{query}\"",
            f"Loytyi **{len(results)}** mallia.",
            "",
            "| Malli | Tyyppi | Lataukset | Tagit |",
            "|-------|--------|-----------|-------|"
        ]

        for r in results:
            downloads = f"{r.downloads:,}" if r.downloads else "0"
            model_name = r.model_id[:35] + "..." if len(r.model_id) > 35 else r.model_id
            task = r.pipeline_tag or "-"
            # Näytä 2-3 tärkeintä tagia
            tags_str = ", ".join(r.tags[:3]) if r.tags else "-"
            if len(tags_str) > 25:
                tags_str = tags_str[:22] + "..."
            lines.append(f"| `{model_name}` | {task} | {downloads} | {tags_str} |")

        lines.append("")
        lines.append("*Kayta `mallin_tiedot` saadaksesi koon ja lisatietoja.*")

        return "\n".join(lines)

    def _tool_model_details(self, model_id: str) -> str:
        if not self.downloader:
            return "HuggingFace-lataustyokalu ei ole kaytettavissa."

        details = self.downloader.get_model_details(model_id)
        if not details:
            return f"Mallia **{model_id}** ei loytynyt."

        # Muotoile koko selkeästi
        size_gb = details.total_size / (1024**3) if details.total_size else 0
        if size_gb >= 1:
            size_str = f"{size_gb:.1f} GB"
        else:
            size_mb = details.total_size / (1024**2) if details.total_size else 0
            size_str = f"{size_mb:.0f} MB"

        lines = [
            f"## Mallin tiedot",
            f"### {details.model_id}",
            "",
            "| Kentta | Arvo |",
            "|--------|------|",
            f"| **Koko** | **{size_str}** |",
            f"| **Tekija** | {details.author} |",
            f"| **Lataukset** | {details.downloads:,} |" if details.downloads else "| **Lataukset** | 0 |",
            f"| **Tykkaykset** | {details.likes} |" if details.likes else "| **Tykkaykset** | 0 |",
            f"| **Tyyppi** | {details.pipeline_tag or '-'} |",
        ]

        if details.tags:
            tags_str = ", ".join([f"`{t}`" for t in details.tags[:6]])
            lines.append(f"| **Tagit** | {tags_str} |")

        # Näytä tiedostotyypit
        safetensors = sum(1 for f in details.files if f['name'].endswith('.safetensors'))
        gguf = sum(1 for f in details.files if f['name'].endswith('.gguf'))
        if safetensors or gguf:
            files_info = []
            if safetensors:
                files_info.append(f"{safetensors} safetensors")
            if gguf:
                files_info.append(f"{gguf} GGUF")
            lines.append(f"| **Tiedostot** | {', '.join(files_info)} |")

        return "\n".join(lines)

    def _tool_calculate_vram(self, parameters_b: float) -> str:
        try:
            params_b = float(parameters_b)
        except (ValueError, TypeError):
            return "Virheellinen parametrien maara."

        lines = [
            f"## VRAM-vaatimukset: {params_b}B malli",
            "",
            "### Arviot eri kvantisoinneilla",
            "",
            "| Kvantisointi | Mallin koko | + 4K konteksti | + 8K konteksti |",
            "|--------------|-------------|----------------|----------------|"
        ]

        calculations = [
            ("F16", 16, "Taysi tarkkuus"),
            ("Q8_0", 8, "8-bittinen"),
            ("Q6_K", 6.5, "6-bit K-quant"),
            ("Q5_K_M", 5.5, "5-bit K-quant"),
            ("**Q4_K_M**", 4.5, "Suositeltu"),
            ("Q4_0", 4.0, "4-bit perus"),
            ("Q3_K_M", 3.5, "3-bit K-quant"),
            ("Q2_K", 2.5, "2-bit (aggressiivinen)"),
        ]

        for name, bits, desc in calculations:
            model_gb = (params_b * 1e9 * bits / 8) / (1024**3)
            ctx_4k = model_gb + 0.5
            ctx_8k = model_gb + 1.0
            lines.append(f"| {name} | {model_gb:.1f} GB | {ctx_4k:.1f} GB | {ctx_8k:.1f} GB |")

        lines.append("")
        lines.append("### Suositus")
        lines.append(f"- **Q4_K_M** tarjoaa parhaan tasapainon koon ja laadun valilla")
        lines.append(f"- Tarvitset vahintaan **{(params_b * 1e9 * 4.5 / 8) / (1024**3):.1f} GB** VRAM:ia")

        return "\n".join(lines)

    def _tool_quantization_types(self) -> str:
        lines = [
            "## Kvantisointityypit",
            "",
            "### Mita kvantisointi on?",
            "Kvantisointi pakkaa mallin painot pienempaan tilaan, vahentaen muistin kayttoa.",
            "",
            "### Tyypit",
            "",
            "| Tyyppi | Bitit | Laatu | Kuvaus |",
            "|--------|-------|-------|--------|",
            "| F16 | 16.0 | Taydellinen | Alkuperainen tarkkuus |",
            "| Q8_0 | 8.0 | Erinomainen | Lahes havioton |",
            "| Q6_K | 6.5 | Erittain hyva | K-quant 6-bit |",
            "| Q5_K_M | 5.5 | Hyva | K-quant 5-bit medium |",
            "| **Q4_K_M** | 4.5 | **Suositeltu** | Paras tasapaino |",
            "| Q4_0 | 4.0 | Kohtalainen | Perus 4-bit |",
            "| Q3_K_M | 3.5 | Heikko | Pieni koko |",
            "| Q2_K | 2.5 | Heikoin | Aggressiivinen pakkaus |",
            "",
            "### Suositus",
            "- **Aloittelijoille:** Q4_K_M",
            "- **Laatu tarkeaa:** Q5_K_M tai Q6_K",
            "- **Rajallinen VRAM:** Q3_K_M",
        ]

        return "\n".join(lines)

    # =========================================================================
    # UUDET TOIMINNALLISET TYÖKALUT (lataus, muunnos, kvantisointi)
    # =========================================================================

    def _tool_download_model(self, model_id: str) -> str:
        """Lataa malli HuggingFacesta."""
        if not self.downloader:
            return "Virhe: HuggingFace-lataustyokalu ei ole kaytettavissa."

        # Tarkista onko jo ladattu
        existing = self.downloader.check_exists(model_id)
        if existing:
            return f"## Malli jo ladattu!\n\nMalli **{model_id}** on jo ladattu kansioon:\n`{existing}`\n\n*Kayta `--force` jos haluat ladata uudelleen.*"

        # Hae mallin tiedot ensin
        details = self.downloader.get_model_details(model_id)
        if not details:
            return f"Virhe: Mallia **{model_id}** ei loytynyt HuggingFacesta."

        size_gb = details.total_size / (1024**3) if details.total_size else 0

        # Lataa malli
        try:
            result = self.downloader.download_model(model_id)
            if result:
                return f"""## Lataus valmis!

Malli **{model_id}** ladattu onnistuneesti!

| Tiedot | |
|--------|------|
| **Sijainti** | `{result}` |
| **Koko** | {size_gb:.1f} GB |

*Voit nyt muuntaa mallin GGUF-muotoon `muunna_gguf` -tyokalulla.*"""
            else:
                return f"Virhe: Mallin **{model_id}** lataus epaonnistui."
        except Exception as e:
            return f"Virhe latauksessa: {e}"

    def _tool_convert_to_gguf(self, model_path: str, output_type: str = "f16") -> str:
        """Muunna HuggingFace-malli GGUF-muotoon."""
        if not self.converter:
            return "Virhe: GGUF-muunnin ei ole kaytettavissa."

        from pathlib import Path

        # Tarkista malli
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            # Kokeile downloads-kansiosta
            if self.downloader:
                folder_name = model_path.replace("/", "_")
                model_path_obj = self.downloader.download_dir / folder_name
                if not model_path_obj.exists():
                    return f"Virhe: Mallia ei loytynyt polusta: {model_path}\n\n*Lataa malli ensin `lataa_malli` -tyokalulla.*"

        # Tarkista llama.cpp
        status = self.converter.check_llama_cpp()
        if not status.get("installed"):
            return "Virhe: llama.cpp ei ole asennettu. Asenna ensin GGUF Converter -valikosta."

        try:
            result = self.converter.convert_to_gguf(
                model_path=str(model_path_obj),
                output_type=output_type,
            )

            if result.get("success"):
                return f"""## GGUF-muunnos valmis!

Malli muunnettu onnistuneesti!

| Tiedot | |
|--------|------|
| **Tiedosto** | `{result['output_path']}` |
| **Koko** | {result['file_size_gb']:.1f} GB |
| **Tyyppi** | {output_type.upper()} |

*Voit nyt kvantisoida mallin pienemmaksi `kvantisoi_malli` -tyokalulla.*"""
            else:
                return f"Virhe muunnoksessa: {result.get('error', 'Tuntematon virhe')}"

        except Exception as e:
            return f"Virhe GGUF-muunnoksessa: {e}"

    def _tool_quantize_model(self, input_file: str, quantization: str = "q4_k_m") -> str:
        """Kvantisoi GGUF-malli pienemmäksi."""
        if not self.converter:
            return "Virhe: GGUF-muunnin ei ole kaytettavissa."

        from pathlib import Path
        from ..core.paths import get_gguf_dir

        # Etsi tiedosto - ensin suoraan, sitten GGUF-kansiosta
        input_path = Path(input_file)
        if not input_path.exists():
            # Kokeile GGUF-kansiosta
            gguf_dir = get_gguf_dir()
            possible_path = gguf_dir / input_file
            if possible_path.exists():
                input_path = possible_path
            else:
                # Etsi .gguf-paatteella
                for f in gguf_dir.glob("*.gguf"):
                    if input_file.lower() in f.name.lower():
                        input_path = f
                        break

        if not input_path.exists():
            return f"Virhe: GGUF-tiedostoa ei loytynyt: {input_file}\n\n*Tarkista tiedostonimi tai muunna malli ensin `muunna_gguf` -tyokalulla.*"

        # Tarkista llama-quantize
        quantize_binary = self.converter._find_quantize_binary()
        if not quantize_binary:
            return "Virhe: llama-quantize ei ole kaytettavissa. Lataa llama.cpp binaarit."

        try:
            result = self.converter.quantize_gguf(
                input_path=str(input_path),
                quantization=quantization,
            )

            if result.get("success"):
                return f"""## Kvantisointi valmis!

GGUF-malli kvantisoitu onnistuneesti!

| Tiedot | |
|--------|------|
| **Tiedosto** | `{result['output_path']}` |
| **Koko** | {result['file_size_gb']:.1f} GB |
| **Kvantisointi** | {quantization.upper()} |

*Malli on nyt valmis kaytettavaksi!*"""
            else:
                return f"Virhe kvantisoinnissa: {result.get('error', 'Tuntematon virhe')}"

        except Exception as e:
            return f"Virhe kvantisoinnissa: {e}"

    def _tool_list_downloads(self) -> str:
        """Listaa ladatut HuggingFace-mallit."""
        if not self.downloader:
            return "HuggingFace-lataustyokalu ei ole kaytettavissa."

        downloaded = self.downloader.list_downloaded()
        if not downloaded:
            return "## Ei ladattuja malleja\n\nLataa malli `lataa_malli` -tyokalulla tai Model Download -valikosta."

        lines = [
            "## Ladatut HuggingFace-mallit",
            f"Loytyi **{len(downloaded)}** mallia.",
            "",
            "| Malli | Koko |",
            "|-------|------|"
        ]

        for m in downloaded:
            size = format_size(m['size'])
            model_id = m['model_id'][:40] + "..." if len(m['model_id']) > 40 else m['model_id']
            lines.append(f"| `{model_id}` | {size} |")

        lines.append("")
        lines.append("*Naita malleja voi muuntaa GGUF-muotoon `muunna_gguf` -tyokalulla.*")

        return "\n".join(lines)

    def _tool_list_gguf(self) -> str:
        """Listaa GGUF-mallit."""
        from ..core.paths import get_gguf_dir

        gguf_dir = get_gguf_dir()
        gguf_files = sorted(gguf_dir.glob("*.gguf"), key=lambda f: f.stat().st_mtime, reverse=True)

        if not gguf_files:
            return "## Ei GGUF-malleja\n\nMuunna malli GGUF-muotoon `muunna_gguf` -tyokalulla."

        lines = [
            "## GGUF-mallit",
            f"Loytyi **{len(gguf_files)}** mallia.",
            "",
            "| Tiedosto | Koko | Kvantisointi |",
            "|----------|------|--------------|"
        ]

        # Tunnista kvantisointi
        def detect_quant(name: str) -> str:
            name_lower = name.lower()
            quants = ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0",
                      "q3_k_m", "q3_k_s", "q2_k", "f16", "f32"]
            for q in quants:
                if q in name_lower:
                    return q.upper()
            return "-"

        for f in gguf_files[:15]:
            size = format_size(f.stat().st_size)
            quant = detect_quant(f.name)
            name = f.stem[:35] + "..." if len(f.stem) > 35 else f.stem
            lines.append(f"| `{name}` | {size} | {quant} |")

        if len(gguf_files) > 15:
            lines.append(f"\n*...ja {len(gguf_files) - 15} muuta mallia.*")

        return "\n".join(lines)

    # =========================================================================
    # TWO-PASS CHAT-LOGIIKKA
    # =========================================================================

    def _get_router_prompt(self) -> str:
        """Luo router-prompt (PASS 1) työkalulistalla."""
        # Luo yksinkertainen JSON-lista työkaluista
        tools_list = []
        for name, tool in self.tools.tools.items():
            params_str = ", ".join(tool["parameters"].keys()) if tool["parameters"] else "ei parametreja"
            tools_list.append(f"- {name}: {tool['description']} ({params_str})")

        tools_json = "\n".join(tools_list)
        return self.ROUTER_PROMPT.format(tools_json=tools_json)

    def _get_response_prompt(self, context: str = "") -> str:
        """Luo response-prompt (PASS 2) kontekstilla."""
        if context:
            # Käytä XML-tageja selkeään erotteluun (best practice pienille malleille)
            context_block = f"""
<tyokalun_tulos>
{context}
</tyokalun_tulos>

Kayta yllaolevan tagin sisaltoa vastauksessasi. Muotoile tieto selkeasti kayttajalle."""
        else:
            context_block = ""
        return self.RESPONSE_PROMPT.format(context=context_block)

    def _clean_response(self, text: str) -> str:
        """
        Siivoa mallin vastaus Chain of Thought -vuodoista.

        Poistaa:
        - <think>...</think> tagit (DeepSeek R1 tyyli)
        - Englanninkieliset ajatusvirrat ("The user wants...", "Let me think...")
        - Muut sisäiset monologit
        """
        if not text:
            return text

        original_text = text

        # 1. Poista <think> tagit (DeepSeek R1 tyyli)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # 2. Poista <reasoning>...</reasoning> tagit
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # 3. Poista englanninkieliset ajatusvirrat alusta
        # Nämä ovat tyypillisiä "reasoning" mallien vuotoja
        cot_patterns = [
            r'^(?:The user wants?|Let me think|I should|I need to|First,? I\'ll|Okay,? so).*?(?=\n\n|\n#|$)',
            r'^(?:Looking at|Based on|According to the).*?(?=\n\n|\n#|$)',
            r'^(?:I\'ll|I will|Let\'s|We need to).*?(?=\n\n|\n#|$)',
        ]

        for pattern in cot_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # 4. Poista tyhjät rivit alusta ja lopusta
        text = text.strip()

        # 5. Jos koko teksti hävisi, palauta alkuperäinen (jotain meni pieleen)
        if not text and original_text:
            return original_text.strip()

        return text

    def _get_system_prompt(self) -> str:
        """Luo system-prompt (yhteensopivuus Ollamalle)."""
        return self._get_response_prompt("")

    def _run_router(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        PASS 1: Router - päättää tarvitaanko työkalua.

        Args:
            user_input: Käyttäjän viesti

        Returns:
            {"tool": "name", "params": {...}} tai {"tool": null} tai None virhetilanteessa
        """
        if not self.backend or not hasattr(self.backend, 'generate'):
            return None

        router_messages = [
            {"role": "system", "content": self._get_router_prompt()},
            {"role": "user", "content": user_input},
        ]

        console.print("[dim]🔍 Analysoidaan...[/dim]", end="\r")

        try:
            # Käytä temperature=0 ja format="json" tarkkaan päätöksentekoon
            response = self.backend.generate(
                router_messages,
                temperature=0.0,
                max_tokens=256,
                top_p=1.0,
                repeat_penalty=1.0,
                format="json",  # Pakottaa Ollaman tuottamaan validia JSONia
            )

            console.print(" " * 30, end="\r")  # Tyhjennä rivi

            # Parsii JSON
            return self._parse_router_response(response)

        except Exception as e:
            console.print(f"[dim]Router error: {e}[/dim]")
            return None

    def _parse_router_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parsii router-vastauksen JSON."""
        response = response.strip()

        # Yritä parsia suoraan
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Etsi JSON vastauksesta
        json_patterns = [
            r'\{[^{}]*"tool"[^{}]*\}',  # Yksinkertainen
            r'\{"tool":\s*(?:"[^"]*"|null),?\s*(?:"params":\s*\{[^}]*\})?\}',  # Tarkempi
        ]

        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        return None

    def _run_response(self, user_input: str, context: str = "") -> str:
        """
        PASS 2: Response - generoi luova vastaus.

        Args:
            user_input: Käyttäjän viesti
            context: Työkalun tulos (jos käytettiin)

        Returns:
            Luova vastaus käyttäjälle
        """
        if not self.backend or not hasattr(self.backend, 'generate'):
            return "Virhe: Backend ei tue generate-metodia."

        response_messages = [
            {"role": "system", "content": self._get_response_prompt(context)},
            {"role": "user", "content": user_input},
        ]

        try:
            # Käytä temperature=0.7 luovaan vastaukseen
            response = self.backend.generate(
                response_messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            # Siivoa Chain of Thought -vuodot
            return self._clean_response(response)
        except Exception as e:
            return f"Virhe: {e}"

    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Parsii työkalukutsun vastauksesta."""
        # Etsi ```tool ... ``` -lohko
        tool_pattern = r'```tool\s*\n?(.*?)\n?```'
        match = re.search(tool_pattern, content, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Yritä löytää JSON suoraan
        json_pattern = r'\{"tool":\s*"[^"]+",\s*"params":\s*\{[^}]*\}\}'
        match = re.search(json_pattern, content)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _print_user_message(self, content: str):
        """Tulosta käyttäjän viesti kauniisti."""
        console.print(Panel(
            Text(content, style="white"),
            title="[bold blue]Sinä[/bold blue]",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            box=box.ROUNDED,
        ))

    def _print_assistant_message(self, content: str, streaming: bool = False):
        """Tulosta assistentin viesti Markdown-muotoiltuna."""
        if streaming:
            # Streaming-tilassa näytä paneeli ilman sisältöä ensin
            return

        # Siivoa CoT-vuodot ja poista tool-kutsut näytöstä
        display_content = self._clean_response(content)
        display_content = re.sub(r'```tool\s*\n?.*?\n?```', '', display_content, flags=re.DOTALL).strip()

        if display_content:
            console.print(Panel(
                Markdown(display_content),
                title="[bold yellow]Tool Master[/bold yellow]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
                box=box.ROUNDED,
            ))

    def _print_streaming_response(self, content: str):
        """Tulosta streaming-vastaus kauniisti kun valmis."""
        # Siivoa CoT-vuodot ja poista tool-kutsut
        display_content = self._clean_response(content)
        display_content = re.sub(r'```tool\s*\n?.*?\n?```', '', display_content, flags=re.DOTALL).strip()

        if display_content:
            console.print(Panel(
                Markdown(display_content),
                title="[bold yellow]Tool Master[/bold yellow]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
                box=box.ROUNDED,
            ))

    def chat(self, user_input: str) -> str:
        """
        Lähetä viesti ja käsittele vastaus.

        Two-Pass arkkitehtuuri (kaikki backendit):
        - PASS 1: Router (temp=0) - päättää tarvitaanko työkalua
        - PASS 2: Response (temp=0.7) - generoi luova vastaus
        """
        if not self.backend:
            return "Virhe: Backendiä ei ole valittu."

        # Näytä käyttäjän viesti
        self._print_user_message(user_input)

        # Lisää käyttäjän viesti historiaan
        self.messages.append(ChatMessage(role="user", content=user_input))

        # ==== TWO-PASS (kaikille backendeille) ====
        return self._chat_two_pass(user_input)

    def _chat_two_pass(self, user_input: str) -> str:
        """Two-Pass chat - toimii kaikilla backendeillä."""

        # ========================================
        # PASS 1: ROUTER (temperature=0)
        # Päättää tarvitaanko työkalua
        # ========================================

        router_result = self._run_router(user_input)
        tool_context = ""

        if router_result and router_result.get("tool"):
            tool_name = router_result["tool"]
            tool_params = router_result.get("params", {})

            # Näytä tool-kutsu
            console.print(Panel(
                f"[cyan]🔧 Työkalu:[/cyan] {tool_name}\n[dim]Parametrit: {json.dumps(tool_params, ensure_ascii=False)}[/dim]",
                title="[bold yellow]Router päätös[/bold yellow]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
                box=box.ROUNDED,
            ))

            # Suorita työkalu
            result = self.tools.execute(tool_name, tool_params)

            if result.get("error"):
                tool_context = f"Työkalun virhe: {result['error']}"
            else:
                tool_context = result.get("result", "Työkalu suoritettu.")

            # Näytä työkalun tulos
            console.print(Panel(
                Markdown(tool_context) if not tool_context.startswith("Virhe") else Text(tool_context, style="red"),
                title=f"[bold]📊 {tool_name}[/bold]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
                box=box.ROUNDED,
            ))

        # ========================================
        # PASS 2: RESPONSE (temperature=0.7)
        # Generoi luova vastaus käyttäjälle
        # ========================================

        console.print("[dim]✨ Generoidaan vastaus...[/dim]", end="\r")

        response = self._run_response(user_input, tool_context)

        console.print(" " * 40, end="\r")  # Tyhjennä rivi

        # Näytä vastaus kauniisti
        self._print_streaming_response(response)

        # Tallenna historiaan
        self.messages.append(ChatMessage(role="assistant", content=response))

        return response

    # =========================================================================
    # KÄYTTÖLIITTYMÄ
    # =========================================================================

    def select_backend(self) -> bool:
        """Anna käyttäjän valita backend."""
        llama_backend = LlamaCppBackend()
        ollama_backend = OllamaBackend()

        choices = []

        # Built-in (aina näytetään, mutta kerrotaan jos ei asennettu)
        if llama_backend.is_available():
            choices.append(questionary.Choice(
                title="🔷 Built-in (llama.cpp)   Käytä GGUF-mallia suoraan (suositeltu)",
                value="builtin"
            ))
        else:
            choices.append(questionary.Choice(
                title="⚪ Built-in (llama.cpp)   (ei asennettu - pip install llama-cpp-python)",
                value="builtin_not_installed"
            ))

        # Ollama
        if ollama_backend.is_available():
            choices.append(questionary.Choice(
                title="🟢 Ollama                 Käytä Ollama-palvelua",
                value="ollama"
            ))
        else:
            choices.append(questionary.Choice(
                title="⚪ Ollama                 (ei käynnissä)",
                value="ollama_not_running"
            ))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="↩️ Back", value="back"))

        selected = questionary.select(
            "Valitse chat-backend:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">",
        ).ask()

        if selected == "builtin":
            self.backend = llama_backend
            return self._select_gguf_model()
        elif selected == "builtin_not_installed":
            print_warning("llama-cpp-python ei ole asennettu.")
            console.print("\n[cyan]Asenna komennolla:[/cyan]")
            console.print("[dim]  pip install llama-cpp-python[/dim]")
            console.print("\n[dim]GPU-tuki (CUDA):[/dim]")
            console.print("[dim]  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return False
        elif selected == "ollama":
            self.backend = ollama_backend
            return self._select_ollama_model()
        elif selected == "ollama_not_running":
            print_warning("Ollama ei ole käynnissä.")
            console.print("\n[cyan]Käynnistä Ollama:[/cyan]")
            console.print("[dim]  1. Asenna: https://ollama.ai[/dim]")
            console.print("[dim]  2. Käynnistä: ollama serve[/dim]")
            console.print("[dim]  3. Lataa malli: ollama pull llama3.2[/dim]")
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return False

        return False

    def _select_gguf_model(self) -> bool:
        """Valitse GGUF-malli GGUF-kansiosta."""
        from ..core.paths import get_gguf_dir

        # Skannaa GGUF-kansio suoraan
        gguf_dir = get_gguf_dir()
        gguf_files = sorted(gguf_dir.glob("*.gguf"), key=lambda f: f.stat().st_mtime, reverse=True)

        if not gguf_files:
            # Ei GGUF-malleja kansiossa
            console.print(Panel(
                "[yellow]GGUF-kansiossa ei ole malleja![/yellow]\n\n"
                f"[dim]Kansio: {gguf_dir}[/dim]\n\n"
                "Hanki GGUF-malli:\n"
                "1. [cyan]Model Download[/cyan] - Lataa GGUF-malli HuggingFacesta\n"
                "2. [cyan]GGUF Converter[/cyan] - Muunna HF-malli GGUF-muotoon\n"
                "3. [cyan]Quantize Tool[/cyan] - Kvantisoi olemassa oleva GGUF",
                title="[bold]Ei malleja[/bold]",
                border_style="yellow",
                box=box.ROUNDED,
            ))
            questionary.press_any_key_to_continue(style=custom_style).ask()
            return False

        # Tunnista kvantisointi tiedostonimestä
        def detect_quant(filename: str) -> str:
            name_lower = filename.lower()
            quants = ["q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q4_1",
                      "q3_k_l", "q3_k_m", "q3_k_s", "q2_k", "iq4_xs", "iq3_m", "iq3_s", "iq2_xs",
                      "f16", "f32", "bf16"]
            for q in quants:
                if q in name_lower or q.replace("_", "-") in name_lower:
                    return q.upper()
            return "F16/F32"

        # Näytä mallit valittavaksi
        console.print(Panel(
            f"[green]Loytyi {len(gguf_files)} GGUF-mallia[/green]",
            border_style="green",
            box=box.ROUNDED,
        ))

        choices = []
        for f in gguf_files:
            size = format_size(f.stat().st_size)
            quant = detect_quant(f.name)
            display_name = f.stem[:35] if len(f.stem) <= 35 else f.stem[:32] + "..."
            choices.append(questionary.Choice(
                title=f"🔷 {display_name:<35} [{quant:<8}] {size:>10}",
                value=str(f)
            ))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="↩️  Takaisin", value=None))

        selected = questionary.select(
            "Valitse malli:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">",
        ).ask()

        if not selected:
            return False

        # Lataa malli järkevillä oletusasetuksilla
        return self.backend.load_model(
            selected,
            n_ctx=4096,      # Hyvä oletus useimmille malleille
            n_gpu_layers=0,  # CPU-only oletuksena (turvallinen)
        )

    def _select_ollama_model(self) -> bool:
        """Valitse Ollama-malli."""
        models = self.backend.get_models()

        if not models:
            print_error("Ei malleja saatavilla. Lataa malli: ollama pull llama3.2")
            return False

        choices = []
        for m in models:
            name = m.get("name", "tuntematon")
            size = m.get("size", 0)
            size_str = format_size(size) if size else ""
            choices.append(questionary.Choice(
                title=f"{name:<30} {size_str:>12}",
                value=name,
            ))

        choices.append(questionary.Separator())
        choices.append(questionary.Choice(title="↩️ Back", value=None))

        selected = questionary.select(
            "Valitse Ollama-malli:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">",
        ).ask()

        if selected:
            return self.backend.load_model(selected)
        return False

    def _print_chat_header(self, model_name: str = ""):
        """Tulosta kaunis chat-otsikko."""
        header = """
[bold yellow]╔══════════════════════════════════════════════════════════════════╗[/bold yellow]
[bold yellow]║[/bold yellow]     [bold white]🔧 TOOL MASTER - AI Toolboxin mestariavustaja 🔧[/bold white]       [bold yellow]║[/bold yellow]
[bold yellow]╠══════════════════════════════════════════════════════════════════╣[/bold yellow]
[bold yellow]║[/bold yellow]  [green]●[/green] Malli ladattu ja valmis keskusteluun                        [bold yellow]║[/bold yellow]
[bold yellow]║[/bold yellow]  [dim]Komennot: 'lopeta' | 'tyhjenna' | 'ohje'[/dim]                     [bold yellow]║[/bold yellow]
[bold yellow]╚══════════════════════════════════════════════════════════════════╝[/bold yellow]
"""
        console.print(header)

    def _print_help(self):
        """Näytä ohje."""
        help_text = """
## Komennot

| Komento | Kuvaus |
|---------|--------|
| `lopeta` | Poistu chatista |
| `tyhjenna` | Tyhjenna keskusteluhistoria |
| `ohje` | Nayta tama ohje |

## Tool Master osaa auttaa sinua

- **Mallien etsiminen** - Etsi malleja HuggingFacesta
- **Kirjaston hallinta** - Listaa ja selaa paikalliset mallit
- **VRAM-laskenta** - Laske muistivaatimukset
- **Kvantisointi-info** - Selita eri kvantisointityypit

## Esimerkkeja

> "Etsi 7B kokoisia malleja HuggingFacesta"
> "Mita malleja minulla on?"
> "Paljonko muistia 13B malli tarvitsee?"
> "Mika on Q4_K_M kvantisointi?"
"""
        console.print(Panel(
            Markdown(help_text),
            title="[bold yellow]Tool Master - Ohje[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
            box=box.ROUNDED,
        ))

    def run(self):
        """Käynnistä interaktiivinen chat."""
        print_mini_banner("Tool Master")

        # Valitse backend
        if not self.select_backend():
            return

        backend_name = self.backend.get_name()

        # Näytä kaunis otsikko
        self._print_chat_header(backend_name)
        console.print(f"  [dim]Backend: {backend_name}[/dim]\n")

        # Chat-silmukka
        try:
            while True:
                try:
                    # Käytä mukavampaa promptia
                    console.print("[bold blue]┌─ Sinä[/bold blue]")
                    user_input = questionary.text(
                        "│",
                        style=custom_style,
                        qmark="",
                    ).ask()

                    if user_input is None or user_input.lower() in ['lopeta', 'exit', 'quit', 'q']:
                        break

                    if user_input.lower() in ['tyhjennä', 'clear', 'uusi']:
                        self.messages.clear()
                        console.print(Panel(
                            "[green]Keskustelu tyhjennetty. Voit aloittaa alusta![/green]",
                            border_style="green",
                            box=box.ROUNDED,
                        ))
                        continue

                    if user_input.lower() in ['ohje', 'help', '?']:
                        self._print_help()
                        continue

                    if not user_input.strip():
                        continue

                    console.print()
                    self.chat(user_input)
                    console.print()

                except KeyboardInterrupt:
                    break

        finally:
            # Vapauta malli
            if self.backend:
                console.print("\n[dim]Vapautetaan malli muistista...[/dim]")
                self.backend.unload_model()

            console.print(Panel(
                "[cyan]Kiitos keskustelusta! Nähdään taas.[/cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))


def ai_chat_menu(library=None, downloader=None, converter=None):
    """Käynnistä AI-chat-valikosta."""
    chat = AIChat(library=library, downloader=downloader, converter=converter)
    chat.run()


if __name__ == "__main__":
    ai_chat_menu()
