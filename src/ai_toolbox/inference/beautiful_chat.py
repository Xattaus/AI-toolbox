"""
AI TOOLBOX - Beautiful Chat Interface
======================================

Kaunis, interaktiivinen chat-kayttoliittyma Ollama-malleille.
Tukee streamingja ja <think>-tagien kasittelya reasoning-malleille.

Features:
- Real-time streaming with Rich Live
- <think> tag parsing for reasoning models
- Beautiful Rich panels with color coding
- Conversation history management
- Spinner animations during thinking
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator

import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich import box

console = Console()

# Thinking animation frames
THINKING_FRAMES = ["", "", "", "", "", "", "", "", "", ""]


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class ChatChunk:
    """Yksittainen streaming-pala."""
    type: str  # "token", "thinking_start", "thinking_content", "thinking_end", "done", "error"
    content: str = ""


@dataclass
class ChatMessage:
    """Keskusteluviesti."""
    role: str  # "user", "assistant", "system"
    content: str
    thinking: str = ""  # <think> sisalto (reasoning-malleille)


# =============================================================================
# OLLAMA STREAM CLIENT
# =============================================================================

class OllamaStreamClient:
    """Ollama streaming client joka parsii <think> tagit lennossa."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        """Tarkista onko Ollama kaytettavissa."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[ChatChunk, None, None]:
        """
        Streamaa chat-vastaus ja parsii <think> tagit lennossa.

        Args:
            model: Mallin nimi
            messages: Viestilista [{"role": "user", "content": "..."}]
            options: Ollama-asetukset (temperature, top_p, jne.)

        Yields:
            ChatChunk objekteja eri tyypeilla
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            )

            if response.status_code != 200:
                yield ChatChunk(type="error", content=f"HTTP {response.status_code}")
                return

            # State machine for parsing <think> tags
            in_think = False
            buffer = ""

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    # Flush remaining buffer
                    if buffer:
                        if in_think:
                            yield ChatChunk(type="thinking_content", content=buffer)
                        else:
                            yield ChatChunk(type="token", content=buffer)
                    yield ChatChunk(type="done")
                    break

                message = data.get("message", {})
                content = message.get("content", "")

                if not content:
                    continue

                # Add to buffer and process
                buffer += content

                # Process buffer for <think> tags
                while True:
                    if not in_think:
                        # Look for <think> opening tag
                        think_start = buffer.find("<think>")
                        if think_start != -1:
                            # Yield content before <think>
                            if think_start > 0:
                                yield ChatChunk(type="token", content=buffer[:think_start])
                            yield ChatChunk(type="thinking_start")
                            buffer = buffer[think_start + 7:]  # len("<think>") = 7
                            in_think = True
                        else:
                            # Check if we might be at a partial tag
                            # Look for '<' at end of buffer that could be start of <think>
                            partial_idx = -1
                            for i in range(1, min(8, len(buffer) + 1)):  # Check last 7 chars
                                if buffer[-i:].startswith("<"):
                                    if "<think>".startswith(buffer[-i:]):
                                        partial_idx = len(buffer) - i
                                        break

                            if partial_idx > 0:
                                # Yield content before potential partial tag
                                yield ChatChunk(type="token", content=buffer[:partial_idx])
                                buffer = buffer[partial_idx:]
                            elif partial_idx == 0:
                                # Buffer starts with potential partial tag, wait for more
                                pass
                            else:
                                # No partial tag, yield everything (only if non-empty)
                                if buffer:
                                    yield ChatChunk(type="token", content=buffer)
                                buffer = ""
                            break
                    else:
                        # Inside <think>, look for </think> closing tag
                        think_end = buffer.find("</think>")
                        if think_end != -1:
                            # Yield thinking content before </think>
                            if think_end > 0:
                                yield ChatChunk(type="thinking_content", content=buffer[:think_end])
                            yield ChatChunk(type="thinking_end")
                            buffer = buffer[think_end + 8:]  # len("</think>") = 8
                            in_think = False
                        else:
                            # Check for partial </think> tag
                            partial_idx = -1
                            for i in range(1, min(9, len(buffer) + 1)):
                                if buffer[-i:].startswith("<"):
                                    if "</think>".startswith(buffer[-i:]):
                                        partial_idx = len(buffer) - i
                                        break

                            if partial_idx > 0:
                                yield ChatChunk(type="thinking_content", content=buffer[:partial_idx])
                                buffer = buffer[partial_idx:]
                            elif partial_idx == 0:
                                pass  # Wait for more
                            else:
                                # Yield thinking content (only if non-empty)
                                if buffer:
                                    yield ChatChunk(type="thinking_content", content=buffer)
                                buffer = ""
                            break

        except requests.exceptions.Timeout:
            yield ChatChunk(type="error", content="Timeout - vastaus kesti liian kauan")
        except requests.exceptions.ConnectionError:
            yield ChatChunk(type="error", content="Yhteysvirhe - Ollama ei vastaa")
        except Exception as e:
            yield ChatChunk(type="error", content=str(e))


# =============================================================================
# CHAT RENDERER
# =============================================================================

class ChatRenderer:
    """Renderoi chat-viestit kauniisti Rich-kirjastolla."""

    def __init__(self, model_name: str, model_size: str = ""):
        self.model_name = model_name
        self.model_size = model_size
        self.show_thoughts = True  # Näytä ajatukset oletuksena

    def render_header(self, thinking_prefill: bool = False):
        """Renderoi chat-otsikko."""
        # Build title with model info
        title_parts = [f"[bold bright_blue]OLLAMA CHAT[/bold bright_blue]"]
        if self.model_name:
            title_parts.append(f"[dim]|[/dim] [cyan]{self.model_name}[/cyan]")
        if self.model_size:
            title_parts.append(f"[dim]{self.model_size}[/dim]")
        if thinking_prefill:
            title_parts.append(f"[yellow]<think>[/yellow]")

        title = "  ".join(title_parts)

        console.print(Panel(
            f"{title}\n[dim]Komennot: lopeta, tyhjenna, nayta/piilota, ohje[/dim]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        console.print()

    def render_user_message(self, content: str):
        """Renderoi kayttajan viesti."""
        console.print(Panel(
            Text(content, style="white"),
            title="[bold blue]Sina[/bold blue]",
            title_align="left",
            border_style="blue",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    def create_thinking_panel(self, content: str, spinner_frame: int = 0) -> Panel:
        """Luo ajattelu-paneeli (ei tulosta, palauttaa Panelin)."""
        # Animation frame
        frame = THINKING_FRAMES[spinner_frame % len(THINKING_FRAMES)]

        # Truncate thinking content if too long
        display_content = content
        if len(content) > 500:
            # Show last 400 chars
            display_content = "..." + content[-400:]

        # Format thinking content (dim and wrapped)
        lines = display_content.split('\n')
        formatted_lines = []
        for line in lines[-8:]:  # Show last 8 lines
            if line.strip():
                formatted_lines.append(f"[dim]{line[:80]}{'...' if len(line) > 80 else ''}[/dim]")

        thinking_text = f"{frame} [yellow]Analysoidaan...[/yellow]\n\n"
        thinking_text += "\n".join(formatted_lines) if formatted_lines else "[dim]...[/dim]"

        return Panel(
            thinking_text,
            title="[bold yellow]Ajattelee...[/bold yellow]",
            title_align="left",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def create_streaming_panel(self, content: str, thinking: str = ""):
        """Luo streaming-vastaus paneeli (ei tulosta, palauttaa Panelin)."""
        # During streaming, show ONLY response panel (thinking shown at end)
        # This prevents layout issues with Rich Live
        try:
            rendered = Markdown(content) if content.strip() else Text("...")
        except Exception:
            rendered = Text(content)

        # Show indicator if there was thinking
        title = f"[bold green]{self.model_name}[/bold green]"
        if thinking:
            title += " [dim yellow](ajatteli)[/dim yellow]"

        return Panel(
            rendered,
            title=title,
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )

    def render_complete_response(self, content: str, thinking: str = ""):
        """Renderoi valmis vastaus."""
        # Show thinking panel if there was thinking content
        if thinking and self.show_thoughts:
            # Show FULL thinking content (not just summary)
            thinking_text = thinking.strip()

            # Format as dim text for readability
            console.print(Panel(
                Text(thinking_text, style="dim"),
                title="[bold yellow]Ajattelu[/bold yellow]",
                title_align="left",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            ))
        elif thinking and not self.show_thoughts:
            # Just show that there was thinking (collapsed)
            thinking_lines = thinking.strip().split('\n')
            console.print(Panel(
                f"[dim italic]({len(thinking_lines)} rivia ajattelua - 'nayta' nayttaa)[/dim]",
                title="[dim yellow]Ajattelu[/dim yellow]",
                title_align="left",
                border_style="dim yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            ))

        # Main response
        try:
            rendered = Markdown(content) if content.strip() else Text("[dim](tyhja vastaus)[/dim]")
        except Exception:
            rendered = Text(content)

        console.print(Panel(
            rendered,
            title=f"[bold green]{self.model_name}[/bold green]",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    def render_error(self, error: str):
        """Renderoi virheviesti."""
        console.print(Panel(
            f"[red]{error}[/red]",
            title="[bold red]Virhe[/bold red]",
            title_align="left",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        ))


# =============================================================================
# CONVERSATION MANAGER
# =============================================================================

class ConversationManager:
    """Hallitsee keskusteluhistoriaa."""

    def __init__(self, system_prompt: str = ""):
        self.messages: List[ChatMessage] = []
        self.system_prompt = system_prompt

    def set_system_prompt(self, prompt: str):
        """Aseta jarjestelma-prompt."""
        self.system_prompt = prompt

    def add_user(self, content: str):
        """Lisaa kayttajan viesti."""
        if content and content.strip():
            self.messages.append(ChatMessage(role="user", content=content.strip()))

    def add_assistant(self, content: str, thinking: str = ""):
        """Lisaa assistentin viesti."""
        if content and content.strip():
            self.messages.append(ChatMessage(
                role="assistant",
                content=content.strip(),
                thinking=thinking.strip() if thinking else ""
            ))

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Palauta viestit Ollama API:n muodossa."""
        api_messages = []

        # Add system prompt first (only if explicitly set and non-empty)
        # NOTE: Ollama already uses system prompt from Modelfile,
        # so only add if we want to override it
        if self.system_prompt and self.system_prompt.strip():
            api_messages.append({
                "role": "system",
                "content": self.system_prompt.strip()
            })

        # Add conversation messages (skip empty content)
        for msg in self.messages:
            if msg.content and msg.content.strip():
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        return api_messages

    def clear(self):
        """Tyhjenna keskusteluhistoria."""
        self.messages.clear()

    def export_history(self) -> List[Dict[str, Any]]:
        """Vie keskusteluhistoria."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "thinking": msg.thinking,
            }
            for msg in self.messages
        ]


# =============================================================================
# BEAUTIFUL CHAT - MAIN CLASS
# =============================================================================

class BeautifulChat:
    """Kaunis interaktiivinen chat Ollama-malleille."""

    # Model name patterns that support <think> prefill mode
    # Only include models known to properly support the <think>...</think> format
    # Note: poro-r1 does NOT support prefill - it generates <think> tags on its own
    THINKING_MODEL_PATTERNS = [
        "deepseek-r1",      # DeepSeek R1 distill models
        "deepseek-reasoner", # DeepSeek reasoning models
    ]

    @classmethod
    def _is_thinking_model(cls, model_name: str) -> bool:
        """Tarkista onko malli reasoning/thinking-malli."""
        name_lower = model_name.lower()
        return any(pattern in name_lower for pattern in cls.THINKING_MODEL_PATTERNS)

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        model_size: str = "",
        thinking_prefill: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.model_size = model_size
        self.client = OllamaStreamClient(base_url)
        self.conversation = ConversationManager()
        self.renderer = ChatRenderer(model_name, model_size)

        # Thinking prefill mode - adds <think> to force reasoning
        # Auto-detect if not explicitly set (None = auto)
        if thinking_prefill is None:
            self.thinking_prefill = self._is_thinking_model(model_name)
        else:
            self.thinking_prefill = thinking_prefill

        # Chat options - optimized for thinking models
        self.options = {
            "temperature": 0.6 if self.thinking_prefill else 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        }

    def set_thinking_prefill(self, enabled: bool):
        """Aseta thinking prefill paalle/pois."""
        self.thinking_prefill = enabled
        # Adjust temperature for thinking mode
        if enabled:
            self.options["temperature"] = 0.6
        else:
            self.options["temperature"] = 0.7

    def set_system_prompt(self, prompt: str):
        """Aseta jarjestelma-prompt."""
        self.conversation.set_system_prompt(prompt)

    def set_options(self, **kwargs):
        """Aseta chat-asetukset."""
        self.options.update(kwargs)

    def chat(self, user_input: str, show_user_message: bool = False) -> str:
        """
        Laheta viesti ja nayta streaming-vastaus.

        Args:
            user_input: Kayttajan viesti
            show_user_message: Nayta kayttajan viesti paneelissa (oletus: False, koska input() nayttaa sen jo)

        Returns:
            Assistentin vastaus
        """
        # Add user message to history
        self.conversation.add_user(user_input)

        # Render user message only if requested (avoid duplicate display)
        if show_user_message:
            self.renderer.render_user_message(user_input)

        # Get messages for API
        api_messages = self.conversation.get_messages_for_api()

        # Add prefill for thinking models - forces model to start with <think>
        # This makes the model generate reasoning before answering
        prefill_used = False
        if self.thinking_prefill:
            api_messages.append({
                "role": "assistant",
                "content": "<think>"
            })
            prefill_used = True

        # Stream response with live display
        response_content = ""
        thinking_content = ""
        spinner_frame = 0

        # When prefill is used, we're already inside <think>
        in_thinking_mode = prefill_used

        # Buffer for detecting </think> across chunk boundaries
        pending_buffer = ""

        try:
            with Live(console=console, refresh_per_second=10, transient=True) as live:
                for chunk in self.client.stream_chat(
                    self.model_name,
                    api_messages,
                    self.options,
                ):
                    if chunk.type == "thinking_start":
                        thinking_content = ""
                        in_thinking_mode = True

                    elif chunk.type == "thinking_content":
                        thinking_content += chunk.content
                        spinner_frame += 1
                        panel = self.renderer.create_thinking_panel(
                            thinking_content, spinner_frame
                        )
                        live.update(panel)

                    elif chunk.type == "thinking_end":
                        in_thinking_mode = False

                    elif chunk.type == "token":
                        # When prefill is used, first tokens are thinking content
                        # until we see </think>
                        if in_thinking_mode:
                            # Add to pending buffer and check for </think>
                            pending_buffer += chunk.content

                            # Check if </think> is complete in buffer
                            if "</think>" in pending_buffer:
                                # Split at </think>
                                parts = pending_buffer.split("</think>", 1)
                                thinking_content += parts[0]
                                in_thinking_mode = False
                                pending_buffer = ""
                                # Rest goes to response
                                if len(parts) > 1:
                                    response_content += parts[1]
                                    if response_content.strip():
                                        panel = self.renderer.create_streaming_panel(response_content, thinking_content)
                                        live.update(panel)
                            else:
                                # Check for partial </think> at end of buffer
                                partial_end_tag = False
                                for i in range(1, min(9, len(pending_buffer) + 1)):
                                    if pending_buffer[-i:] == "</think>"[:i]:
                                        partial_end_tag = True
                                        break

                                if partial_end_tag:
                                    # Keep potential partial tag in buffer
                                    # Add everything except last 8 chars to thinking
                                    safe_len = max(0, len(pending_buffer) - 8)
                                    if safe_len > 0:
                                        thinking_content += pending_buffer[:safe_len]
                                        pending_buffer = pending_buffer[safe_len:]
                                else:
                                    # No partial tag, flush buffer to thinking
                                    thinking_content += pending_buffer
                                    pending_buffer = ""

                                spinner_frame += 1
                                panel = self.renderer.create_thinking_panel(
                                    thinking_content, spinner_frame
                                )
                                live.update(panel)
                        else:
                            response_content += chunk.content
                            panel = self.renderer.create_streaming_panel(response_content, thinking_content)
                            live.update(panel)

                    elif chunk.type == "error":
                        self.renderer.render_error(chunk.content)
                        return ""

                    elif chunk.type == "done":
                        # Flush any remaining pending buffer
                        if pending_buffer:
                            if in_thinking_mode:
                                thinking_content += pending_buffer
                            else:
                                response_content += pending_buffer
                        break

        except KeyboardInterrupt:
            console.print("\n[yellow]Keskeytetty.[/yellow]")
            return ""

        # Handle case where model didn't close </think> tag
        # If still in thinking mode at end, try to extract response from thinking
        if in_thinking_mode and thinking_content and not response_content:
            # Check if </think> is in thinking content (partial parsing issue)
            if "</think>" in thinking_content:
                parts = thinking_content.split("</think>", 1)
                thinking_content = parts[0]
                if len(parts) > 1:
                    response_content = parts[1].strip()

        # Render complete response (replaces live display)
        if response_content.strip():
            self.renderer.render_complete_response(response_content, thinking_content)
            # Add to history only if we got a real response
            self.conversation.add_assistant(response_content, thinking_content)
        elif thinking_content.strip():
            # Model generated content but never closed </think>
            # This likely means the model doesn't support <think> prefill
            # Treat the "thinking" content as the actual response
            if prefill_used:
                # Prefill was used but model didn't follow the format
                # Show the content as response (not as thinking)
                self.renderer.render_complete_response(thinking_content, "")
                self.conversation.add_assistant(thinking_content, "")
            else:
                # Model started thinking but never finished - unusual
                console.print(Panel(
                    f"[dim]Malli generoi ajattelua mutta ei vastausta.[/dim]",
                    title="[yellow]Huomio[/yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                ))
                self.renderer.render_complete_response(thinking_content, "")
                self.conversation.add_assistant(thinking_content, "")
        else:
            # Truly empty response
            self.renderer.render_error("Tyhja vastaus mallilta")

        return response_content

    def _print_help(self):
        """Nayta ohjeet."""
        prefill_status = "PAALLA" if self.thinking_prefill else "POIS"
        thoughts_status = "PAALLA" if self.renderer.show_thoughts else "POIS"
        help_text = f"""
## Komennot

| Komento | Kuvaus |
|---------|--------|
| `lopeta` / `exit` | Poistu chatista |
| `tyhjenna` / `clear` | Tyhjenna keskusteluhistoria |
| `nayta` / `show` | Nayta ajatukset ({thoughts_status}) |
| `piilota` / `hide` | Piilota ajatukset |
| `prefill` / `ajattelu` | Vaihda prefill-tila ({prefill_status}) |
| `ohje` / `help` | Nayta tama ohje |

## Ajatuksien Nayttaminen

- **nayta**: Nayta mallin ajatukset kokonaan (oletus)
- **piilota**: Nayta vain tiivistelma ajatuksista

## Prefill-tila (edistynyt)

Kun prefill on PAALLA:
- Malli pakotetaan aloittamaan `<think>`-tagilla
- Vain DeepSeek R1 -malleille

Kun prefill on POIS (oletus):
- Malli paattaa itse milloin ajattelee
- Toimii poro-r1 ja muiden mallien kanssa

## Vinkkeja

- Reasoning-mallit nayttavat ajattelun automaattisesti
- Keskusteluhistoria sailyy sessiossa
- Ctrl+C keskeyttaa vastauksen

"""
        console.print(Panel(
            Markdown(help_text),
            title="[bold cyan]Ohje[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    def run(self):
        """Kaynnista interaktiivinen chat-sessio."""
        # Print header
        self.renderer.render_header(self.thinking_prefill)

        # Main loop
        try:
            while True:
                try:
                    # Get user input
                    console.print("[bold blue]>>[/bold blue] ", end="")
                    user_input = input().strip()

                    # Handle commands (with or without / prefix)
                    if not user_input:
                        continue

                    # Normalize command (remove leading /)
                    cmd = user_input.lower().lstrip("/")

                    if cmd in ["lopeta", "exit", "quit", "q"]:
                        break

                    if cmd in ["tyhjenna", "clear", "uusi"]:
                        self.conversation.clear()
                        console.print(Panel(
                            "[green]Keskustelu tyhjennetty.[/green]",
                            border_style="green",
                            box=box.ROUNDED,
                        ))
                        continue

                    if cmd in ["ajattelu", "think", "thinking", "prefill"]:
                        self.set_thinking_prefill(not self.thinking_prefill)
                        status = "[green]PAALLA[/green]" if self.thinking_prefill else "[red]POIS[/red]"
                        console.print(Panel(
                            f"Prefill-tila: {status}\n[dim]Temperature: {self.options['temperature']}[/dim]",
                            border_style="yellow",
                            box=box.ROUNDED,
                        ))
                        continue

                    if cmd in ["nayta", "show"]:
                        self.renderer.show_thoughts = True
                        console.print(Panel(
                            "[green]Ajatukset naytetaan[/green]",
                            border_style="green",
                            box=box.ROUNDED,
                        ))
                        continue

                    if cmd in ["piilota", "hide"]:
                        self.renderer.show_thoughts = False
                        console.print(Panel(
                            "[yellow]Ajatukset piilotettu[/yellow]",
                            border_style="yellow",
                            box=box.ROUNDED,
                        ))
                        continue

                    if cmd in ["ohje", "help", "?"]:
                        self._print_help()
                        continue

                    # Chat
                    console.print()
                    self.chat(user_input)
                    console.print()

                except KeyboardInterrupt:
                    console.print("\n[yellow]Ctrl+C - kirjoita 'lopeta' poistuaksesi[/yellow]")
                    continue

        except EOFError:
            pass

        # Goodbye
        console.print(Panel(
            "[cyan]Kiitos keskustelusta![/cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_system_prompt_from_modelfile(modelfile: str) -> Optional[str]:
    """
    Poimi system prompt Modelfilesta.

    Args:
        modelfile: Modelfile-sisalto

    Returns:
        System prompt tai None
    """
    if not modelfile:
        return None

    # Try triple-quoted: SYSTEM """..."""
    match = re.search(r'SYSTEM\s+"""(.*?)"""', modelfile, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try double-quoted: SYSTEM "..."
    match = re.search(r'SYSTEM\s+"([^"]*)"', modelfile)
    if match:
        return match.group(1)

    # Try single-quoted: SYSTEM '...'
    match = re.search(r"SYSTEM\s+'([^']*)'", modelfile)
    if match:
        return match.group(1)

    # Try unquoted (single line): SYSTEM text here
    match = re.search(r'SYSTEM\s+([^\n]+)', modelfile)
    if match:
        return match.group(1).strip()

    return None


def get_model_size_from_ollama(model_name: str, base_url: str = "http://localhost:11434") -> str:
    """
    Hae mallin koko Ollamasta.

    Args:
        model_name: Mallin nimi
        base_url: Ollama URL

    Returns:
        Koko merkkijonona (esim. "8.0B") tai ""
    """
    try:
        # "model" on nykyinen avain, "name" vanhempien Ollama-versioiden
        response = requests.post(
            f"{base_url}/api/show",
            json={"model": model_name, "name": model_name},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            # Look for parameter count in details
            details = data.get("details", {})
            param_size = details.get("parameter_size", "")
            if param_size:
                return param_size

        # Fallback: /api/tags listaa mallit levykokoineen
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            for model in response.json().get("models", []):
                if model.get("name") == model_name or model.get("model") == model_name:
                    size_bytes = model.get("size", 0)
                    if size_bytes:
                        return f"{size_bytes / (1024 ** 3):.1f} GB"
    except Exception:
        pass
    return ""
