"""
AI TOOLBOX - Inference Module
=============================

Chat, benchmark, and assistant functionality for AI model inference.
"""

from .chat import (
    AIChat,
    ChatBackend,
    ChatMessage,
    LlamaCppBackend,
    OllamaBackend,
    ToolRegistry,
    ai_chat_menu,
)

from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    ComparisonReport,
    DEFAULT_PROMPTS,
)

from .assistant import (
    ai_assistant_menu,
    check_claude_cli,
    launch_claude,
    launch_claude_dev_mode,
    launch_claude_with_prompt,
    quick_prompts_menu,
    custom_prompt_menu,
    show_claude_setup,
)

from .beautiful_chat import (
    BeautifulChat,
    ChatRenderer,
    ConversationManager,
    OllamaStreamClient,
    extract_system_prompt_from_modelfile,
    get_model_size_from_ollama,
)

__all__ = [
    # Chat
    "AIChat",
    "ChatBackend",
    "ChatMessage",
    "LlamaCppBackend",
    "OllamaBackend",
    "ToolRegistry",
    "ai_chat_menu",
    # Benchmark
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComparisonReport",
    "DEFAULT_PROMPTS",
    # Assistant
    "ai_assistant_menu",
    "check_claude_cli",
    "launch_claude",
    "launch_claude_dev_mode",
    "launch_claude_with_prompt",
    "quick_prompts_menu",
    "custom_prompt_menu",
    "show_claude_setup",
    # Beautiful Chat
    "BeautifulChat",
    "ChatRenderer",
    "ConversationManager",
    "OllamaStreamClient",
    "extract_system_prompt_from_modelfile",
    "get_model_size_from_ollama",
]
