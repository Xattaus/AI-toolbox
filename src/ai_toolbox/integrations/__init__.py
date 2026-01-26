"""
AI Toolbox - Integrations Module
================================

External integrations including MCP server and Ollama manager.
"""

from .mcp_server import AIToolboxMCP, create_mcp_server, run_server, main
from .ollama import OllamaManager, SYSTEM_PROMPTS, OllamaModelInfo, ModelfileConfig

__all__ = [
    "AIToolboxMCP",
    "create_mcp_server",
    "run_server",
    "main",
    "OllamaManager",
    "SYSTEM_PROMPTS",
    "OllamaModelInfo",
    "ModelfileConfig",
]
