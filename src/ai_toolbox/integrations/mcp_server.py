"""
AI TOOLBOX - MCP Server
=======================

Model Context Protocol server for Claude Code integration.
Allows Claude to access AI Toolbox functionality as tools.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None  # Type placeholder when MCP not installed
    Tool = None
    TextContent = None

from ..models.library import ModelLibrary
from ..models.downloader import ModelDownloader
from ..conversion.converter import GGUFConverter


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


class AIToolboxMCP:
    """MCP Server for AI Toolbox."""

    def __init__(self):
        self.library = ModelLibrary()
        self.downloader = ModelDownloader()
        self.converter = GGUFConverter()

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools."""
        return [
            # === Model Search & Download ===
            {
                "name": "search_models",
                "description": "Search for AI models on HuggingFace Hub. Returns model IDs, download counts, and metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'llama', 'mistral', 'qwen')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10,
                        },
                        "task": {
                            "type": "string",
                            "description": "Filter by task type",
                            "enum": [
                                "text-generation",
                                "text2text-generation",
                                "feature-extraction",
                            ],
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_model_info",
                "description": "Get detailed information about a specific HuggingFace model including files, size, and metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')",
                        }
                    },
                    "required": ["model_id"],
                },
            },
            {
                "name": "download_model",
                "description": "Download a model from HuggingFace Hub to the local library.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "HuggingFace model ID to download",
                        },
                        "safetensors_only": {
                            "type": "boolean",
                            "description": "Download only safetensors files (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["model_id"],
                },
            },
            {
                "name": "check_downloaded",
                "description": "Check if a model is already downloaded.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "HuggingFace model ID to check",
                        }
                    },
                    "required": ["model_id"],
                },
            },
            # === Model Library ===
            {
                "name": "list_library",
                "description": "List all models in the local AI Toolbox library.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "format_filter": {
                            "type": "string",
                            "description": "Filter by format (e.g., 'gguf', 'safetensors')",
                        },
                        "source_filter": {
                            "type": "string",
                            "description": "Filter by source (e.g., 'huggingface', 'local')",
                        },
                    },
                },
            },
            {
                "name": "library_stats",
                "description": "Get statistics about the model library (total models, size, etc.).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            # === GGUF Conversion ===
            {
                "name": "convert_to_gguf",
                "description": "Convert a HuggingFace model to GGUF format. Requires the model to be downloaded first.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to the HuggingFace model directory",
                        },
                        "output_type": {
                            "type": "string",
                            "description": "Output type (f32, f16, bf16, q8_0)",
                            "default": "f16",
                            "enum": ["f32", "f16", "bf16", "q8_0"],
                        },
                    },
                    "required": ["model_path"],
                },
            },
            {
                "name": "quantize_gguf",
                "description": "Quantize a GGUF model to a smaller size. Requires llama-quantize binary.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": "Path to the input GGUF file",
                        },
                        "quantization": {
                            "type": "string",
                            "description": "Quantization type (q4_k_m recommended)",
                            "default": "q4_k_m",
                            "enum": [
                                "q8_0",
                                "q6_k",
                                "q5_k_m",
                                "q5_k_s",
                                "q4_k_m",
                                "q4_k_s",
                                "q4_0",
                                "q3_k_m",
                                "q3_k_s",
                                "q2_k",
                            ],
                        },
                    },
                    "required": ["input_path"],
                },
            },
            {
                "name": "convert_and_quantize",
                "description": "Convert HuggingFace model to GGUF and quantize in one step. This is the recommended way to create GGUF models.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to the HuggingFace model directory",
                        },
                        "quantization": {
                            "type": "string",
                            "description": "Target quantization (q4_k_m recommended)",
                            "default": "q4_k_m",
                            "enum": [
                                "q8_0",
                                "q6_k",
                                "q5_k_m",
                                "q5_k_s",
                                "q4_k_m",
                                "q4_k_s",
                                "q4_0",
                                "q3_k_m",
                                "q3_k_s",
                                "q2_k",
                            ],
                        },
                        "keep_f16": {
                            "type": "boolean",
                            "description": "Keep intermediate F16 file (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["model_path"],
                },
            },
            {
                "name": "list_quantization_types",
                "description": "List all available quantization types with their properties (bits per weight, quality level, description).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "recommend_quantization",
                "description": "Get recommended quantization types based on model size and available RAM.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_params_billion": {
                            "type": "number",
                            "description": "Model parameters in billions (e.g., 7, 13, 70)",
                        },
                        "available_ram_gb": {
                            "type": "number",
                            "description": "Available RAM in GB (auto-detected if not provided)",
                        },
                    },
                    "required": ["model_params_billion"],
                },
            },
            {
                "name": "estimate_model_size",
                "description": "Estimate the output size of a model after conversion/quantization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to the HuggingFace model directory",
                        },
                        "quantization": {
                            "type": "string",
                            "description": "Target quantization type",
                            "default": "q4_k_m",
                        },
                    },
                    "required": ["model_path"],
                },
            },
            # === Converter Setup ===
            {
                "name": "check_converter_status",
                "description": "Check if the GGUF converter (llama.cpp) is installed and ready.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "setup_llama_cpp",
                "description": "Download and set up llama.cpp for GGUF conversion. Required before converting models.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force re-download even if exists",
                            "default": False,
                        }
                    },
                },
            },
            # === Utilities ===
            {
                "name": "calculate_vram",
                "description": "Calculate VRAM requirements for a model with different quantization levels.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "parameters_billion": {
                            "type": "number",
                            "description": "Model parameters in billions (e.g., 7, 13, 70)",
                        }
                    },
                    "required": ["parameters_billion"],
                },
            },
            {
                "name": "get_system_info",
                "description": "Get system information (RAM, CPU) for conversion recommendations.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        try:
            # Model Search & Download
            if name == "search_models":
                return self._search_models(
                    arguments.get("query", ""), arguments.get("limit", 10), arguments.get("task")
                )
            elif name == "get_model_info":
                return self._get_model_info(arguments["model_id"])
            elif name == "download_model":
                return self._download_model(
                    arguments["model_id"], arguments.get("safetensors_only", True)
                )
            elif name == "check_downloaded":
                return self._check_downloaded(arguments["model_id"])

            # Model Library
            elif name == "list_library":
                return self._list_library(
                    arguments.get("format_filter"), arguments.get("source_filter")
                )
            elif name == "library_stats":
                return self._library_stats()

            # GGUF Conversion
            elif name == "convert_to_gguf":
                return self._convert_to_gguf(
                    arguments["model_path"], arguments.get("output_type", "f16")
                )
            elif name == "quantize_gguf":
                return self._quantize_gguf(
                    arguments["input_path"], arguments.get("quantization", "q4_k_m")
                )
            elif name == "convert_and_quantize":
                return self._convert_and_quantize(
                    arguments["model_path"],
                    arguments.get("quantization", "q4_k_m"),
                    arguments.get("keep_f16", False),
                )
            elif name == "list_quantization_types":
                return self._list_quantization_types()
            elif name == "recommend_quantization":
                return self._recommend_quantization(
                    arguments["model_params_billion"], arguments.get("available_ram_gb")
                )
            elif name == "estimate_model_size":
                return self._estimate_model_size(
                    arguments["model_path"], arguments.get("quantization", "q4_k_m")
                )

            # Converter Setup
            elif name == "check_converter_status":
                return self._check_converter_status()
            elif name == "setup_llama_cpp":
                return self._setup_llama_cpp(arguments.get("force", False))

            # Utilities
            elif name == "calculate_vram":
                return self._calculate_vram(arguments["parameters_billion"])
            elif name == "get_system_info":
                return self._get_system_info()

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # === Model Search & Download ===

    def _search_models(self, query: str, limit: int, task: Optional[str]) -> str:
        """Search HuggingFace models."""
        results = self.downloader.search_models(query, limit=limit, filter_task=task)

        if not results:
            return json.dumps({"results": [], "message": "No models found"})

        models = []
        for r in results:
            models.append(
                {
                    "model_id": r.model_id,
                    "author": r.author,
                    "downloads": r.downloads,
                    "likes": r.likes,
                    "task": r.pipeline_tag,
                    "tags": r.tags[:5] if r.tags else [],
                }
            )

        return json.dumps({"results": models, "total": len(models)})

    def _get_model_info(self, model_id: str) -> str:
        """Get detailed model information."""
        details = self.downloader.get_model_details(model_id)

        if not details:
            return json.dumps({"error": f"Model not found: {model_id}"})

        existing = self.downloader.check_exists(model_id)

        return json.dumps(
            {
                "model_id": details.model_id,
                "author": details.author,
                "downloads": details.downloads,
                "likes": details.likes,
                "task": details.pipeline_tag,
                "total_size": format_size(details.total_size),
                "total_size_bytes": details.total_size,
                "file_count": len(details.files),
                "tags": details.tags[:10] if details.tags else [],
                "already_downloaded": str(existing) if existing else None,
                "files": [
                    {"name": f["name"], "size": format_size(f["size"])}
                    for f in sorted(details.files, key=lambda x: x["size"], reverse=True)[:10]
                ],
            }
        )

    def _download_model(self, model_id: str, safetensors_only: bool) -> str:
        """Download a model."""
        existing = self.downloader.check_exists(model_id)
        if existing:
            return json.dumps(
                {
                    "status": "already_exists",
                    "path": str(existing),
                    "message": f"Model already downloaded at {existing}",
                }
            )

        include_patterns = None
        exclude_patterns = None

        if safetensors_only:
            include_patterns = ["*.safetensors", "*.json", "*.txt", "*.model"]
            exclude_patterns = ["*.bin", "*.md"]

        path = self.downloader.download_model(
            model_id, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )

        if path:
            try:
                entry = self.library.add_model(
                    path=str(path), source="huggingface", source_id=model_id
                )
                return json.dumps(
                    {
                        "status": "success",
                        "path": str(path),
                        "library_id": entry.id,
                        "message": f"Downloaded and added to library: {entry.name}",
                    }
                )
            except Exception as e:
                return json.dumps(
                    {
                        "status": "partial",
                        "path": str(path),
                        "message": f"Downloaded but failed to add to library: {e}",
                    }
                )
        else:
            return json.dumps(
                {
                    "status": "failed",
                    "message": "Download failed. Check if the model exists and you have access.",
                }
            )

    def _check_downloaded(self, model_id: str) -> str:
        """Check if a model is downloaded."""
        existing = self.downloader.check_exists(model_id)

        if existing:
            return json.dumps({"downloaded": True, "path": str(existing)})
        else:
            return json.dumps({"downloaded": False, "path": None})

    # === Model Library ===

    def _list_library(self, format_filter: Optional[str], source_filter: Optional[str]) -> str:
        """List models in library."""
        models = self.library.list_models(format_filter=format_filter, source_filter=source_filter)

        if not models:
            return json.dumps({"models": [], "message": "No models in library"})

        result = []
        for m in models:
            result.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "format": m.format,
                    "quantization": m.quantization,
                    "size": format_size(m.size_bytes),
                    "source": m.source,
                    "source_id": m.source_id,
                    "path": m.path,
                }
            )

        return json.dumps({"models": result, "total": len(result)})

    def _library_stats(self) -> str:
        """Get library statistics."""
        stats = self.library.get_stats()
        return json.dumps(
            {
                "total_models": stats["total_models"],
                "total_size_gb": round(stats["total_size_gb"], 2),
                "formats": stats.get("formats", {}),
                "sources": stats.get("sources", {}),
            }
        )

    # === GGUF Conversion ===

    def _convert_to_gguf(self, model_path: str, output_type: str) -> str:
        """Convert model to GGUF."""
        result = self.converter.convert_to_gguf(model_path=model_path, output_type=output_type)
        return json.dumps(result)

    def _quantize_gguf(self, input_path: str, quantization: str) -> str:
        """Quantize GGUF model."""
        result = self.converter.quantize_gguf(input_path=input_path, quantization=quantization)
        return json.dumps(result)

    def _convert_and_quantize(self, model_path: str, quantization: str, keep_f16: bool) -> str:
        """Convert and quantize in one step."""
        result = self.converter.convert_and_quantize(
            model_path=model_path, quantization=quantization, keep_f16=keep_f16
        )
        return json.dumps(result)

    def _list_quantization_types(self) -> str:
        """List quantization types."""
        types = self.converter.list_quantization_types()
        return json.dumps({"quantization_types": types})

    def _recommend_quantization(
        self, model_params_billion: float, available_ram_gb: Optional[float]
    ) -> str:
        """Get quantization recommendations."""
        recommendations = self.converter.recommend_quantization(
            model_params_billions=model_params_billion, available_ram_gb=available_ram_gb
        )
        return json.dumps(
            {"model_params_billion": model_params_billion, "recommendations": recommendations}
        )

    def _estimate_model_size(self, model_path: str, quantization: str) -> str:
        """Estimate model size."""
        estimate = self.converter.estimate_model_size(
            model_path=Path(model_path), quantization=quantization
        )
        return json.dumps(estimate)

    # === Converter Setup ===

    def _check_converter_status(self) -> str:
        """Check converter status."""
        status = self.converter.check_llama_cpp()
        return json.dumps(status)

    def _setup_llama_cpp(self, force: bool) -> str:
        """Set up llama.cpp."""
        success = self.converter.setup_llama_cpp(force=force)
        if success:
            status = self.converter.check_llama_cpp()
            return json.dumps(
                {"success": True, "message": "llama.cpp set up successfully", "status": status}
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "message": "Failed to set up llama.cpp. Make sure git is installed.",
                }
            )

    # === Utilities ===

    def _calculate_vram(self, params_b: float) -> str:
        """Calculate VRAM requirements."""
        calculations = [
            ("F16", 16, "Full precision"),
            ("Q8_0", 8, "8-bit quantization"),
            ("Q6_K", 6.5, "6-bit K-quant"),
            ("Q5_K_M", 5.5, "5-bit K-quant"),
            ("Q4_K_M", 4.5, "4-bit K-quant (recommended)"),
            ("Q4_0", 4.0, "4-bit basic"),
            ("Q3_K_M", 3.5, "3-bit K-quant"),
            ("Q2_K", 2.5, "2-bit K-quant"),
        ]

        results = []
        for name, bits, desc in calculations:
            model_gb = (params_b * 1e9 * bits / 8) / (1024**3)
            ctx_4k = model_gb + 0.5
            ctx_8k = model_gb + 1.0

            results.append(
                {
                    "quantization": name,
                    "bits": bits,
                    "model_size_gb": round(model_gb, 1),
                    "with_4k_context_gb": round(ctx_4k, 1),
                    "with_8k_context_gb": round(ctx_8k, 1),
                    "description": desc,
                }
            )

        return json.dumps({"model_parameters_billion": params_b, "requirements": results})

    def _get_system_info(self) -> str:
        """Get system info."""
        info = self.converter.get_system_info()
        return json.dumps(info)


def create_mcp_server() -> Optional[Any]:
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        return None

    server = Server("ai-toolbox")
    toolbox = AIToolboxMCP()

    @server.list_tools()
    async def list_tools():
        """List available tools."""
        tools = toolbox.get_tools()
        return [
            Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        """Handle tool calls."""
        result = toolbox.call_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    return server


async def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("MCP not available. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = create_mcp_server()
    if server:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)


def main():
    """Entry point for MCP server."""
    import asyncio

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
