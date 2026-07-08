"""Smoke tests for integrations.mcp_server: tool schemas and dispatch."""

from ai_toolbox.integrations.mcp_server import AIToolboxMCP, format_size


def test_format_size_returns_readable_string():
    for n in (0, 1024, 5 * 1024 * 1024, 3 * 1024**3):
        out = format_size(n)
        assert isinstance(out, str) and out
        assert any(ch.isdigit() for ch in out)


def test_get_tools_returns_valid_schemas():
    tools = AIToolboxMCP().get_tools()
    assert len(tools) >= 5
    names = set()
    for tool in tools:
        assert {"name", "description", "inputSchema"} <= set(tool)
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]
        names.add(tool["name"])
    assert len(names) == len(tools)  # names are unique


def test_get_tools_exposes_core_capabilities():
    names = {t["name"] for t in AIToolboxMCP().get_tools()}
    assert "search_models" in names
    assert "download_model" in names
    assert any("gguf" in n or "quant" in n for n in names)


def test_call_unknown_tool_returns_error_string():
    result = AIToolboxMCP().call_tool("no_such_tool", {})
    assert isinstance(result, str) and result
