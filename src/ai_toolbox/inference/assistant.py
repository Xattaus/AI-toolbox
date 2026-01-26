"""
AI TOOLBOX - AI Assistant Module
================================

Claude CLI integration for development assistance.
"""

import subprocess
import shutil
from pathlib import Path

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel

console = Console()

# AI Toolbox root directory
TOOLBOX_ROOT = Path(__file__).parent.parent.parent.parent

# Custom questionary style
custom_style = Style([
    ('qmark', 'fg:#ff9d00 bold'),
    ('question', 'fg:white bold'),
    ('answer', 'fg:#00d7ff bold'),
    ('pointer', 'fg:#ff9d00 bold'),
    ('highlighted', 'fg:#ff9d00 bold'),
    ('selected', 'fg:#00ff00'),
    ('separator', 'fg:#666666'),
    ('instruction', 'fg:#666666'),
])


def print_mini_banner(tool_name: str):
    """Print a minimal banner for individual tools."""
    console.print(f"""
[bold orange1]AI TOOLBOX[/bold orange1] [dim]>[/dim] [bold white]{tool_name}[/bold white]
[dim]{'-' * 50}[/dim]
""")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[bold red][ERROR][/bold red] {message}")


def check_claude_cli() -> bool:
    """Check if Claude CLI is installed."""
    return shutil.which("claude") is not None


def launch_claude():
    """Launch Claude CLI in interactive mode."""
    if not check_claude_cli():
        print_error("Claude CLI not found!")
        show_claude_setup()
        return

    console.print("\n[cyan]Launching Claude CLI...[/cyan]")
    console.print("[dim]Press Ctrl+C to exit Claude and return to AI Toolbox[/dim]\n")

    try:
        subprocess.run(
            ["claude"],
            cwd=str(TOOLBOX_ROOT),
            shell=True
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Returned to AI Toolbox[/yellow]")
    except Exception as e:
        print_error(f"Failed to launch Claude: {e}")

    questionary.press_any_key_to_continue(style=custom_style).ask()


def launch_claude_dev_mode():
    """Launch Claude with AI Toolbox development context."""
    if not check_claude_cli():
        print_error("Claude CLI not found!")
        show_claude_setup()
        return

    dev_prompt = """You are helping develop AI TOOLBOX - a local AI model management tool.

Project structure:
- src/ai_toolbox/main.py - Main application with menus
- src/ai_toolbox/ui.py - UI components (Rich library)
- src/ai_toolbox/model_library.py - Model library management
- src/ai_toolbox/model_downloader.py - HuggingFace downloader
- src/ai_toolbox/gguf_converter.py - GGUF conversion and quantization
- src/ai_toolbox/mcp_server.py - MCP server for Claude Code

Tech stack: Python, Rich, Questionary, HuggingFace Hub

Focus on: Clean code, good UX, helpful error messages."""

    console.print("\n[cyan]Launching Claude in Development Mode...[/cyan]")
    console.print("[dim]Claude has context about AI Toolbox project[/dim]\n")

    try:
        subprocess.run(
            ["claude", "-p", dev_prompt],
            cwd=str(TOOLBOX_ROOT),
            shell=True
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Returned to AI Toolbox[/yellow]")
    except Exception as e:
        print_error(f"Failed to launch Claude: {e}")

    questionary.press_any_key_to_continue(style=custom_style).ask()


def launch_claude_with_prompt(prompt: str):
    """Launch Claude CLI with a specific prompt."""
    if not check_claude_cli():
        print_error("Claude CLI not found!")
        show_claude_setup()
        return

    console.print(f"\n[cyan]Launching Claude with prompt...[/cyan]")
    console.print(f"[dim]{prompt[:60]}{'...' if len(prompt) > 60 else ''}[/dim]\n")

    try:
        subprocess.run(
            ["claude", "-p", prompt],
            cwd=str(TOOLBOX_ROOT),
            shell=True
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Returned to AI Toolbox[/yellow]")
    except Exception as e:
        print_error(f"Failed to launch Claude: {e}")

    questionary.press_any_key_to_continue(style=custom_style).ask()


def quick_prompts_menu():
    """Show pre-defined development prompts."""
    print_mini_banner("Quick Prompts")

    prompts = [
        {
            "name": "Add new feature",
            "prompt": "Help me add a new feature to AI Toolbox. First, let me describe what I want to build.",
            "desc": "Start building a new feature"
        },
        {
            "name": "Fix a bug",
            "prompt": "I found a bug in AI Toolbox. Let me describe the issue and help me fix it.",
            "desc": "Debug and fix issues"
        },
        {
            "name": "Improve UI",
            "prompt": "Help me improve the user interface of AI Toolbox. Review the current UI code and suggest improvements.",
            "desc": "Enhance user experience"
        },
        {
            "name": "Add tests",
            "prompt": "Help me write tests for AI Toolbox. Start by examining the current code structure.",
            "desc": "Write unit tests"
        },
        {
            "name": "Code review",
            "prompt": "Review the AI Toolbox codebase. Look for potential improvements, bugs, and best practices.",
            "desc": "Review code quality"
        },
        {
            "name": "Documentation",
            "prompt": "Help me improve the documentation for AI Toolbox. Review the existing docs and suggest additions.",
            "desc": "Write documentation"
        },
        {
            "name": "GGUF Converter",
            "prompt": "Help me implement the GGUF conversion functionality in AI Toolbox. It should use llama.cpp's convert scripts.",
            "desc": "Implement conversion"
        },
        {
            "name": "Performance",
            "prompt": "Help me optimize the performance of AI Toolbox. Profile the code and suggest improvements.",
            "desc": "Optimize speed"
        },
    ]

    choices = []
    for p in prompts:
        title = f"{p['name']:<20} {p['desc']}"
        choices.append(questionary.Choice(title=title, value=p['prompt']))

    choices.append(questionary.Separator())
    choices.append(questionary.Choice(title="Back", value="back"))

    selected = questionary.select(
        "Select a development task:",
        choices=choices,
        style=custom_style,
        qmark=">>",
        pointer=">"
    ).ask()

    if selected and selected != "back":
        launch_claude_with_prompt(selected)


def custom_prompt_menu():
    """Launch Claude with a custom prompt."""
    prompt = questionary.text(
        "Enter your prompt for Claude:",
        style=custom_style,
        multiline=False
    ).ask()

    if prompt:
        launch_claude_with_prompt(prompt)


def show_claude_setup():
    """Show Claude CLI setup instructions."""
    print_mini_banner("Claude CLI Setup")

    setup_text = """
[bold white]Claude CLI Installation[/bold white]

[cyan]1. Install Node.js[/cyan]
   Download from: https://nodejs.org/

[cyan]2. Install Claude CLI[/cyan]
   npm install -g @anthropic-ai/claude-code

[cyan]3. Authenticate[/cyan]
   claude login

[cyan]4. Verify installation[/cyan]
   claude --version

[bold white]Environment Variables (optional)[/bold white]

Set ANTHROPIC_API_KEY for API access:
   set ANTHROPIC_API_KEY=your-api-key

[bold white]MCP Server Setup[/bold white]

AI Toolbox includes an MCP server for Claude Code.
See CLAUDE_MCP_SETUP.md for configuration instructions.

[bold white]Usage Tips[/bold white]

- Use /help in Claude for commands
- Press Ctrl+C to exit
- Files are saved automatically
"""
    console.print(Panel(
        setup_text,
        title="[bold]Setup Instructions[/bold]",
        border_style="cyan",
        padding=(1, 2)
    ))

    questionary.press_any_key_to_continue(style=custom_style).ask()


def ai_assistant_menu():
    """AI Assistant submenu - Claude CLI integration."""
    while True:
        print_mini_banner("AI Assistant")

        # Check if Claude CLI is available
        claude_available = check_claude_cli()

        if claude_available:
            console.print("[green]Claude CLI: Available[/green]\n")
        else:
            console.print("[yellow]Claude CLI: Not found[/yellow]")
            console.print("[dim]Install with: npm install -g @anthropic-ai/claude-code[/dim]\n")

        choices = [
            questionary.Choice(
                title="Start Claude               Interactive development session",
                value="start"
            ),
            questionary.Choice(
                title="Development Mode          Claude with toolbox context",
                value="dev"
            ),
            questionary.Separator(),
            questionary.Choice(
                title="Quick Prompts             Pre-defined development tasks",
                value="prompts"
            ),
            questionary.Choice(
                title="Custom Prompt             Start with your own prompt",
                value="custom"
            ),
            questionary.Separator(),
            questionary.Choice(
                title="Setup Instructions        How to configure Claude CLI",
                value="setup"
            ),
            questionary.Choice(
                title="Back                      Return to main menu",
                value="back"
            ),
        ]

        choice = questionary.select(
            "AI Assistant:",
            choices=choices,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()

        if choice is None or choice == "back":
            break
        elif choice == "start":
            launch_claude()
        elif choice == "dev":
            launch_claude_dev_mode()
        elif choice == "prompts":
            quick_prompts_menu()
        elif choice == "custom":
            custom_prompt_menu()
        elif choice == "setup":
            show_claude_setup()
