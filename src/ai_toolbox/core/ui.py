"""
AI TOOLBOX - UI Components
==========================

Beautiful terminal UI components using the Rich library.
Follows CLI design best practices from clig.dev.
"""

import os
import sys
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.style import Style
from rich.theme import Theme
from rich import box

# ============================================================================
# THEME & COLOR SCHEME
# ============================================================================
# Consistent color palette across the entire application
# Orange/Amber = branding, Cyan = info, Green = success, Yellow = warning, Red = error

TOOLBOX_THEME = Theme({
    "brand": "bold orange1",
    "brand.dim": "dim orange3",
    "accent": "cyan",
    "accent.bright": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "muted": "dim white",
    "highlight": "bold white",
    "key": "bold yellow",
    "value": "white",
    "separator": "dim cyan",
})

# Global console instance with theme
console = Console(theme=TOOLBOX_THEME)

# ============================================================================
# QUESTIONARY STYLE (for interactive menus)
# ============================================================================
from questionary import Style as QStyle

MENU_STYLE = QStyle([
    ('qmark', 'fg:#ff9d00 bold'),       # Question mark
    ('question', 'fg:#ffffff bold'),     # Question text
    ('answer', 'fg:#00d7ff bold'),       # Selected answer
    ('pointer', 'fg:#ff9d00 bold'),      # Selection pointer >
    ('highlighted', 'fg:#ff9d00 bold'),  # Highlighted option
    ('selected', 'fg:#00ff00 bold'),     # Multi-select selected
    ('separator', 'fg:#666666'),         # Separator lines
    ('instruction', 'fg:#888888'),       # Instructions
    ('text', 'fg:#ffffff'),              # Regular text
    ('disabled', 'fg:#666666 italic'),   # Disabled options
])


def print_toolbox_banner(subtitle: str = ""):
    """Print the AI TOOLBOX banner with optional subtitle."""

    banner = """
[bold orange1]
       _    ___   _____ ___   ___  _     ____   _____  __
      / \\  |_ _| |_   _/ _ \\ / _ \\| |   | __ ) / _ \\ \\/ /
     / _ \\  | |    | || | | | | | | |   |  _ \\| | | \\  /
    / ___ \\ | |    | || |_| | |_| | |___| |_) | |_| /  \\
   /_/   \\_\\___|   |_| \\___/ \\___/|_____|____/ \\___/_/\\_\\
[/bold orange1]
[dim]============================================================[/dim]"""

    if subtitle:
        banner += f"\n[bold white]  {subtitle}[/bold white]"

    banner += "\n[dim]  Local AI Toolkit[/dim]\n"

    console.print(banner)


def print_mini_banner(tool_name: str, subtitle: str = ""):
    """Print a beautiful banner for individual tools."""
    width = 56

    # Top border
    console.print(f"\n[dim]{'─' * width}[/dim]")

    # Title line
    console.print(f"[bold orange1]  AI TOOLBOX[/bold orange1] [dim]»[/dim] [bold white]{tool_name}[/bold white]")

    # Subtitle if provided
    if subtitle:
        console.print(f"[dim]  {subtitle}[/dim]")

    # Bottom border
    console.print(f"[dim]{'─' * width}[/dim]\n")


def create_menu_table(title: str, items: list, show_index: bool = True) -> Table:
    """Create a styled menu table."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        padding=(0, 1),
    )

    if show_index:
        table.add_column("#", style="bold yellow", width=4, justify="right")
    table.add_column("Option", style="white")
    table.add_column("Description", style="dim")

    for i, item in enumerate(items, 1):
        if show_index:
            table.add_row(str(i), item.get('name', ''), item.get('description', ''))
        else:
            table.add_row(item.get('name', ''), item.get('description', ''))

    return table


# ============================================================================
# MENU FORMATTING HELPERS
# ============================================================================
# Standard format: "Technical Term          Finnish description"
# Width: 24 chars for term, rest for description

def format_menu_item(term: str, description: str, width: int = 24) -> str:
    """
    Format a menu item with English term and Finnish description.

    Args:
        term: Technical term in English (e.g., "SLERP Merge")
        description: Description in Finnish
        width: Width for the term column

    Returns:
        Formatted string like "SLERP Merge             Yhdistä kaksi mallia"
    """
    return f"{term:<{width}}{description}"


def format_menu_separator(label: str = "", char: str = "─", width: int = 50) -> str:
    """
    Create a styled menu separator.

    Args:
        label: Optional label for the separator
        char: Character to use for the line
        width: Total width

    Returns:
        Formatted separator string
    """
    if label:
        # Center the label in the separator
        label_formatted = f" {label} "
        side_len = (width - len(label_formatted)) // 2
        return f"{char * side_len}{label_formatted}{char * side_len}"
    return char * width


def create_status_line(items: dict, separator: str = " │ ") -> str:
    """
    Create a status line showing multiple status items.

    Args:
        items: Dict of {label: (value, color)} or {label: value}
        separator: String to separate items

    Returns:
        Formatted status line
    """
    parts = []
    for label, value in items.items():
        if isinstance(value, tuple):
            val, color = value
            parts.append(f"[dim]{label}:[/dim] [{color}]{val}[/{color}]")
        else:
            parts.append(f"[dim]{label}:[/dim] [white]{value}[/white]")
    return separator.join(parts)


def print_section_header(title: str, subtitle: str = ""):
    """Print a styled section header."""
    console.print(f"\n[bold cyan]▸ {title}[/bold cyan]")
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def print_key_value(key: str, value: str, key_width: int = 20, indent: int = 2):
    """Print a key-value pair with consistent formatting."""
    spaces = " " * indent
    console.print(f"{spaces}[cyan]{key:<{key_width}}[/cyan] {value}")


def print_divider(style: str = "light"):
    """Print a horizontal divider."""
    chars = {
        "light": "─",
        "heavy": "━",
        "double": "═",
        "dotted": "┄",
    }
    char = chars.get(style, "─")
    console.print(f"[dim]{char * 56}[/dim]")


# ============================================================================
# MENU CHOICE BUILDERS (for questionary)
# ============================================================================

def build_menu_choices(
    items: list,
    back_label: str = "Palaa",
    back_value: str = "back",
    include_back: bool = True
) -> list:
    """
    Build questionary choices from a list of menu items.

    Args:
        items: List of dicts with keys:
            - term: English technical term
            - desc: Finnish description
            - value: Return value when selected
            - disabled: Optional, if True shows as disabled
            - separator: Optional, if True this is a separator (term = label)
        back_label: Label for back option
        back_value: Value for back option
        include_back: Whether to include back option

    Returns:
        List of questionary.Choice and questionary.Separator objects
    """
    import questionary

    choices = []

    for item in items:
        if item.get("separator"):
            # It's a separator
            label = item.get("term", "")
            if label:
                choices.append(questionary.Separator(f"── {label} ──"))
            else:
                choices.append(questionary.Separator())
        else:
            # Regular choice
            term = item.get("term", "")
            desc = item.get("desc", "")
            value = item.get("value", term.lower().replace(" ", "_"))
            disabled = item.get("disabled", False)

            title = format_menu_item(term, desc)

            if disabled:
                choices.append(questionary.Choice(
                    title=title,
                    value=value,
                    disabled=item.get("disabled_reason", "Ei saatavilla")
                ))
            else:
                choices.append(questionary.Choice(title=title, value=value))

    if include_back:
        choices.append(questionary.Separator())
        choices.append(questionary.Choice(
            title=format_menu_item(f"← {back_label}", ""),
            value=back_value
        ))

    return choices


def create_summary_panel(
    title: str,
    items: dict,
    footer: str = "",
    border_style: str = "yellow"
) -> Panel:
    """
    Create a summary panel for confirmations.

    Args:
        title: Panel title
        items: Dict of {label: value} pairs to display
        footer: Optional footer text
        border_style: Border color

    Returns:
        Rich Panel
    """
    lines = []
    for key, value in items.items():
        lines.append(f"[white]{key}:[/white] {value}")

    if footer:
        lines.append("")
        lines.append(f"[dim]{footer}[/dim]")

    return Panel(
        "\n".join(lines),
        title=f"[bold]{title}[/bold]",
        border_style=border_style,
        padding=(1, 2),
        box=box.ROUNDED
    )


def create_result_panel(
    success: bool,
    title: str,
    items: dict,
) -> Panel:
    """
    Create a result panel for operation outcomes.

    Args:
        success: Whether operation succeeded
        title: Panel title
        items: Dict of {label: value} pairs to display

    Returns:
        Rich Panel with appropriate styling
    """
    if success:
        icon = "[bold green]✓[/bold green]"
        border = "green"
        status = "[green]Valmis![/green]"
    else:
        icon = "[bold red]✗[/bold red]"
        border = "red"
        status = "[red]Epäonnistui[/red]"

    lines = [f"{icon} {status}", ""]
    for key, value in items.items():
        lines.append(f"[white]{key}:[/white] {value}")

    return Panel(
        "\n".join(lines),
        title=f"[bold]{title}[/bold]",
        border_style=border,
        padding=(1, 2),
        box=box.ROUNDED
    )


def create_model_card(model_info: dict) -> Panel:
    """Create a styled model info card."""
    content = Text()

    content.append("Model: ", style="cyan")
    content.append(f"{model_info.get('name', 'Unknown')}\n", style="bold white")

    if model_info.get('size'):
        content.append("Size: ", style="cyan")
        content.append(f"{model_info['size']}\n", style="white")

    if model_info.get('quantization'):
        content.append("Quantization: ", style="cyan")
        content.append(f"{model_info['quantization']}\n", style="yellow")

    if model_info.get('format'):
        content.append("Format: ", style="cyan")
        content.append(f"{model_info['format']}\n", style="green")

    if model_info.get('path'):
        content.append("Path: ", style="cyan")
        content.append(f"{model_info['path']}", style="dim")

    return Panel(
        content,
        title=f"[bold]{model_info.get('name', 'Model')}[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )


def create_info_panel(title: str, content: str, border_style: str = "cyan") -> Panel:
    """Create a styled information panel."""
    return Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style=border_style,
        padding=(1, 2),
    )


def print_success(message: str):
    """Print a success message."""
    console.print(f"[bold green][OK][/bold green] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[bold red][ERROR][/bold red] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[bold yellow][WARNING][/bold yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[bold cyan][INFO][/bold cyan] {message}")


def print_step(step: int, total: int, message: str):
    """Print a step progress message."""
    console.print(f"[bold blue][{step}/{total}][/bold blue] {message}")


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 0:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_number(num: int) -> str:
    """Format number with thousand separators."""
    return f"{num:,}"


def create_progress_bar() -> Progress:
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def create_spinner(description: str = "Working...") -> Progress:
    """Create a simple spinner for indeterminate progress."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def clear_screen():
    """Clear the terminal screen."""
    if sys.platform == 'win32':
        os.system('cls')
    else:
        os.system('clear')


# ==========================================
# Categorized Model Display Components
# ==========================================

# Category configuration for display
CATEGORY_CONFIG = {
    'safetensors': {
        'title': 'SafeTensors (Muunnettavat)',
        'icon': '🏠',
        'border_color': 'green',
        'description': 'HuggingFace-mallit, jotka voidaan muuntaa GGUF-muotoon'
    },
    'gguf_f16': {
        'title': 'GGUF F16 (Kvantisoitavat)',
        'icon': '📦',
        'border_color': 'yellow',
        'description': 'F16/F32 GGUF-mallit, jotka voidaan kvantisoida'
    },
    'gguf_quantized': {
        'title': 'GGUF Kvantisoidut',
        'icon': '⚡',
        'border_color': 'cyan',
        'description': 'Valmiit kvantisoidut mallit'
    },
    'merged': {
        'title': 'Yhdistetyt Mallit',
        'icon': '🔀',
        'border_color': 'magenta',
        'description': 'LoRA-yhdistetyt tai muuten yhdistellyt mallit'
    },
    'ollama': {
        'title': 'Ollama-mallit',
        'icon': '🤖',
        'border_color': 'blue',
        'description': 'Ollamassa kaynnistettavat mallit'
    },
    'adapter': {
        'title': 'LoRA-adapterit',
        'icon': '🔧',
        'border_color': 'white',
        'description': 'Koulutetut LoRA-adapterit'
    },
    'other': {
        'title': 'Muut',
        'icon': '📄',
        'border_color': 'dim',
        'description': 'Muut formaatit'
    }
}


def create_model_preview_card(model, show_path: bool = False) -> Panel:
    """
    Create a detailed preview card for a model.

    Args:
        model: ModelEntry object
        show_path: Whether to show full path

    Returns:
        Rich Panel with model details
    """
    # Guard against string being passed instead of ModelEntry
    if isinstance(model, str):
        return Panel(
            f"[yellow]Model ID:[/yellow] {model}\n[dim]Virhe: Odotettiin ModelEntry-objektia[/dim]",
            title="[bold]Mallin tiedot[/bold]",
            border_style="yellow",
            padding=(1, 2),
            box=box.ROUNDED
        )

    # Determine category icon
    cat_icons = {'base': '🏠', 'adapter': '🔧', 'merged': '🔀', 'ollama': '🤖'}
    icon = cat_icons.get(model.category, '📄')

    # Format details
    lines = [
        f"[bold white]{icon} {model.name}[/bold white]",
        "",
        f"[cyan]Formaatti:[/cyan]    {model.format.upper()}",
        f"[cyan]Koko:[/cyan]         {format_size(model.size_bytes)}",
    ]

    if model.quantization:
        lines.append(f"[cyan]Kvantisointi:[/cyan] [yellow]{model.quantization}[/yellow]")

    if model.source:
        lines.append(f"[cyan]Lähde:[/cyan]        {model.source}")

    if model.source_id:
        lines.append(f"[cyan]Lähde-ID:[/cyan]     [dim]{model.source_id}[/dim]")

    if model.added_date:
        date_str = model.added_date[:10]
        lines.append(f"[cyan]Lisätty:[/cyan]      {date_str}")

    if model.tags:
        lines.append(f"[cyan]Tagit:[/cyan]        {', '.join(model.tags)}")

    if show_path:
        lines.append("")
        lines.append(f"[dim]{model.path}[/dim]")

    content = "\n".join(lines)

    # Determine border color based on format
    if model.format == 'gguf':
        if model.quantization and model.quantization not in ('F16', 'F32', 'BF16'):
            border = 'cyan'
        else:
            border = 'yellow'
    elif model.format in ('safetensors', 'pytorch'):
        border = 'green'
    else:
        border = 'white'

    return Panel(
        content,
        title="[bold]Mallin tiedot[/bold]",
        border_style=border,
        padding=(1, 2),
        box=box.ROUNDED
    )


def format_model_choice_title(model, category_key: str = None, index: int = None, use_display_name: bool = True) -> str:
    """
    Format a two-line title for questionary Choice.

    Line 1: Icon + Display name (clean, readable)
    Line 2: Details (format, size, quantization, date)

    Args:
        model: ModelEntry object
        category_key: Category for icon selection
        index: Optional index number
        use_display_name: Use cleaned display name instead of full name

    Returns:
        Formatted two-line string for questionary
    """
    # Guard against string being passed instead of ModelEntry
    if isinstance(model, str):
        if index is not None:
            return f"📄 [{index}] {model}\n     [Virhe: merkkijono ModelEntryn sijaan]"
        return f"📄 {model}\n     [Virhe: merkkijono ModelEntryn sijaan]"

    # Get icon
    if category_key:
        config = CATEGORY_CONFIG.get(category_key, CATEGORY_CONFIG['other'])
        icon = config['icon']
    else:
        cat_icons = {'base': '🏠', 'adapter': '🔧', 'merged': '🔀', 'ollama': '🤖'}
        icon = cat_icons.get(model.category, '📄')

    # Get display name (clean version for readability)
    if use_display_name:
        from ..models.library import format_display_name
        display_name = format_display_name(model.name, max_length=45)
    else:
        display_name = model.name if len(model.name) <= 45 else model.name[:42] + "..."

    # Line 1: Display name
    if index is not None:
        line1 = f"{icon} [{index:>2}] {display_name}"
    else:
        line1 = f"{icon} {display_name}"

    # Line 2: Details
    size_str = format_size(model.size_bytes)
    format_str = model.format.upper()
    quant_str = f"[{model.quantization}]" if model.quantization else ""
    date_str = model.added_date[:10] if model.added_date else ""

    # Build detail line with aligned columns
    details = f"       {format_str:<6} {quant_str:<10} {size_str:>10}  {date_str}"

    return f"{line1}\n{details}"


def build_model_choices(models_by_category: dict, include_back: bool = True) -> list:
    """
    Build questionary choices from categorized models.

    Two-line format per model, with category separators.
    NO duplicate display - this IS the display.

    Args:
        models_by_category: Dict of {category_key: [ModelEntry, ...]}
        include_back: Whether to include Back option

    Returns:
        List of questionary.Choice and questionary.Separator objects
    """
    import questionary

    choices = []
    total = sum(len(models) for models in models_by_category.values())

    if total == 0:
        return choices

    idx = 1
    for category_key, models in models_by_category.items():
        if not models:
            continue

        config = CATEGORY_CONFIG.get(category_key, CATEGORY_CONFIG['other'])
        icon = config['icon']
        title = config['title']

        # Category separator with count
        choices.append(questionary.Separator(
            f"\n{'─' * 60}\n   {icon} {title} ({len(models)})\n{'─' * 60}"
        ))

        for model in models:
            # Two-line title
            choice_title = format_model_choice_title(model, category_key, idx)
            choices.append(questionary.Choice(title=choice_title, value=model))
            idx += 1

    if include_back:
        choices.append(questionary.Separator("\n" + "─" * 60))
        choices.append(questionary.Choice(title="⬅️  Palaa / Back", value=None))

    return choices


def create_model_section_panel(
    category_key: str,
    models: list,
    show_details: bool = True
) -> Panel:
    """
    Create a Rich Panel for a category of models.

    Args:
        category_key: Key from CATEGORY_CONFIG
        models: List of ModelEntry objects
        show_details: Whether to show size and quantization

    Returns:
        Rich Panel with formatted model list
    """
    config = CATEGORY_CONFIG.get(category_key, CATEGORY_CONFIG['other'])
    icon = config['icon']
    title = config['title']
    border_color = config['border_color']

    if not models:
        content = "[dim]Ei malleja[/dim]"
    else:
        lines = []
        for model in models:
            # Format model line
            name = model.name[:35] + ".." if len(model.name) > 37 else model.name
            size_str = format_size(model.size_bytes)

            if show_details:
                quant = f"[{model.quantization}]" if model.quantization else ""
                line = f"  {icon} {name:<38} [dim]{quant:>10}[/dim] [cyan]{size_str:>10}[/cyan]"
            else:
                line = f"  {icon} {name}"
            lines.append(line)

        content = "\n".join(lines)

    return Panel(
        content,
        title=f"[bold]{icon} {title} ({len(models)})[/bold]",
        border_style=border_color,
        padding=(0, 1),
        box=box.ROUNDED
    )


def display_categorized_library(
    grouped_models: dict,
    categories_to_show: list = None
):
    """
    Display models grouped by category as panels.

    Args:
        grouped_models: Dict from library.get_models_grouped_by_format()
        categories_to_show: List of category keys to show (None = all non-empty)
    """
    console.print("\n[bold cyan]=== KIRJASTO ===[/bold cyan]\n")

    # Default: show all non-empty categories
    if categories_to_show is None:
        categories_to_show = [
            'safetensors', 'gguf_f16', 'gguf_quantized',
            'merged', 'ollama', 'adapter', 'other'
        ]

    total_count = 0
    total_size = 0

    for category in categories_to_show:
        models = grouped_models.get(category, [])
        if models:
            panel = create_model_section_panel(category, models)
            console.print(panel)
            console.print()  # Add spacing between panels

            total_count += len(models)
            total_size += sum(m.size_bytes for m in models)

    # Print summary
    if total_count > 0:
        console.print(
            f"[dim]Yhteensa: {total_count} mallia, {format_size(total_size)}[/dim]\n"
        )
    else:
        console.print("[yellow]Kirjasto on tyhja.[/yellow]\n")


def display_tool_model_selection(
    models: list,
    category_key: str,
    title: str = None
) -> Panel:
    """
    Create a panel for tool-specific model selection.

    Args:
        models: List of ModelEntry objects
        category_key: Category for styling
        title: Optional custom title

    Returns:
        Rich Panel with numbered model list for selection
    """
    config = CATEGORY_CONFIG.get(category_key, CATEGORY_CONFIG['other'])
    icon = config['icon']
    display_title = title or config['title']
    border_color = config['border_color']

    if not models:
        content = "[dim]Ei malleja saatavilla[/dim]"
    else:
        lines = []
        for i, model in enumerate(models, 1):
            name = model.name[:32] + ".." if len(model.name) > 34 else model.name
            size_str = format_size(model.size_bytes)
            quant = f"[{model.quantization}]" if model.quantization else ""
            line = f"  [yellow]{i:>2}.[/yellow] {icon} {name:<35} [dim]{quant:>10}[/dim] [cyan]{size_str:>8}[/cyan]"
            lines.append(line)

        content = "\n".join(lines)

    return Panel(
        content,
        title=f"[bold]{icon} {display_title}[/bold]",
        border_style=border_color,
        padding=(0, 1),
        box=box.ROUNDED
    )
