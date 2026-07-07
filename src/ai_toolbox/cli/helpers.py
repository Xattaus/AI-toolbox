"""
AI TOOLBOX - CLI Helpers
========================

Reusable menu builders and wizard helpers.
"""

from typing import Any, Callable, List, Optional

import questionary
from questionary import Style
from ..core.ui import console


# Questionary style used throughout the app
MENU_STYLE = Style([
    ('qmark', 'fg:orange bold'),
    ('question', 'bold'),
    ('answer', 'fg:orange bold'),
    ('pointer', 'fg:orange bold'),
    ('highlighted', 'fg:orange bold'),
    ('selected', 'fg:green'),
])


def create_menu_choices(items: list, back_label: str = "Back") -> list:
    """Create standardized menu choices with separator and back option."""
    choices = items.copy()
    choices.append(questionary.Separator())
    choices.append(questionary.Choice(f"<- {back_label}", "back"))
    return choices


def run_menu(title: str, choices: list, style=None) -> str:
    """Run a menu and return the selection."""
    from ..core.ui import print_mini_banner
    print_mini_banner(title)

    result = questionary.select(
        "Select an option:",
        choices=choices,
        style=style or MENU_STYLE,
    ).ask()

    return result


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for confirmation."""
    return questionary.confirm(message, default=default, style=MENU_STYLE).ask()


def prompt_text(message: str, default: str = "", validate=None) -> str:
    """Prompt for text input."""
    return questionary.text(message, default=default, validate=validate, style=MENU_STYLE).ask()


def prompt_path(message: str, default: str = "") -> str:
    """Prompt for a file/directory path."""
    return questionary.path(message, default=default, style=MENU_STYLE).ask()


def prompt_select(message: str, choices: list, style=None) -> Any:
    """Prompt for a selection from choices."""
    return questionary.select(
        message,
        choices=choices,
        style=style or MENU_STYLE,
        qmark=">>",
        pointer=">"
    ).ask()


def prompt_checkbox(message: str, choices: list, style=None) -> List[Any]:
    """Prompt for multiple selections from choices."""
    return questionary.checkbox(
        message,
        choices=choices,
        style=style or MENU_STYLE,
    ).ask()


def prompt_number(message: str, default: str = "", min_value: Optional[float] = None,
                  max_value: Optional[float] = None) -> Optional[str]:
    """
    Prompt for a numeric input with optional range validation.

    Args:
        message: The prompt message
        default: Default value as string
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)

    Returns:
        The entered value as string, or None if cancelled
    """
    def validate_number(text):
        if not text:
            return True  # Allow empty for default
        try:
            val = float(text)
            if min_value is not None and val < min_value:
                return f"Value must be >= {min_value}"
            if max_value is not None and val > max_value:
                return f"Value must be <= {max_value}"
            return True
        except ValueError:
            return "Please enter a valid number"

    return questionary.text(
        message,
        default=default,
        validate=validate_number,
        style=MENU_STYLE
    ).ask()


def press_any_key(message: str = "Press any key to continue..."):
    """Wait for user to press any key."""
    questionary.press_any_key_to_continue(message, style=MENU_STYLE).ask()


def run_wizard_step(
    step: int,
    total: int,
    title: str,
    action: Callable,
    skip_condition: Callable = None
) -> tuple[bool, Any]:
    """
    Run a single wizard step with step counter display.

    Args:
        step: Current step number
        total: Total number of steps
        title: Step title to display
        action: Function to execute for this step
        skip_condition: Optional function that returns True if step should be skipped

    Returns:
        Tuple of (success, result) where success is False if user cancelled
    """
    if skip_condition and skip_condition():
        return True, None

    console.print(f"\n[bold blue][{step}/{total}][/bold blue] {title}")

    try:
        result = action()
        if result is None:
            return False, None
        return True, result
    except KeyboardInterrupt:
        return False, None


class WizardContext:
    """Context manager for multi-step wizards."""

    def __init__(self, title: str, total_steps: int):
        """
        Initialize wizard context.

        Args:
            title: Wizard title
            total_steps: Total number of steps
        """
        self.title = title
        self.total_steps = total_steps
        self.current_step = 0
        self.results = {}
        self.cancelled = False

    def __enter__(self):
        from ..core.ui import print_mini_banner
        print_mini_banner(self.title)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            self.cancelled = True
            return True  # Suppress the exception
        return False

    def step(self, name: str, action: Callable, description: str = "") -> Any:
        """
        Execute a wizard step.

        Args:
            name: Step name/key for storing result
            action: Function to execute
            description: Optional description to display

        Returns:
            Result of the action, or None if cancelled
        """
        if self.cancelled:
            return None

        self.current_step += 1

        step_text = f"[bold blue][{self.current_step}/{self.total_steps}][/bold blue]"
        if description:
            console.print(f"\n{step_text} {description}")

        try:
            result = action()
            if result is None:
                self.cancelled = True
                return None
            self.results[name] = result
            return result
        except KeyboardInterrupt:
            self.cancelled = True
            return None

    def get_result(self, name: str) -> Any:
        """Get a stored result by name."""
        return self.results.get(name)

    @property
    def is_cancelled(self) -> bool:
        """Check if wizard was cancelled."""
        return self.cancelled


def create_choice(title: str, value: Any, disabled: str = None) -> questionary.Choice:
    """
    Create a questionary Choice with consistent formatting.

    Args:
        title: Display text
        value: Value returned when selected
        disabled: If set, shows as disabled with this message

    Returns:
        questionary.Choice instance
    """
    return questionary.Choice(title=title, value=value, disabled=disabled)


def create_separator(text: str = "") -> questionary.Separator:
    """
    Create a questionary Separator with optional text.

    Args:
        text: Optional separator text

    Returns:
        questionary.Separator instance
    """
    if text:
        return questionary.Separator(f"-- {text} --")
    return questionary.Separator()
