from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Button, Static, DataTable, Log, Label, Markdown,
    Input, Select,
)
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual import on, work

import psutil
from rich.markdown import Markdown as RichMarkdown

# Tuodaan olemassa olevat logiikat (käytetään relatiivisia importteja)
from ..models.library import ModelLibrary
from ..integrations.ollama import OllamaManager


class ConfirmScreen(ModalScreen[bool]):
    """Yksinkertainen kyllä/ei-vahvistusdialogi."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog"):
            yield Label(self.message, classes="modal-message")
            with Horizontal(classes="modal-buttons"):
                yield Button("Kyllä", id="confirm-yes", variant="error")
                yield Button("Peruuta", id="confirm-no")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "confirm-yes")


class CreateModelScreen(ModalScreen):
    """Lomake Ollama-mallin luontiin kirjaston GGUF-tiedostosta."""

    def __init__(self) -> None:
        super().__init__()
        try:
            library = ModelLibrary()
            self.gguf_models = [
                m for m in library.list_models() if m.format == "gguf"
            ]
        except Exception:
            self.gguf_models = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog"):
            yield Label("Luo Ollama-malli GGUF-tiedostosta", classes="modal-message")
            yield Input(placeholder="Mallin nimi (esim. my-model)", id="create-name")
            if self.gguf_models:
                yield Select(
                    [(m.name, m.id) for m in self.gguf_models],
                    prompt="Valitse GGUF-malli kirjastosta",
                    id="create-gguf",
                )
            else:
                yield Label(
                    "Kirjastossa ei ole GGUF-malleja - lisää ensin CLI:llä (ai-toolbox)",
                    classes="modal-warning",
                )
            with Horizontal(classes="modal-buttons"):
                yield Button("Luo", id="create-confirm", variant="primary")
                yield Button("Peruuta", id="create-cancel")

    @on(Button.Pressed, "#create-cancel")
    def handle_cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#create-confirm")
    def handle_confirm(self, event: Button.Pressed) -> None:
        event.stop()
        name = self.query_one("#create-name", Input).value.strip()
        if not name or not self.gguf_models:
            self.dismiss(None)
            return

        select = self.query_one("#create-gguf", Select)
        if select.value is Select.BLANK:
            return  # Ei valintaa - pidä dialogi auki

        model = next((m for m in self.gguf_models if m.id == select.value), None)
        if model is None:
            self.dismiss(None)
            return

        self.dismiss({"name": name, "gguf_path": str(model.path)})

class Sidebar(Static):
    """Sivupalkki navigaatiolle."""
    def compose(self) -> ComposeResult:
        yield Label("AI TOOLBOX", classes="sidebar-title")
        yield Button("🏠 Dashboard", id="view-dashboard", variant="primary")
        yield Button("📚 Library", id="view-library")
        yield Button("🦙 Ollama Manager", id="view-ollama")
        yield Static(classes="spacer")
        yield Button("❌ Exit", id="exit-app")

class DashboardView(Container):
    """Etusivun näkymä."""
    
    def compose(self) -> ComposeResult:
        with Horizontal(id="stats-container"):
            yield Static("CPU: 0%", classes="stat-card", id="stat-cpu")
            yield Static("RAM: 0GB", classes="stat-card", id="stat-ram")
            yield Static("Models: 0", classes="stat-card", id="stat-models")
        
        welcome_md = """
# Welcome to AI Toolbox

Select a tool from the sidebar to get started.

- **Library**: Manage your local GGUF and PyTorch models
- **Ollama**: Manage your Ollama service and models
- **Training, merging & abliteration**: use the full CLI (`ai-toolbox`)
        """
        yield Markdown(welcome_md, classes="welcome-text")

class LibraryView(Container):
    """Kirjastonäkymä split-screenillä."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="library-container"):
            with Vertical(id="model-list"):
                yield DataTable()
            
            with Vertical(id="model-details"):
                yield Label("Model Details", classes="details-title")
                yield Static("Select a model to view details", id="details-content")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Name", "Category", "Format", "Size", "Quant")
        table.cursor_type = "row"
        self.load_models()

    def load_models(self):
        table = self.query_one(DataTable)
        table.clear()

        # Category icons for display
        category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}

        try:
            library = ModelLibrary()
            models = library.list_models()
            for model in models:
                size_gb = f"{model.size_bytes / (1024**3):.1f} GB"
                cat_icon = category_icons.get(model.category, "📄")
                # Tallennetaan model.id rivin avaimeksi (row key)
                table.add_row(
                    model.name,
                    f"{cat_icon} {model.category}",
                    model.format.upper(),
                    size_gb,
                    model.quantization or "-",
                    key=model.id
                )

            self.app.model_count = len(models)

        except Exception as e:
            table.add_row("Error loading models", str(e), "", "", "")

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None: 
        """Kun rivi valitaan, näytä tiedot oikealla."""
        model_id = event.row_key.value
        self.show_details(model_id)

    def show_details(self, model_id):
        try:
            library = ModelLibrary()
            model = library.get_model(model_id)
            if not model:
                return

            # Category icon
            category_icons = {"base": "🏠", "adapter": "🔧", "merged": "🔀", "ollama": "🤖"}
            category_icon = category_icons.get(model.category, "📄")

            # Parent info
            parent_name = "None"
            if model.parent_id:
                parent = library.get_model(model.parent_id)
                if parent:
                    parent_name = parent.name

            # Children count
            children = library.get_children(model.id)
            children_str = f"{len(children)} models" if children else "None"

            details = f"""
{category_icon} **{model.name}**

**Category:** {model.category.upper()}
**Format:** {model.format}
**Size:** {format_size(model.size_bytes)}
**Path:** {model.path}
**Added:** {model.added_date[:10]}
**Source:** {model.source}

---
**Quantization:** {model.quantization or 'N/A'}
**Tags:** {', '.join(model.tags) if model.tags else 'None'}

---
**Parent:** {parent_name}
**Children:** {children_str}
            """

            # Add training info if adapter
            if model.training_info:
                info = model.training_info
                details += f"""
---
**Training Info:**
- Backend: {info.get('backend', 'unknown')}
- Base model: {info.get('base_model', 'unknown')}
"""

            # Add ollama info if ollama
            if model.ollama_info:
                info = model.ollama_info
                details += f"""
---
**Ollama Info:**
- Name: {info.get('ollama_name', 'unknown')}
- Template: {info.get('template', 'none')}
"""

            # Rich-Markdown renderöitäväksi - textual.widgets.Markdown on
            # widget eikä kelpaa Static.update():lle
            self.query_one("#details-content", Static).update(RichMarkdown(details))

        except Exception as e:
            self.query_one("#details-content", Static).update(f"Error: {e}")

class OllamaView(Container):
    """Ollama hallinta - käyttää OllamaManageria."""

    def compose(self) -> ComposeResult:
        yield Static("Ollama Status: Checking...", id="ollama-status")
        yield DataTable(id="ollama-table")
        with Horizontal(id="ollama-buttons"):
            yield Button("🔄 Refresh", id="refresh-ollama")
            yield Button("➕ Create from GGUF", id="create-ollama-model")
            yield Button("🗑️ Delete Model", id="delete-ollama-model", variant="error")

    def on_mount(self) -> None:
        self.manager = OllamaManager()
        self.check_status()

        table = self.query_one("#ollama-table", DataTable)
        table.add_columns("Model", "Size", "Modified")
        table.cursor_type = "row"

    def check_status(self):
        status_widget = self.query_one("#ollama-status", Static)

        if self.manager.is_available():
            status_widget.update("🟢 OLLAMA ONLINE")
            status_widget.classes = "status-online"
            self.load_models()
        else:
            status_widget.update("🔴 OLLAMA OFFLINE - Is it running?")
            status_widget.classes = "status-offline"

    def load_models(self):
        table = self.query_one("#ollama-table", DataTable)
        table.clear()

        try:
            models = self.manager.list_models()
            for model in models:
                table.add_row(
                    model.name,
                    model.size,
                    model.modified,
                    key=model.name
                )
        except Exception as e:
            table.add_row("Error", str(e), "")

    def _selected_model_name(self):
        """Palauta valitun rivin mallinimi tai None."""
        table = self.query_one("#ollama-table", DataTable)
        if table.row_count == 0:
            return None
        try:
            cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
            return cell_key.row_key.value
        except Exception:
            return None

    @on(Button.Pressed, "#refresh-ollama")
    def handle_refresh(self, event: Button.Pressed) -> None:
        event.stop()
        self.check_status()
        self.app.log_message("Ollama: lista päivitetty")

    @on(Button.Pressed, "#delete-ollama-model")
    def handle_delete(self, event: Button.Pressed) -> None:
        event.stop()
        model_name = self._selected_model_name()
        if not model_name:
            self.app.log_message("Ollama: valitse ensin malli taulukosta")
            return

        def on_confirm(confirmed) -> None:
            if not confirmed:
                return
            success, msg = self.manager.delete_model(model_name)
            self.app.log_message(f"Ollama: {msg}")
            if success:
                self.load_models()

        self.app.push_screen(
            ConfirmScreen(f"Poistetaanko Ollama-malli '{model_name}'?"),
            on_confirm,
        )

    @on(Button.Pressed, "#create-ollama-model")
    def handle_create(self, event: Button.Pressed) -> None:
        event.stop()

        def on_result(result) -> None:
            if not result:
                return
            self.app.log_message(f"Ollama: luodaan mallia '{result['name']}'...")
            self._create_model(result["name"], result["gguf_path"])

        self.app.push_screen(CreateModelScreen(), on_result)

    @work(thread=True, exclusive=True)
    def _create_model(self, name: str, gguf_path: str) -> None:
        """Luo malli taustasäikeessä - ollama create voi kestää minuutteja."""
        success, msg = self.manager.create_model(name, gguf_path)
        self.app.call_from_thread(self._on_create_done, success, msg)

    def _on_create_done(self, success: bool, msg: str) -> None:
        self.app.log_message(f"Ollama: {msg.splitlines()[0]}")
        if success:
            self.load_models()

class ToolboxApp(App):
    """Pääsovellus."""
    
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]
    
    model_count = reactive(0)

    def compose(self) -> ComposeResult:
        """Luo käyttöliittymän rakenne."""
        yield Header(show_clock=True)
        
        with Horizontal():
            yield Sidebar(id="sidebar")
            
            with Vertical(id="main-content"):
                yield DashboardView(id="view-dashboard-container")
                yield Log(id="activity-log")
        
        yield Footer()

    def on_mount(self) -> None:
        """Kun sovellus käynnistyy."""
        self.log_message("System started. Welcome!")
        self.update_system_stats()
        self.set_interval(2, self.update_system_stats)

    def log_message(self, message: str):
        """Kirjoita lokiin."""
        try:
            log = self.query_one(Log)
            log.write_line(message)
        except:
            pass

    def update_system_stats(self):
        """Päivitä järjestelmätiedot."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_gb = ram.used / (1024**3)
        total_ram_gb = ram.total / (1024**3)
        
        try:
            if self.query_one("#view-dashboard-container"):
                self.query_one("#stat-cpu", Static).update(f"[b]{cpu}%[/b]\nCPU Usage")
                self.query_one("#stat-ram", Static).update(f"[b]{ram_gb:.1f} GB[/b]\nRAM Usage")
                self.query_one("#stat-models", Static).update(f"[b]{self.model_count}[/b]\nModels")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Käsittele napin painallukset."""
        button_id = event.button.id
        
        # Päivitä aktiivinen nappi tyylittelyä varten
        for btn in self.query("Sidebar Button"):
            btn.remove_class("-active")
        event.button.add_class("-active")

        if button_id == "exit-app":
            self.exit()
            
        elif button_id == "view-library":
            self.switch_view(LibraryView())
            self.log_message("Switched to Library view")
            
        elif button_id == "view-dashboard":
            self.switch_view(DashboardView())
            self.log_message("Switched to Dashboard view")

        elif button_id == "view-ollama":
            self.switch_view(OllamaView())
            self.log_message("Switched to Ollama Manager")

    def switch_view(self, new_widget):
        """Vaihda päänäkymä."""
        content = self.query_one("#main-content")
        # Poista vanha view (ensimmäinen lapsi, mutta jätä Log)
        old_views = content.children[:-1] # Log on viimeinen
        for view in old_views:
            view.remove()
            
        # Lisää uusi view ennen Logia
        content.mount(new_widget, before="#activity-log")

def format_size(size_bytes):
    return f"{size_bytes / (1024**3):.1f} GB"

if __name__ == "__main__":
    app = ToolboxApp()
    app.run()