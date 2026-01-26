"""
AI TOOLBOX - Custom Exceptions
==============================

Custom exception classes for the toolbox.
"""


class ToolboxError(Exception):
    """Base exception for all AI Toolbox errors."""
    pass


class ModelError(ToolboxError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a model cannot be found."""
    pass


class ModelDownloadError(ModelError):
    """Raised when model download fails."""
    pass


class ModelConversionError(ModelError):
    """Raised when model conversion fails."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class TrainingError(ToolboxError):
    """Base exception for training-related errors."""
    pass


class DatasetError(TrainingError):
    """Raised when there's a dataset issue."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found."""
    pass


class DatasetFormatError(DatasetError):
    """Raised when dataset format is invalid."""
    pass


class TrainingConfigError(TrainingError):
    """Raised when training configuration is invalid."""
    pass


class ConversionError(ToolboxError):
    """Base exception for conversion errors."""
    pass


class LlamaCppNotFoundError(ConversionError):
    """Raised when llama.cpp binaries are not found."""
    pass


class QuantizationError(ConversionError):
    """Raised when quantization fails."""
    pass


class UnsupportedModelFormatError(ConversionError):
    """Raised when model format is not supported."""
    pass


class ConfigError(ToolboxError):
    """Base exception for configuration errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Raised when configuration file is not found."""
    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


class DependencyError(ToolboxError):
    """Base exception for dependency errors."""
    pass


class MissingDependencyError(DependencyError):
    """Raised when a required dependency is missing."""

    def __init__(self, dependency: str, install_hint: str = None):
        self.dependency = dependency
        self.install_hint = install_hint
        message = f"Missing dependency: {dependency}"
        if install_hint:
            message += f". Install with: {install_hint}"
        super().__init__(message)


class IntegrationError(ToolboxError):
    """Base exception for integration errors."""
    pass


class OllamaError(IntegrationError):
    """Raised when Ollama integration fails."""
    pass


class HuggingFaceError(IntegrationError):
    """Raised when HuggingFace integration fails."""
    pass


# =============================================================================
# Merge Errors
# =============================================================================

class MergeError(ModelError):
    """Base exception for merge-related errors."""
    pass


class VocabSizeMismatchError(MergeError):
    """Raised when vocab sizes don't match and can't be reconciled."""

    def __init__(self, vocab1: int, vocab2: int, max_diff: int = 1000):
        self.vocab1 = vocab1
        self.vocab2 = vocab2
        self.diff = abs(vocab1 - vocab2)
        message = f"Vocab size mismatch: {vocab1} vs {vocab2} (diff: {self.diff})"
        if self.diff > max_diff:
            message += f". Difference exceeds maximum allowed ({max_diff})"
        super().__init__(message)


class PrecisionMismatchError(MergeError):
    """Raised when model precisions don't match."""

    def __init__(self, dtype1: str, dtype2: str):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        message = f"Precision mismatch: {dtype1} vs {dtype2}"
        super().__init__(message)


class IncompatibleArchitectureError(MergeError):
    """Raised when model architectures are incompatible for merging."""

    def __init__(self, arch1: str, arch2: str):
        self.arch1 = arch1
        self.arch2 = arch2
        message = f"Incompatible architectures: {arch1} vs {arch2}"
        super().__init__(message)
