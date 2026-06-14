class MlopsAgentError(Exception):
    """Base exception for project-specific errors."""


class NotebookFileNotFoundError(MlopsAgentError):
    """Raised when a notebook file cannot be found."""


class NotebookParsingError(MlopsAgentError):
    """Raised when notebook parsing fails."""


class LLMClientError(MlopsAgentError):
    """Raised when LLM communication fails."""
