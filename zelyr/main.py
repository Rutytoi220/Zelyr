f"""Zelyr main entry point and initialization."""

from zelyr.core import Engine
from zelyr.llm import OllamaClient
from zelyr.api.cli import CLI, run_cli
from zelyr.config import ZelyrConfig, get_config, set_config

__version__ = "1.0.0"
__author__ = "Zelyr Team"

__all__ = [
    "Engine",
    "OllamaClient",
    "CLI",
    "run_cli",
    "ZelyrConfig",
    "get_config",
    "set_config",
    "__version__",
]


def main():
    """Run Zelyr CLI."""
    run_cli()


if __name__ == "__main__":
    main()
