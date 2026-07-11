"""Componentes reutilizables del proyecto de detección de fraude."""

from __future__ import annotations

from .config import PROJECT_ROOT, RANDOM_STATE

__all__ = ["PROJECT_ROOT", "RANDOM_STATE", "FraudScoringPipeline"]
__version__ = "0.3.0"


def __getattr__(name: str):
    """Carga objetos pesados únicamente cuando son solicitados."""
    if name == "FraudScoringPipeline":
        from .modeling import FraudScoringPipeline

        return FraudScoringPipeline
    raise AttributeError(name)
