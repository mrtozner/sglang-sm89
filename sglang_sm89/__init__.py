"""sglang-sm89: Ada Lovelace (sm_89) SGLang speculative & constrained decoding acceleration."""

from __future__ import annotations

__version__ = "0.3.1"

from .plugin import register

__all__ = ["__version__", "register"]
