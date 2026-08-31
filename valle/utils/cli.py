"""Command-line helpers shared by VioLM entry points.

Public options use kebab-case (``--output-dir``). Older MIDI-VALLE commands
often used underscores; they remain accepted through a compatibility
normalizer so existing experiment scripts do not fail abruptly.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence


class ConsistentArgumentParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` with backwards-compatible option spelling."""

    _warned_legacy_options: set[tuple[str, str]] = set()

    @classmethod
    def _normalize_legacy_options(cls, args: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for token in args:
            if token.startswith("--"):
                option, separator, value = token.partition("=")
                canonical = option.replace("_", "-")
                if canonical != option:
                    pair = (option, canonical)
                    if pair not in cls._warned_legacy_options:
                        warnings.warn(
                            f"{option} is deprecated; use {canonical}.",
                            FutureWarning,
                            stacklevel=3,
                        )
                        cls._warned_legacy_options.add(pair)
                    token = canonical + (separator + value if separator else "")
            normalized.append(token)
        return normalized

    def parse_known_args(self, args=None, namespace=None):
        raw_args = sys.argv[1:] if args is None else list(args)
        return super().parse_known_args(
            self._normalize_legacy_options(raw_args), namespace
        )
