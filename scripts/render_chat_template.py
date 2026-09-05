#!/usr/bin/env python3
"""Adapt the immutable upstream template to the deployment's thinking-off API."""
from pathlib import Path
import sys

UPSTREAM = "<|assistant|>{{- '<think>' -}}"
COMPATIBLE = "<|assistant|>{{- '<think>' if enable_thinking is not defined or enable_thinking else '<think></think>' -}}"


def adapt(source: str) -> str:
    if source.count(UPSTREAM) != 1 or not source.rstrip().endswith(UPSTREAM + "\n{%- endif -%}"):
        raise ValueError("Unexpected upstream generation prompt; review template compatibility")
    return source.replace(UPSTREAM, COMPATIBLE)


if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    rendered = adapt(source.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(rendered)
    temporary.replace(destination)
