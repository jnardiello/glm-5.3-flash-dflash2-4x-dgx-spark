#!/usr/bin/env python3
"""Inspect a flat Hugging Face snapshot against a pinned SHA-256 manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != 1:
        raise ValueError(f"{path}: unsupported manifest schema")
    revision = manifest.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{path}: revision is not a full lowercase commit SHA")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{path}: files must be a non-empty list")
    seen: set[str] = set()
    for item in files:
        name = item.get("path", "")
        parts = Path(name).parts
        if (
            not name
            or name.startswith("/")
            or "\t" in name
            or "\n" in name
            or ".." in parts
            or name in seen
        ):
            raise ValueError(f"{path}: unsafe or duplicate file path {name!r}")
        seen.add(name)
        size = item.get("size")
        digest = item.get("sha256", "")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"{path}: invalid size for {name}")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{path}: invalid sha256 for {name}")
    return manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(manifest: dict, names: list[str] | None) -> list[dict]:
    files = manifest["files"]
    if names is None:
        return files
    by_name = {item["path"]: item for item in files}
    unknown = sorted(set(names) - by_name.keys())
    if unknown:
        raise ValueError("paths absent from manifest: " + ", ".join(unknown))
    wanted = set(names)
    return [item for item in files if item["path"] in wanted]


def inspect(directory: Path, files: list[dict]) -> list[tuple[str, int, str]]:
    problems: list[tuple[str, int, str]] = []
    for item in files:
        name, expected_size = item["path"], item["size"]
        candidate = directory / name
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            problems.append(("MISSING", expected_size, name))
            continue
        if not candidate.is_file() or stat.st_size != expected_size:
            problems.append(("SIZE", expected_size, name))
            continue
        if sha256(candidate) != item["sha256"]:
            problems.append(("SHA256", expected_size, name))
    return problems


def print_problems(problems: list[tuple[str, int, str]], stream=sys.stdout) -> None:
    for state, size, name in problems:
        print(f"{state}\t{size}\t{name}", file=stream)


def command_plan(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    print_problems(inspect(args.directory, manifest["files"]))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    names = None
    if args.paths_from_stdin:
        names = [line.rstrip("\n") for line in sys.stdin if line.rstrip("\n")]
    files = select_files(manifest, names)
    problems = inspect(args.directory, files)
    if problems:
        print_problems(problems, sys.stderr)
        return 1
    total = sum(item["size"] for item in files)
    print(f"OK\t{len(files)}\t{total}")
    return 0


def command_field(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if args.field == "revision":
        print(manifest["revision"])
    elif args.field == "repository":
        print(manifest["repository"])
    elif args.field == "file-count":
        print(len(manifest["files"]))
    elif args.field == "total-size":
        print(sum(item["size"] for item in manifest["files"]))
    elif args.field == "shard-count":
        print(sum(item["path"].endswith(".safetensors") and item["path"].startswith("model-") for item in manifest["files"]))
    elif args.field in {"sha256", "size"}:
        if not args.file:
            raise ValueError(f"field {args.field} requires --file")
        item = next((item for item in manifest["files"] if item["path"] == args.file), None)
        if item is None:
            raise ValueError(f"{args.file}: absent from manifest")
        print(item[args.field])
    return 0


def command_exclude_check(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    patterns = [pattern for pattern in args.patterns.split(",") if pattern]
    matches = sorted(
        item["path"]
        for item in manifest["files"]
        if any(fnmatch.fnmatchcase(item["path"], pattern) for pattern in patterns)
    )
    if matches:
        print("HF_EXCLUDE matches required manifest files: " + ", ".join(matches), file=sys.stderr)
        return 1
    print("OK")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="print missing/corrupt files as STATE<TAB>SIZE<TAB>PATH")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("directory", type=Path)
    plan.set_defaults(func=command_plan)
    verify = sub.add_parser("verify", help="verify the whole manifest or paths read from stdin")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("directory", type=Path)
    verify.add_argument("--paths-from-stdin", action="store_true")
    verify.set_defaults(func=command_verify)
    field = sub.add_parser("field", help="print one manifest value")
    field.add_argument("manifest", type=Path)
    field.add_argument("field", choices=("revision", "repository", "file-count", "total-size", "shard-count", "sha256", "size"))
    field.add_argument("--file")
    field.set_defaults(func=command_field)
    exclude = sub.add_parser("exclude-check", help="reject excludes that match required files")
    exclude.add_argument("manifest", type=Path)
    exclude.add_argument("patterns")
    exclude.set_defaults(func=command_exclude_check)
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"model-manifest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
