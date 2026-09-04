#!/usr/bin/env python3
"""Merge several fused-MoE Triton tuning JSONs into one.

    python3 merge-configs.py [--force] [--only-keys 16,24,32,48] [--out PATH] \\
        OUT.json IN1.json IN2.json [...]

v2 (2026-09-04), selective mode — active when --only-keys and/or --out is given:
- OUT.json, if it exists, is loaded as the BASE (its triton_version and every
  batch-size key), so a promoted hybrid file can receive a few re-tuned keys with
  everything else untouched;
- --only-keys K1,K2,... takes only those batch-size keys from the inputs (an input
  key outside the list is ignored and reported); every listed key must be found;
- --out PATH writes the result to PATH instead of OUT.json (PATH must not exist
  unless --force), e.g. "v2-mid-E=288,...,block_shape=[128,128].json" as an A/B
  candidate next to the promoted file, which stays byte-identical (keep the .json
  suffix: scripts/deploy.sh pushes node/moe-configs/*.json only).
Without those two flags the behaviour is the original one below.

Why: benchmark_moe.py names its output from (E, N, dtype, block_shape) only —
NOT from the batch sizes it tuned. Two runs (decode set, prefill set) therefore
write the SAME filename in their own --save-dir and would overwrite each other
if pointed at one directory. We run them into separate directories and merge
here.

Rules:
- the batch-size keys are unioned; on a conflict the LATER file wins and the
  override is printed;
- exactly one "triton_version" is kept (mismatches are refused: the configs
  would not be comparable);
- keys are sorted numerically (1, 2, ... 8192), not lexicographically;
- every config must carry BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages;
- OUT.json is written atomically (tmp + os.replace) and an existing OUT.json is never
  overwritten without --force: the inputs cost hours to produce;
- the inputs must all share the same E=...,N=...,device_name=...,dtype=...,
  block_shape=... filename, i.e. describe the same kernel shape.
"""

import json
import os
import re
import sys
import tempfile

REQUIRED_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)

# E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json
# Matched anywhere in the basename, so a renamed copy (decode_E=...json) still works.
SHAPE_RE = re.compile(
    r"E=\d+,N=\d+,device_name=[^,]+,dtype=[^,]+,block_shape=\[[^\]]*\]"
)


def shape_of(path: str) -> str:
    name = os.path.basename(path)
    m = SHAPE_RE.search(name)
    if not m:
        die(f"{name}: not a benchmark_moe output filename (no 'E=..,N=..,..' part)")
    return m.group(0)


def die(msg: str) -> None:
    print(f"merge-configs: {msg}", file=sys.stderr)
    sys.exit(1)


USAGE = (
    "usage: merge-configs.py [--force] [--only-keys K1,K2,...] [--out PATH] "
    "OUT.json IN1.json IN2.json [...]"
)


def parse_args(argv: list[str]) -> tuple[bool, set[str] | None, str | None, list[str]]:
    force = False
    only_keys: set[str] | None = None
    out_override: str | None = None
    rest: list[str] = []
    args = argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--force":
            force = True
        elif a == "--only-keys":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                die(f"--only-keys needs a comma list of batch sizes\n{USAGE}")
            keys = [k.strip() for k in args[i + 1].split(",") if k.strip()]
            if not keys or not all(k.isdigit() for k in keys):
                die(f"--only-keys: expected a comma list of batch sizes, got {args[i + 1]!r}")
            only_keys = set(keys)
            i += 1
        elif a == "--out":
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                die(f"--out needs a path\n{USAGE}")
            out_override = args[i + 1]
            i += 1
        elif a.startswith("--"):
            die(f"unknown option {a}\n{USAGE}")
        else:
            rest.append(a)
        i += 1
    return force, only_keys, out_override, rest


def main(argv: list[str]) -> None:
    force, only_keys, out_override, args = parse_args(argv)
    if len(args) < 2:
        die(USAGE)
    base_path, in_paths = args[0], args[1:]
    selective = only_keys is not None or out_override is not None
    out_path = out_override or base_path
    if os.path.exists(out_path) and not force:
        die(f"refusing: {out_path} exists (pass --force to overwrite)")

    if selective and not os.path.exists(base_path):
        die(
            f"selective mode (--only-keys/--out): base file {base_path} does not exist; "
            "the first positional file must be the JSON to take the untouched keys from"
        )

    shapes = {p: shape_of(p) for p in in_paths}
    if selective:
        shapes[base_path] = shape_of(base_path)
    distinct = sorted(set(shapes.values()))
    if len(distinct) > 1:
        for p, s in shapes.items():
            print(f"  {p} -> {s}", file=sys.stderr)
        die(f"refusing: inputs describe {len(distinct)} different kernel shapes")

    merged: dict[str, dict] = {}
    origin: dict[str, str] = {}
    triton_version = None
    triton_origin = None
    taken: set[str] = set()

    def load(path: str) -> dict:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            die(f"{path}: top level is not a JSON object")
        return data

    def absorb(path: str, data: dict, keys_filter: set[str] | None) -> None:
        nonlocal triton_version, triton_origin
        version = data.pop("triton_version", None)
        if version is not None:
            if triton_version is None:
                triton_version, triton_origin = version, path
            elif version != triton_version:
                die(
                    f"refusing: triton_version mismatch — {triton_origin} has "
                    f"{triton_version!r}, {path} has {version!r}"
                )
        for key, cfg in data.items():
            if not key.isdigit():
                die(f"{path}: key {key!r} is not a batch size")
            if not isinstance(cfg, dict):
                die(f"{path}: config for batch {key} is not an object")
            missing = [k for k in REQUIRED_KEYS if k not in cfg]
            if missing:
                die(f"{path}: config for batch {key} misses {', '.join(missing)}")
            if keys_filter is not None and key not in keys_filter:
                print(f"skipped: batch {key} from {path} (not in --only-keys)", file=sys.stderr)
                continue
            if key in merged and cfg != merged[key]:
                print(
                    f"override: batch {key} from {path} replaces {origin[key]}",
                    file=sys.stderr,
                )
            merged[key] = cfg
            origin[key] = path
            if keys_filter is not None:
                taken.add(key)

    # v2: in selective mode OUT.json is the base every other key is kept from.
    if selective:
        absorb(base_path, load(base_path), None)
        print(f"base: {base_path} ({len(merged)} batch sizes kept)", file=sys.stderr)

    for path in in_paths:
        absorb(path, load(path), only_keys)

    if only_keys is not None:
        not_found = sorted(only_keys - taken, key=int)
        if not_found:
            die(f"refusing: --only-keys {', '.join(not_found)} not present in any input")

    if not merged:
        die("refusing: nothing to merge (no batch-size keys in the inputs)")

    ordered = {k: merged[k] for k in sorted(merged, key=int)}
    payload: dict = {}
    if triton_version is not None:
        payload["triton_version"] = triton_version
    payload.update(ordered)

    # Atomic: unique temp file in the target directory, flushed and fsync'ed, then
    # renamed over the destination (same filesystem, so os.replace is atomic).
    out_dir = os.path.dirname(os.path.abspath(out_path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    batches = ", ".join(sorted(merged, key=int))
    print(f"wrote {out_path}: {len(merged)} batch sizes ({batches})")
    if only_keys is not None:
        print(f"taken from inputs: {', '.join(sorted(taken, key=int))}")
    if triton_version is not None:
        print(f"triton_version: {triton_version} (from {triton_origin})")


if __name__ == "__main__":
    main(sys.argv)
