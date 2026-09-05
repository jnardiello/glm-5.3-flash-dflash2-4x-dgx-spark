#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1

public_path() {
  case "$1" in
    ./.git/*|./.claude/*|./bench-results/*|./experiments/*|./reports/*|./todo/*|\
    ./node/moe-tune/*|./node/nccl-bench/*|./node/host/nsys-entry.sh|\
    ./scripts/prof-capture.sh|./scripts/nccl-bench.sh|./scripts/mirror-snapshot.sh|\
    ./scripts/mirror-allow.txt|./scripts/mirror-private-terms.example)
      return 1 ;;
    *) return 0 ;;
  esac
}

required=(
  scripts/tp4ctl
  scripts/launcher/launch-glm53-tp4.sh
  scripts/agent-preflight.sh
  scripts/bootstrap-node.sh
  scripts/deploy.sh
  scripts/deploy-host.sh
  scripts/fetch-fp8-weights.sh
  scripts/render-netplan.sh
  scripts/verify-node.sh
  scripts/bench/run_ab.sh
  scripts/bench/bench_decode.py
  scripts/bench/bench_prefill.py
  scripts/bench/bench_longctx.py
  scripts/bench/compare.py
  scripts/bench/perf-table.py
  scripts/bench/thermal-snapshot.sh
  scripts/node/flusher-unconditional.sh
  scripts/node/sparse_attn_indexer_kpool_sm121.py
  scripts/node/host/tp4-gpu-clocks.sh
  scripts/node/host/tp4-iommu.sh
  scripts/node/nccl/build.sh
  scripts/node/nccl/install-nccl.sh
  scripts/node/patches/adaptive_k_scheduler.py
)
for path in "${required[@]}"; do
  [ -e "$path" ] || { echo "check: missing required public path: $path" >&2; exit 1; }
done

shell_count=0
while IFS= read -r file; do
  public_path "$file" || continue
  bash -n "$file"
  shell_count=$((shell_count + 1))
done < <(find . -type f -name '*.sh' -print | sort)
echo "bash-syntax: PASS ($shell_count files)"
bash -n scripts/tp4ctl
echo "controller-syntax: PASS"

python_count=0
while IFS= read -r file; do
  public_path "$file" || continue
  python3 -c 'import ast, pathlib, sys; p=pathlib.Path(sys.argv[1]); ast.parse(p.read_text(encoding="utf-8"), str(p))' "$file"
  python_count=$((python_count + 1))
done < <(find . -type f -name '*.py' -print | sort)
echo "python-ast: PASS ($python_count files)"

python3 scripts/check_markdown_links.py

./scripts/tp4ctl --help >/dev/null
./scripts/deploy.sh --help >/dev/null
./scripts/deploy-host.sh --help >/dev/null
./scripts/bootstrap-node.sh --help >/dev/null
./scripts/verify-node.sh --help >/dev/null
./scripts/fetch-fp8-weights.sh --help >/dev/null
./scripts/render-netplan.sh --help >/dev/null
./scripts/agent-preflight.sh --help >/dev/null 2>&1
scripts/node/nccl/build.sh --help >/dev/null
scripts/node/nccl/install-nccl.sh --help >/dev/null
echo "command-help: PASS"

doc_count=$(find docs -type f -name '*.md' | wc -l | tr -d ' ')
[ "$doc_count" = 5 ] || { echo "check: docs/ must contain exactly 5 Markdown files (got $doc_count)" >&2; exit 1; }

./scripts/tests/test-agent-preflight.sh
./scripts/tests/test-host-lifecycle.sh
python3 scripts/tests/test-model-snapshot.py
python3 scripts/tests/test-chat-template.py
python3 scripts/node/patches/test_adaptive_k_policy.py

echo "check: PASS"
