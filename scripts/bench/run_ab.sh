#!/usr/bin/env bash
# run_ab.sh — one full A/B pass against the stack that is currently up:
#   decode structured x1, decode prose x1, decode structured concurrency 4,
#   prefill ~30k, prefill ~100k.
# <label> is the variant under test, e.g. baseline | candidate.
# Results go to bench-results/<timestamp>-<pid>-<label>.json and a NEUTRAL delta
# table vs the public production baseline is printed (no verdicts: the owner
# sets the decision criteria after seeing all variants — use compare.py to line
# them up). Env overrides:
#   BENCH_URL and BENCH_MODEL (default: http://$MASTER_IP:$API_PORT and $SERVED_NAME
#   from cluster.env, or http://localhost:8000 and glm-5.3-flash without it),
#   RUNS (decode waves,
#   default 3), CONCURRENCY (default 4), DECODE_MAX_TOKENS (default 200),
#   PREFILL_RUNS (if set, overrides the run count for BOTH prefill phases; when
#   unset, ~30k runs 3x and ~100k 2x), PREFILL_SMALL (default 30000),
#   PREFILL_LARGE (default 100000), LONG_DECODE=1 to add the 1400-token decode
#   pass (prompt count 1->3000: the short structured task ends naturally at
#   ~404 tok, finish_reason=stop — see the live pass 2026-09-01). The run
#   records carry the finish_reason assertion and the table flags early stops.
set -u

usage() { echo "usage: $0 <label>   # label = the variant under test, e.g. baseline | candidate"; }
# The -h/--help arm comes FIRST: both are valid label characters, so without it `--help`
# would be taken as a variant name and start a full bench pass against the endpoint.
case ${1:-} in
  -h|--help) usage; exit 0 ;;
  ""|*[!A-Za-z0-9._+-]*) usage >&2; exit 2 ;;
  *) LABEL="$1" ;;
esac

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$ROOT/bench-results"
TS="$(date +%Y%m%d-%H%M%S)"
FINAL="$OUT_DIR/${TS}-$$-${LABEL}.json"   # $$: collision-proof within the same second
mkdir -p "$OUT_DIR"

# Endpoint defaults come from the production recipe when the checkout has one, so a pass
# needs no environment at all; BENCH_URL/BENCH_MODEL still win over it. TP4_ENV is honoured
# exactly as in tp4ctl, so an experiment window benches the endpoint IT is serving.
# shellcheck disable=SC2034  # read by scripts/lib/common.sh (log/warn/die prefix)
TP4_LOG_TAG='[run_ab]'
# A checkout WITHOUT cluster.env still benches (BENCH_URL/BENCH_MODEL, or the localhost
# defaults below); one WITH it must have a complete recipe, or the endpoint the table is
# labelled with is not the one the recipe describes: hence --require.
if [ -f "$ROOT/cluster.env" ]; then
  # shellcheck source=../lib/common.sh
  . "$ROOT/scripts/lib/common.sh"
  tp4_load_env "$ROOT" --require --overlay
fi
URL="${BENCH_URL:-http://${MASTER_IP:-localhost}:${API_PORT:-8000}}"
MODEL="${BENCH_MODEL:-${SERVED_NAME:-glm-5.3-flash}}"
RUNS="${RUNS:-3}"
CONC="${CONCURRENCY:-4}"
DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-200}"
PREFILL_RUNS="${PREFILL_RUNS:-}"   # if set, same run count for BOTH prefill phases
PREFILL_SMALL="${PREFILL_SMALL:-30000}"
PREFILL_LARGE="${PREFILL_LARGE:-100000}"
LONG_DECODE="${LONG_DECODE:-0}"

# Production reference 2026-09-05, adaptive-k recipe, two clean rank-0 loopback
# passes. docs/bench.md publishes the method and limits. Values are the mean of each
# pass median; noise is ±3-5% on three-run decode medians and ±2-3% on prefill.
# Used only as a neutral delta column, never as an automatic decision gate.
BASE_DECODE_STRUCTURED=72.5
BASE_DECODE_PROSE=41.4
BASE_DECODE_C4=201.9
BASE_DECODE_LONG=60.9
BASE_PREFILL_SMALL=2195.1
BASE_PREFILL_LARGE=2213.7

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
run_phase() { # name -> args...
  local name="$1" rc=0; shift
  echo "[run_ab] phase $name: $*" >&2
  python3 "$@" --out "$TMP/$name.json" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[run_ab] WARNING: phase $name failed (rc=$rc)" >&2
    FAIL=1
    # the caller reports this code as "endpoint unusable (rc=...)": pass python's own
    return "$rc"
  fi
  return 0
}

# First phase is fatal: if the endpoint can't serve it, the pass is pointless.
echo "[run_ab] label=$LABEL url=$URL model=$MODEL runs=$RUNS" >&2
run_phase decode_structured "$HERE/bench_decode.py" \
  --label decode-structured --prompt structured --runs "$RUNS" \
  --max-tokens "$DECODE_MAX_TOKENS" --url "$URL" --model "$MODEL" || {
  rc=$?
  echo "[run_ab] endpoint unusable, aborting (rc=$rc)" >&2
  # leave a disk trace of the failed pass before bailing out
  printf '{"label": "%s", "ts": %s, "fatal": "first phase decode_structured failed (rc=%s); endpoint unusable"}\n' \
    "$LABEL" "$(date +%s)" "$rc" > "$FINAL"
  exit 2
}

run_phase decode_prose "$HERE/bench_decode.py" \
  --label decode-prose --prompt prose --runs "$RUNS" \
  --max-tokens "$DECODE_MAX_TOKENS" --url "$URL" --model "$MODEL"

run_phase decode_structured_c4 "$HERE/bench_decode.py" \
  --label decode-structured-c4 --prompt structured --runs "$RUNS" \
  --concurrency "$CONC" --max-tokens "$DECODE_MAX_TOKENS" \
  --url "$URL" --model "$MODEL"

if [ "$LONG_DECODE" = "1" ]; then
  run_phase decode_structured_1400 "$HERE/bench_decode.py" \
    --label decode-structured-1400 --prompt count3000 --runs "$RUNS" \
    --max-tokens 1400 --url "$URL" --model "$MODEL"
fi

run_phase prefill_30k "$HERE/bench_prefill.py" \
  --label prefill-30k --target-tokens "$PREFILL_SMALL" --runs "${PREFILL_RUNS:-3}" \
  --url "$URL" --model "$MODEL"

run_phase prefill_100k "$HERE/bench_prefill.py" \
  --label prefill-100k --target-tokens "$PREFILL_LARGE" --runs "${PREFILL_RUNS:-2}" \
  --url "$URL" --model "$MODEL"

python3 - "$FINAL" "$LABEL" "$TMP" "$LONG_DECODE" \
  "$BASE_DECODE_STRUCTURED" "$BASE_DECODE_PROSE" "$BASE_DECODE_C4" "$BASE_DECODE_LONG" \
  "$BASE_PREFILL_SMALL" "$BASE_PREFILL_LARGE" \
  "$PREFILL_SMALL" "$PREFILL_LARGE" <<'PY'
import json, sys, time

(final, label, tmp, long_decode, b_structured, b_prose, b_c4, b_long,
 b_pf_small, b_pf_large, pf_small, pf_large) = sys.argv[1:13]
names = ["decode_structured", "decode_prose", "decode_structured_c4"]
if long_decode == "1":
    names.append("decode_structured_1400")
names += ["prefill_30k", "prefill_100k"]

rec = {"kind": "ab-pass", "label": label, "ts": time.time()}
for name in names:
    try:
        with open(f"{tmp}/{name}.json") as fh:
            rec[name] = json.load(fh)
    except OSError:
        rec[name] = None
with open(final, "w") as fh:
    json.dump(rec, fh, indent=2)

# headline metric per phase: per-stream decode tok/s (median), aggregate tok/s
# for the c4 wave, prefill tok/s for the long prompts
def head(rec):
    if rec is None:
        return None
    key = "prefill_tok_s_median" if rec.get("kind") == "prefill" else "tok_s_median"
    return rec.get(key)

def agg(rec):
    return None if rec is None else rec.get("aggregate_tok_s_median")

def kfmt(v):
    if v is None:
        return "n/d"
    return f"{v / 1000:.1f}k" if v >= 1000 else str(int(v))

def pf_label(rec, nominal):
    # A1: row name carries the ACTUAL median prompt_tokens from the server,
    # never the nominal target
    if rec is not None and rec.get("prompt_tokens_median") is not None:
        lab = f"prefill {kfmt(rec['prompt_tokens_median'])} tok"
        if (rec.get("sizing_correction") or {}).get("triggered"):
            lab += f" (nominal {kfmt(nominal)})"
        return lab
    return f"prefill ~{kfmt(nominal)} {'FAILED' if rec is not None else '(absent)'}"

def stop_finish_count(rec):
    if rec is None:
        return None
    return sum(1 for run in (rec.get("runs_detail") or [])
                 for s in (run.get("per_stream") or [])
                 if isinstance(s, dict) and "error" not in s
                 and s.get("finish_reason") == "stop")

rows = [
    ("decode structured x1", head(rec["decode_structured"]),   float(b_structured), ""),
    ("decode prose x1",      head(rec["decode_prose"]),        float(b_prose), ""),
    ("decode structured x4", agg(rec["decode_structured_c4"]), float(b_c4), ""),
]
if long_decode == "1":
    stops = stop_finish_count(rec["decode_structured_1400"])
    note = "FAILED" if stops is None else (
        f"STOPPED EARLY in {stops} stream(s) (finish_reason=stop)"
        if stops else "ok (all finish_reason=length)")
    rows.append(("decode @1400 (count 1->3000)", head(rec["decode_structured_1400"]),
                 float(b_long), note))
rows += [
    (pf_label(rec["prefill_30k"],  int(pf_small)),  head(rec["prefill_30k"]),  float(b_pf_small), ""),
    (pf_label(rec["prefill_100k"], int(pf_large)),  head(rec["prefill_100k"]), float(b_pf_large), ""),
]

fmt = lambda v: "FAILED" if v is None else f"{v:.1f}"
print()
print(f"== A/B pass '{label}' — delta vs the public 2026-09-05 production reference "
      "(docs/bench.md, section 'Public reference — 2026-09-05') ==")
print("baseline col = mean of two clean passes on the cluster.env recipe; "
      "noise ±3-5% on decode medians, ±2-3% on prefill")
print(f"{'test':34} {'this run':>10} {'baseline':>12}  {'delta':>8}  note")
for name, v, b, note in rows:
    delta = f"{(v - b) / b * 100:+.1f}%" if v is not None else "—"
    print(f"{name:34} {fmt(v):>10} {b:>12.1f}  {delta:>8}  {note}")

# survival flags already present in the phase JSON: needle hits in the prefill
# phases, healthy vs failed streams in the c4 wave (bench_decode.py recorded them)
def needle_count(rec):
    if rec is None:
        return "—"
    runs = rec.get("runs_detail") or []
    return f"{sum(1 for r in runs if r.get('needle_ok'))}/{len(runs)}"

def c4_counts(rec):
    if rec is None:
        return "—"
    ok = sum(r.get("ok_streams", 0) for r in (rec.get("runs_detail") or []))
    return f"{ok} ok / {rec.get('failed_streams', '?')} failed"

print()
print(f"needle recovered (prefill): ~30k {needle_count(rec['prefill_30k'])}, "
      f"~100k {needle_count(rec['prefill_100k'])}"
      f"   |   c4 streams: {c4_counts(rec['decode_structured_c4'])}")
print()
print(f"results: {final}")
print("cross-variant compare: "
      "python3 scripts/bench/compare.py bench-results/<pass>.json ...")
sys.exit(1 if rec["decode_structured"] is None and rec["prefill_30k"] is None else 0)
PY
rc=$?
[ "$FAIL" -eq 1 ] && rc=1
exit "$rc"
