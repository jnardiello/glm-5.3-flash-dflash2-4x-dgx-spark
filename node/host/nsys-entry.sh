#!/usr/bin/env bash
set -euo pipefail

# nsys-entry.sh — container entrypoint wrapper that starts the vLLM engine under
# `nsys launch` (Nsight Systems), so a GPU trace can be captured later, on demand,
# on the already-running stack, without a restart.
#
# PURPOSE
#   `nsys launch` only ATTACHES the profiler and opens the named session `tp4`; it
#   collects NOTHING until `nsys start --session=tp4` runs against that session.
#   Capture is driven from the workstation by scripts/prof-capture.sh.
#
# HOW IT IS MOUNTED (never by cluster.env, only by an experiment overlay)
#   scripts/deploy-host.sh pushes node/host/*.sh to ~/tp4/host/ on every node (push
#   phase only, no --run: this script is not an --apply/--revert knob).
#   experiments/2026-09-04-prof-nsys.env then adds, via EXTRA_DOCKER_ENV:
#     -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro   (image has no nsys)
#     -v $HOME/tp4/host/nsys-entry.sh:/nsys-entry.sh:ro   (expanded on the node by the launcher)
#     --entrypoint /nsys-entry.sh
#   `docker run --entrypoint` takes one executable with no arguments, which is why the
#   original entrypoint is reproduced here instead of being passed inline.
#   Without that overlay this file is inert: nothing on the node references it.
#
# ORIGINAL ENTRYPOINT (inspected on rank 0, 2026-09-03)
#   sudo docker inspect glm53_fp8_dflash_tp4 \
#     --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'
#   -> Entrypoint: ["vllm","serve"]
#   -> Cmd:        ["/model","--served-model-name","glm-5.3-flash",...]
#   The Cmd is not baked into the image: launcher/launch-glm53-tp4.sh appends it after
#   $IMAGE on every `docker run`, so docker passes it here as "$@". This wrapper only
#   restores the `vllm serve` prefix and forwards "$@" untouched.
#
# NSYS FLAGS (verified against `nsys launch --help`, 2025.3.2.474 on rank 0)
#   --session-new=tp4          named session, so `nsys start --session=tp4` finds it
#   --trace=cuda,nvtx          `start` has NO --trace switch: the API set is fixed here
#   --trace-fork-before-exec=true
#                              vLLM forks EngineCore and Worker children; without this
#                              they are not followed and the trace shows no GPU work
#   --cuda-graph-trace=node    REQUIRED for this study. With CUDA driver >= 11.7 the
#                              default is 'graph', which traces each captured CUDA graph
#                              as ONE opaque range — and vLLM replays decode as CUDA
#                              graphs, so the whole point of the capture (per-kernel
#                              decode mix: MoE vs NCCL vs attention) would be invisible.
#                              'node' collects the individual node activities instead.
#                              nsys's own help warns this "may cause significant runtime
#                              overhead"; it is confined to the start/stop window.
#   NOT set here: --sample / --cpuctxsw. In 2025.3.2 both are "(Deprecated) ... no longer
#   supported" on `launch` and must be set on `start`, which prof-capture.sh does
#   (both =none: RmProfilingAdminOnly=1 and perf_event_paranoid=4 on these nodes make
#   GPU-metrics and CPU sampling unavailable anyway).
#
# OPERATIONAL WARNINGS — read before using the overlay
#   * NO BENCH NUMBER TAKEN UNDER THIS OVERLAY IS COMPARABLE TO PRODUCTION. CUPTI is
#     attached from process start, for the whole life of the container, not just inside
#     the start/stop window; --cuda-graph-trace=node adds more. Use this overlay to read
#     the kernel MIX (shares), never to produce tok/s figures for a lane comparison.
#   * PID 1 IS NSYS, not vLLM. `docker stop` signals nsys, which forwards to the engine,
#     but a collection left running can lose its report. ALWAYS `nsys stop --session=tp4`
#     BEFORE any `tp4ctl down`/`restart` (prof-capture.sh does this on EXIT via trap).
#   * FAILURE MODE: if the nsys tree is missing on a node, this script exits 1, that rank
#     never starts, and the communicator never forms — `tp4ctl up` will not fail fast, it
#     waits out its 35-min health timeout. Run `scripts/prof-capture.sh precheck` BEFORE
#     the restart (and again right after) to catch this in seconds instead.

# Single pinned path, identical to the one used by scripts/prof-capture.sh. Deliberately
# NOT a glob: a silent fallback to another version would desynchronize the two sides.
NSYS=/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys
if [ ! -x "$NSYS" ]; then
  echo "nsys-entry: FATAL: $NSYS is missing or not executable in the container." >&2
  echo "nsys-entry: the host /opt/nvidia/nsight-systems bind mount is absent or the" >&2
  echo "nsys-entry: node does not carry Nsight Systems 2025.3.2." >&2
  echo "nsys-entry: THIS RANK WILL NOT START and tp4ctl up will hang until its timeout." >&2
  echo "nsys-entry: run scripts/prof-capture.sh precheck from the workstation." >&2
  exit 1
fi

echo "nsys-entry: launching under nsys session tp4" >&2

exec "$NSYS" launch \
  --session-new=tp4 \
  --trace=cuda,nvtx \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -- vllm serve "$@"
