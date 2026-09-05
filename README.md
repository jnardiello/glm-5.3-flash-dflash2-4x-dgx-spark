# GLM-5.3-Flash on four NVIDIA GB10 nodes

This repository installs and operates one vLLM tensor-parallel cluster serving
GLM-5.3-Flash FP8 across four NVIDIA GB10 systems. It was verified on four ASUS
Ascent GX10 nodes connected as a switchless ConnectX-7 RoCE ring. Other GB10
systems may need different interface, HCA, GID, renderer, or package settings.

The public API is OpenAI compatible and is exposed by rank 0. It has no API key,
TLS, rate limit, or caller isolation. Keep the management network and all four
fabric links private; expose the endpoint only through a trusted VPN or an
authenticating reverse proxy. The deployment account receives passwordless sudo
because Docker, systemd, network setup, and cache management require it.

## Hardware

| Item | Verified configuration |
| --- | --- |
| Nodes | 4 × ASUS Ascent GX10, one NVIDIA GB10 and 128 GB unified memory each |
| Storage | 1 TB NVMe per node; allow at least 330 GiB free for a fresh model fetch |
| Fabric | one ConnectX-7 per node, two addressed ports, 4 × QSFP28 DAC, 200 Gb/s, MTU 9000 |
| Topology | closed ring: rank 0 ↔ 1 ↔ 2 ↔ 3 ↔ 0, one private /24 per link |
| Management | separate Ethernet or trusted VPN path for SSH and the rank-0 API |

All site addresses and hardware selections belong in the gitignored `cluster.env`.
[`cluster.env.example`](cluster.env.example) is the annotated source for every
public recipe value and its rollback.

## Start here

An agent must read [`AGENTS.md`](AGENTS.md) first. For a new installation, follow
[`docs/install-from-zero.md`](docs/install-from-zero.md) in order. For an existing
cluster, begin with the read-only status procedure in
[`docs/operations.md`](docs/operations.md).

The shortest safe sequence for a new checkout is:

```sh
TP4_HOSTS='user@node0 user@node1 user@node2 user@node3' \
  ./scripts/agent-preflight.sh --report /tmp/tp4-preflight.json

cp cluster.env.example cluster.env
$EDITOR cluster.env
./scripts/render-netplan.sh --write
```

The public checkout intentionally has no `cluster.env`. The user creates this required,
gitignored site file from the template after the targeted read-only preflight, which
uses explicit `TP4_HOSTS`. Run documented shell blocks with Bash because several
procedures use Bash arrays and loops.

Review the proposed rank and cable map before changing any node. With an approved
installation window, complete the bootstrap, NCCL, image, and weight steps in the
installation guide. With a separately approved serving window:

```sh
./scripts/verify-node.sh
./scripts/tp4ctl fabric-check
./scripts/tp4ctl up
./scripts/tp4ctl health
```

`/health` returning 200 is the readiness signal. `/v1/models` may answer while the
engine is still loading. A cold start takes roughly 16 minutes on the verified
hardware; do not issue a second `up` while it is loading.

## Use the endpoint

From a client that is already inside the trusted network:

```sh
curl http://<MGMT_IP_RANK0>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-flash",
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [{"role": "user", "content": "Reply with READY."}]
  }'
```

The current local chat-template adapter closes an empty `<think></think>` block
when `enable_thinking=false`. This is a local compatibility behavior, not a native
GLM-5.3-Flash reasoning mode. The official request values remain `low`, `high`, and
`max`; see [`docs/production-recipe.md`](docs/production-recipe.md).

## Benchmark results

Latest clean campaign: 2026-09-05, rank-0 loopback after reboot. Each headline is the
mean of two pass medians after a discarded warm-up.

### Current — 5 September 2026

| Phase | Current result |
| --- | ---: |
| Structured decode ×1 | 72.5 tok/s |
| Prose decode ×1, standard harness | 41.4 tok/s |
| Code decode ×1 | 45.1 tok/s |
| Prose decode ×1, separate prompt run | 42.7 tok/s |
| Four-stream aggregate | 201.9 tok/s |
| Four-stream per stream | 52.1 tok/s |
| Sustained decode, 1400-token limit | 60.9 tok/s |
| Prefill ~30K | 2195.1 tok/s |
| Prefill ~100K | 2213.7 tok/s |

### Initial campaign baseline — 2 September 2026

This repository's campaign start was the fixed DFlash2 k=3 configuration. The table shows its
historical single-pass medians beside the current results; percentages use the raw
medians before display rounding.

| Phase | Initial | Current | Historical change |
| --- | ---: | ---: | ---: |
| Structured decode ×1 | 50.2 tok/s | 72.5 tok/s | +44.4% |
| Prose decode ×1, standard harness | 35.6 tok/s | 41.4 tok/s | +16.2% |
| Code decode ×1 | 35.5 tok/s | 45.1 tok/s | +27.0% |
| Prose decode ×1, separate prompt run | 34.3 tok/s | 42.7 tok/s | +24.5% |
| Four-stream aggregate | 126.8 tok/s | 201.9 tok/s | +59.2% |
| Four-stream per stream | 33.6 tok/s | 52.1 tok/s | +55.1% |
| Sustained decode, 1400-token limit | — | 60.9 tok/s | — |
| Prefill ~30K | 1907.6 tok/s | 2195.1 tok/s | +15.1% |
| Prefill ~100K | 2085.2 tok/s | 2213.7 tok/s | +6.2% |

The initial configuration used fixed k=3, async off, a 524288×4 context/sequence setting,
workstation-to-LAN requests, and a different warm-up method. The current lane uses
adaptive k=3/5, 262144×6, rank-0 loopback, a newer upstream snapshot, and the local
chat-template adapter; the initial thinking-off behavior was declared but not proven.
The two current passes used the same image and weight shards. This is a historical
deployment comparison, not a controlled A/B or a causal engine gain. One of six
current sustained requests ended at 724 output tokens instead of the 1400-token limit.
Some retrieval requests produced refusals accepted by the owner as safe outcomes;
their literal `needle_ok` flag remains false. See
[`docs/bench.md`](docs/bench.md) for the protocol, acceptance counts, long-context
results, and other limits.

## Documentation

| Need | Read |
| --- | --- |
| Install from OS + driver + Docker, including image and weights | [`docs/install-from-zero.md`](docs/install-from-zero.md) |
| Inspect, deploy, start, stop, recover, roll back, or promote | [`docs/operations.md`](docs/operations.md) |
| Cable, address, verify, or diagnose the RoCE ring and patched NCCL | [`docs/fabric.md`](docs/fabric.md) |
| Understand the current runtime recipe and its customizations | [`docs/production-recipe.md`](docs/production-recipe.md) |
| Run acceptance gates or reproducible benchmarks; read public results | [`docs/bench.md`](docs/bench.md) |
| Understand files copied to the nodes | [`scripts/node/README.md`](scripts/node/README.md) |
| Rebuild and install the patched NCCL library | [`scripts/node/nccl/README.md`](scripts/node/nccl/README.md) |

Changes are recorded incrementally in [`CHANGELOG.md`](CHANGELOG.md). Attribution
and third-party terms are in [`CREDITS.md`](CREDITS.md); original project material
is under [`LICENSE`](LICENSE), while derived files and fetched artifacts retain
their own terms.

## Development checks

Install Python 3.9+ and `Jinja2==3.1.6`, then run the offline check:

```sh
python3 -m pip install 'Jinja2==3.1.6'
./scripts/check.sh
```

The check uses no GPU, Docker daemon, SSH connection, private site configuration,
or model weights. It validates shell and Python syntax, local Markdown links,
command help, manifests, chat-template rendering, host lifecycle fixtures, agent
preflight fixtures, and the adaptive-k policy.
