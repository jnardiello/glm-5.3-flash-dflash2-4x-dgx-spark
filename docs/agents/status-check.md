# Status check — what is production serving right now (60 seconds)

Read-only. Run this **before** anything else, and again before any measurement. It answers one
question: is the endpoint serving the documented recipe, or has it drifted? Nothing here restarts,
deploys or repairs: on any mismatch the rule is **report to the owner, touch nothing**
(`AGENTS.md` §6).

## 1. The visible layer — `tp4ctl`

```sh
./tp4ctl status                          # per-node containers + endpoint health
./tp4ctl health                          # /health 200 + smoke chat completion ("2+2")
```

Or unfiltered, which is what you want when `status` looks odd (it filters on `CONTAINER`):

```sh
. ./cluster.env
for h in $NODES; do
  echo "== $h"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$h" \
    'sudo -n docker ps --format "{{.Names}} | {{.Image}} | {{.Status}}"'
done
curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://<MGMT_IP_RANK0>:8000/health
curl -s -m 5 http://<MGMT_IP_RANK0>:8000/v1/models \
  | python3 -c 'import sys,json; [print(m["id"], m["max_model_len"]) for m in json.load(sys.stdin)["data"]]'
```

Healthy shape (production since 2026-09-04 — a recipe, not a permanent truth):

```
== <ALIAS_RANK0>
glm53_fp8_dflash_tp4 | ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 | Up 3 hours
== <ALIAS_RANK1> … <ALIAS_RANK3>: identical
200
glm-5.3-flash 262144
```

Health is always `/health`, never `/v1/models`, which answers 200 before the engine is ready
(`README.md` § Troubleshooting).

Off-LAN the workstation cannot reach `MASTER_IP` and `./tp4ctl status` prints `-> 000` for the
endpoint; check from the node instead:

```sh
ssh <ALIAS_RANK0> 'curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health'   # -> 200
```

## 2. The four invisible signatures

`docker ps` + `/health` cannot see any of these four. Each one is part of the production recipe and
each one has been observed to drift silently, costing performance with no error anywhere.

| # | Signature | Command that reveals it | Healthy | Drifted looks like |
| --- | --- | --- | --- | --- |
| 1 | Container name identical on all 4 ranks | the `docker ps` loop above | `glm53_fp8_dflash_tp4` `Up` on 4 nodes | a rank missing, or one node on a different name → half-done switch or a broken cluster |
| 2 | GB10-tuned fused-MoE kernel config | `./tp4ctl logs` (rank 0), grep `MoE layer` | `Using configuration from /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json for MoE layer.` and `Using TRITON Fp8 MoE backend out of potential backends: [...]` | `Using default MoE config. Performance might be sub-optimal! Config file not found at …` → the bind mount did not land on that rank; a missing `TRITON` line means `--moe-backend triton` was dropped |
| 3 | Host tier: SMMU in passthrough on **all 4** nodes | `./scripts/deploy-host.sh --no-push --run tp4-iommu.sh --status` | `state: passthrough (cmdline) / drop-in installed / grub.cfg in sync` on 4 nodes | `translated` on any node → that node boots without `iommu.passthrough=1` and paces all four (TP4: the slowest node wins) |
| 4 | Adaptive draft length active | `./tp4ctl logs` (rank 0), grep `adaptive-k` and `num_speculative_tokens` | `adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 up=0.58 down=0.42 alpha=0.15 seed=1.0 mode=per-request signal=pos …) engine_k=5 async=True dynamic_sd_table=[…]` plus the engine init line carrying `num_speculative_tokens=5` and `num_speculative_tokens_per_batch_size=[(1, 1, 5), (2, 6, 3)]` | no `adaptive-k` line at all → the `--scheduler-cls` mount or `PYTHONPATH` is missing and the engine runs a fixed k=5 (prose regression, no gain guarantee); `async=False` → the policy disabled itself and every request runs the stock placeholder length; the init line without the table → `SPEC_EXTRA_JSON` was lost and only one decode CUDA-graph family was captured |

Signature 4 also has a periodic line every 200 verify steps once traffic flows —
`adaptive-k: tracked=N obs=… decisions lo/hi=a/b switches=s uniform-steps lo/hi=…` — which is the
cheapest proof that the policy is actually deciding and not stuck. Details:
`docs/adaptive-k.md`.

One grep that covers signatures 2 and 4 on rank 0:

```sh
ssh <ALIAS_RANK0> 'sudo -n docker logs glm53_fp8_dflash_tp4 2>&1 | grep -E "MoE layer|MoE backend|adaptive-k|num_speculative_tokens" | head -20'
```

## 3. Idle check — before any measurement

The endpoint is shared with the owner's clients (opencode, pi, a phone). A pass measured while
someone else is generating is invalid, not noisy:

```sh
curl -s http://<MGMT_IP_RANK0>:8000/metrics | grep vllm:num_requests_running    # must read 0
```

`vllm:num_requests_running` must be `0` before every phase; after the pass the rank-0 log must show
no `Running: N reqs` above that phase's concurrency. Full procedure:
`docs/agents/bench-protocol.md`.

## 4. Units on rank 0 (only when the cluster came back by itself)

```sh
ssh <ALIAS_RANK0> 'systemctl status tp4-autostart tp4-fabric-iptables --no-pager'
ssh <ALIAS_RANK0> 'journalctl -u tp4-autostart -n 50 --no-pager'
```

`tp4-autostart.service` runs `~/tp4/tp4ctl up` with **no** `TP4_ENV`, i.e. whatever is in
`~/tp4/cluster.env`: after a rank 0 reboot the cluster serves the deployed production recipe,
never an overlay. The `tp4-flusher` unit is **transient** (`systemd-run` from `tp4ctl up`): it
exists only while the weights load and is stopped when `/health` returns 200 — its absence is
normal.

## What a healthy boot looks like

`tp4ctl up` runs a fixed sequence: wait for peers over ssh (up to 10 min) → `fabric-check` (aborts
on failure) → start the page-cache flusher on all nodes → full teardown → launch ranks **3 → 2 → 1
→ 0** with 10 s between them → poll `/health` every 30 s (timeout 35 min, printing rank 0's last
log line) → stop the flusher → print the endpoint and `/v1/models`. The headless ranks must be
waiting before rank 0 opens the port, hence the descending order.

In the rank-0 log expect 62 safetensors shards loading over ~10 minutes with no output in between,
then the DFlash2 drafter, then:

```
Using TRITON Fp8 MoE backend out of potential backends: [...]
Using configuration from …/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json for MoE layer.
adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 …) engine_k=5 async=True dynamic_sd_table=[...]
GPU KV cache size: 2,143,717 tokens
Maximum concurrency for 262,144 tokens per request: 8.18x
Graph capturing finished in 27 secs
```

and finally the API server binding `:8000`. The two KV-cache lines of the 2026-09-04 12:12
production boot are archived verbatim in
[`../../bench-results/2026-09-04-kv-cache-lines.txt`](../../bench-results/2026-09-04-kv-cache-lines.txt);
re-read them from the rank-0 log after any recipe change rather than trusting the numbers above.
A measured cold boot to `/health` 200 is ~16 min. Ten
minutes of silence during `up` is normal — do not retry `up` while it is running.

Within **2 minutes** of `/health` 200 the sanity gate is mandatory on any boot that changed
something: `docs/agents/bench-protocol.md` § Preconditions, `docs/bench.md` § Post-boot sanity gate.
