# Production recipe — the ledger

The single answer to two questions: **what is production, exactly**, and **how did every piece of
it get there**. It is a ledger, not a runbook: the ordered rebuild is
[`install-from-zero.md`](install-from-zero.md), the day-to-day operation is
[`../AGENTS.md`](../AGENTS.md), the per-knob rationale is `cluster.env.example`. Nothing here
restates a procedure — every row points at the file that owns it.

Placeholders are the repo-wide ones: `<USER>` the login account, `<MGMT_IP_RANKn>` /
`<FABRIC_IP_RANKn_Fm>` the addresses (all in the gitignored `cluster.env`), `<TAILNET>` the
Tailscale tailnet.

Last verified: **2026-09-04** — production runs adaptive draft length v1 (promoted that day at
12:05), and a node rebuild was rehearsed from zero the same afternoon
([`install-from-zero.md`](install-from-zero.md) § Verified).

---

## 1. Snapshot of production

Everything that has to be true for the endpoint to serve what it serves today. "Owned by" is the
file that carries the value; changing it anywhere else does not change production.

| # | Component | Value in production | Owned by |
| --- | --- | --- | --- |
| 1 | Model checkpoint | `zai-org/GLM-5.3-Flash`, FP8, 62 shards `model-*-of-00062.safetensors`, ~306 GiB, flat dir `~/glm53-flash-fp8-zai`, mounted `/model:ro` | `cluster.env` (`MODEL_DIR`, `MODEL_REPO`), [`weights.md`](weights.md) |
| 2 | Checkpoint revision | **not pinned** — `MODEL_REV=""` follows the card's HEAD; only `scripts/fetch-fp8-weights.sh` reads it | `cluster.env` (`MODEL_REV`) |
| 3 | Drafter | `incoai/GLM-5.3-Flash-DFlash2`, 5 layers, `block_size` 8, `target_layer_ids [5,14,24,33,42]`, 2.34 GB bf16, `~/glm53-dflash2-draft` → `/draft:ro`. CC BY-NC-ND 4.0, non-commercial. **Revision not pinned** | `cluster.env` (`DRAFT_DIR`), [`weights.md`](weights.md) |
| 4 | Container image | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`, 31 233 908 429 B ≈ 29.1 GiB on the node; not built here | `cluster.env.example` (`IMAGE`), [`image-rebuild-study.md`](image-rebuild-study.md) |
| 5 | vLLM build inside the image | `0.1.dev20051+g487ecf187`, i.e. vLLM commit `487ecf187d3dfe74d2cf6119a92881dba403c219` (2026-08-25) | [`../node/moe-tune/README.md`](../node/moe-tune/README.md), [`../node/patches/README.md`](../node/patches/README.md) |
| 6 | NCCL | patched `libnccl.so.2`, base tag `v2.30.7-1` (commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`) + the `skip-tree-connect` overlay from `josephdrose/nccl-spark-switchless` commit `27ca6d3bdc43d6c2978fc34b920cdc8a218a333a`; sha256 `1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5`, 61 581 280 B, 165 dynamic symbols (the 2026-09-04 rebuild, canonical); host-preloaded from `~/nccl-patched/` | [`../node/nccl/SHA256SUMS`](../node/nccl/SHA256SUMS), [`../node/nccl/expected.env`](../node/nccl/expected.env), [`nccl.md`](nccl.md) |
| 7 | NCCL runtime env | `NCCL_SKIP_TREE_CONNECT=1`, `NCCL_ALGO=Ring`, `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`, `NCCL_IB_GID_INDEX=3`, RoCE v2, `NCCL_{MIN,MAX}_NCHANNELS=4`, `NCCL_PROTO=LL,LL128,Simple` (never forced), relay **not** enabled | [`../launcher/launch-glm53-tp4.sh`](../launcher/launch-glm53-tp4.sh) (fixed env block) |
| 8 | Parallelism | tensor parallel 4, one GB10 per node, `--nnodes 4 --node-rank <n>`, `--distributed-executor-backend mp`, rendez-vous `<MGMT_IP_RANK0>:29520` | `cluster.env` (`NODES`, `MASTER_IP`, `MASTER_PORT`), launcher |
| 9 | Context lane | `MAX_MODEL_LEN=262144` × `MAX_NUM_SEQS=6` | `cluster.env.example` |
| 10 | KV cache | `KV_CACHE_DTYPE=fp8_e4m3`, pool pinned at 16 GiB/rank (`--kv-cache-memory=17179869184`), `GPU_MEM_UTIL=0.85` | `cluster.env.example` |
| 11 | Batching | `BATCHED_TOKENS=8192` (`--max-num-batched-tokens`), `BLOCK_SIZE=2304` | `cluster.env.example` |
| 12 | Speculative decoding | DFlash2, `SPEC_TOKENS=5` (engine draft length) + `--scheduler-cls adaptive_k_scheduler.AdaptiveKScheduler`: the drafter always drafts 5, the scheduler verifies 3 or 5 **per request** from that request's own acceptance history | `cluster.env.example`, [`adaptive-k.md`](adaptive-k.md), [`../node/patches/adaptive_k_scheduler.py`](../node/patches/adaptive_k_scheduler.py) |
| 13 | Dynamic-SD table | `SPEC_EXTRA_JSON='"num_speculative_tokens_per_batch_size":[[1,1,5],[2,6,3]]'` — kept only so FULL decode CUDA graphs are captured for both verify sizes (4 and 6 tokens) | `cluster.env.example` |
| 14 | Adaptive-k knobs | `VLLM_ADAPTIVE_K_MODE=per-request`, `_SEED=1.0`, `_DOWN=0.42`, `_UP=0.58`, `_ALPHA=0.15`, `_SIGNAL=pos` (defaults `_LO=3`, `_HI=5`, `_LOG_EVERY=200`) | `cluster.env.example` (`EXTRA_DOCKER_ENV`) |
| 15 | MoE backend | `--moe-backend triton` for the FP8 experts (dense FP8 linears stay on DeepGEMM E8M0) | `cluster.env.example` (`EXTRA_VLLM_ARGS`) |
| 16 | MoE kernel config | the **hybrid** GB10-tuned JSON `E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json` — tuned entries for M ≤ 8 only, vLLM defaults above — bind-mounted over the image's copy | [`../node/moe-configs/`](../node/moe-configs/), `cluster.env.example` (`EXTRA_DOCKER_ENV`) |
| 17 | Scheduling | `ASYNC_SCHEDULING=0`, i.e. **no** `--async-scheduling` flag; the base scheduler class is still `AsyncScheduler` (dflash is Eagle-type), which the adaptive-k patch requires | `cluster.env.example` |
| 18 | Container patches | sm_121 sparse-attention indexer K-pool fix mounted over the vLLM module; `adaptive_k_scheduler.py` at `/opt/tp4` with `PYTHONPATH=/opt/tp4`; `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | [`../node/patches/README.md`](../node/patches/README.md), launcher |
| 19 | Serving surface | model id `glm-5.3-flash`, no API key, rank 0 `--host 0.0.0.0 --port 8000`, ranks 1-3 `--headless`, `--network host`, container `glm53_fp8_dflash_tp4`, `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --trust-remote-code` | `cluster.env.example`, launcher |
| 20 | Host tier | kernel `6.17.0-1031-nvidia` (pinned, apt-held), driver `580.173.02`, booted with `iommu.passthrough=1` via the GRUB drop-in `zz-tp4-perf.cfg` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env), [`../node/host/README.md`](../node/host/README.md) |
| 21 | Host `/etc` set | `98-tp4-fabric.conf` (`net.ipv4.ip_forward=1`), `99-tp4-vm.conf` (`vm.swappiness=0`), `tp4-fabric-iptables.{sh,service}` (DOCKER-USER ACCEPT for the 4 fabric netdevs), `99-tp4-nopasswd` sudoers, per-node `40-cx7.yaml` netplan (MTU 9000) | [`../node/README-node-assets.md`](../node/README-node-assets.md), [`../node/etc/`](../node/etc/) |
| 22 | Fabric | switchless CX-7 RoCE ring `<ALIAS_RANK0>↔<ALIAS_RANK1>↔<ALIAS_RANK2>↔<ALIAS_RANK3>↔<ALIAS_RANK0>`, 4 DAC cables, 200 Gb/s, MTU 9000, one `/24` per link | [`fabric.md`](fabric.md) |
| 23 | Autostart | `tp4-autostart.service` on rank 0 only, `enabled`, runs `~/tp4/tp4ctl up` on whatever `~/tp4/cluster.env` holds | [`../node/tp4-autostart.service.example`](../node/tp4-autostart.service.example) |

Reference numbers for this snapshot (mean of two `run_ab.sh` passes, 2026-09-04 13:19 and 13:25):
structured ×1 **71.0**, prose ×1 **42.7** by hand / 40.3 harness, code ×1 **51.1**, four-stream
aggregate **228.5** (57.1 per stream), @1400 **63.4**, prefill 30k **2189.7**, prefill 100k
**2209.3** tok/s, needle 3/3 · 2/2, tool call PASS — [`gate.md`](gate.md) § Baseline.

---

## 2. Prerequisites below the IaC line

`scripts/bootstrap-node.sh` starts one layer **above** the OS. Everything in this section is
assumed to exist already; none of it is installed by this repo.

### 2.1 Hardware BOM

| Item | Quantity | Notes |
| --- | --- | --- |
| ASUS Ascent GX10 node | 4 | aliased `<ALIAS_RANK0>`..`<ALIAS_RANK3>`, rank 0..3 |
| NVIDIA GB10, `sm_121` (`compute_121`) | 1 per node | 128 GB unified memory per node |
| NVMe | 1 TB per node | the FP8 checkpoint alone is ~306 GiB; the fetch preflight wants ≥ 330 GiB free |
| ConnectX-7 (MT2910) | 1 per node | 2 physical QSFP ports, exposed as 4 netdevs over 2 PCIe links |
| QSFP28 DAC cable | 4 | the closed ring, no switch, no `1↔3` and no `2↔4` |
| 1 GbE management | 1 per node | `enP7s7`; plus Tailscale SSH for remote access |

Cabling, port identification, the IP plan and the re-addressing checklist: [`fabric.md`](fabric.md).

### 2.2 Software below the line, as measured today

Read-only on **rank 2** on 2026-09-04 (rank 0 runs the API, rank 3 was being rebuilt); every node is
expected to match.

| What | Measured value | Command |
| --- | --- | --- |
| OS | `Ubuntu 24.04.4 LTS` | `lsb_release -ds` |
| Kernel | `6.17.0-1031-nvidia` | `uname -r` |
| GPU / driver | `NVIDIA GB10`, driver `580.173.02` | `nvidia-smi --query-gpu=driver_version,name --format=csv,noheader` |
| Docker | `29.2.1` (build `a5c7197`) | `docker --version` |
| NVIDIA Container Toolkit | `1.20.0` (`nvidia-container-toolkit 1.20.0-1 arm64`) | `nvidia-ctk --version`, `dpkg -l nvidia-container-toolkit` |
| rdma-core / ibverbs-utils | `50.0-2ubuntu0.2` (arm64) | `dpkg -l rdma-core ibverbs-utils` |
| RoCE HCAs | `rocep1s0f0`, `rocep1s0f1`, fw `28.45.4028` | `ibv_devinfo \| grep -E 'fw_ver\|hca_id'` |
| CX-7 driver / firmware | `mlx5_core`, module version `6.17.0-1031-nvidia`, firmware `28.45.4028 (NVD0000000087)` | `ethtool -i enp1s0f0np0` |
| Tailscale | `1.102.3` | `tailscale version` |
| RAM | 121 GiB total, 114 GiB available | `free -g` |
| Disk (`/`, carries `/home`) | 916 G total, 571 G used, 299 G free (66%) | `df -h /home` |
| Python (host) | `3.12.3` | `python3 --version` |

Workstation (macOS, 2026-09-04):

| What | Measured value |
| --- | --- |
| macOS | `26.6.2` |
| bash | `GNU bash 3.2.57(1)-release` (arm64-apple-darwin) |
| ssh | `OpenSSH_10.3p1, LibreSSL 3.3.6` |
| python3 | `3.14.6` |
| git | `2.55.0` |
| `hf` CLI | **not installed** — and not needed here: it belongs on the nodes (rank 0 for the weight fetch, every node for the drafter) |
| `timeout` / `gtimeout` | **absent** — without it `tp4ctl` has no per-command guard |

The workstation must also have `scp`, `rsync`, `shasum`, `curl`, `ssh-keyscan` and ssh
reachability to `<ALIAS_RANK0>`..`<ALIAS_RANK3>`. **What to install on a fresh workstation**, given the two
absences above: `brew install coreutils` (provides `gtimeout`) and, on rank 0 rather than here,
`pip install -U 'huggingface_hub[cli]'`. The full list with the script that needs each tool is
[`install-from-zero.md`](install-from-zero.md) § 0.

### 2.3 Accounts and access

- **Hugging Face** — `zai-org/GLM-5.3-Flash` and `incoai/GLM-5.3-Flash-DFlash2` are public today,
  so anonymous download works. Authenticate on **rank 0 only** (`hf auth login` or `HF_TOKEN`) if
  rate-limited or if either repo becomes gated ([`weights.md`](weights.md) § 2).
- **Tailscale SSH** — `<ALIAS_RANK0>`..`<ALIAS_RANK3>` are MagicDNS names
  `<ALIAS_RANKn>.<TAILNET>.ts.net`; the tailnet
  identity maps to the local `<USER>` on every node ([`../AGENTS.md`](../AGENTS.md) § 2).
- **Same login account on all four nodes** — the autostart unit hard-codes one user for all ranks.

### 2.4 Not automated by this repo

| Not automated | Who does it | Why |
| --- | --- | --- |
| Ubuntu install | operator | out of scope, stated in [`install-from-zero.md`](install-from-zero.md) |
| NVIDIA driver install | operator | a driver difference is reported FAIL (drift), never auto-applied — [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) |
| Docker + GPU runtime install | operator | `bootstrap-node.sh --check` verifies `--gpus all` works, does not install it |
| Per-node netplan `node/etc/<ALIAS_RANKn>/40-cx7.yaml` | operator, from [`../node/etc/40-cx7.yaml.example`](../node/etc/40-cx7.yaml.example) | gitignored (site addressing); `bootstrap-node.sh` exits 3 if missing |
| `cluster.env` from `cluster.env.example` | operator | gitignored (site addressing) |
| First `/etc/sudoers.d/99-tp4-nopasswd` | `bootstrap-node.sh --apply --phase sudoers` | the only interactive-password step; chicken-and-egg with `sudo -n` |
| Every reboot (kernel pin, `iommu.passthrough=1`) | owner, rolling rank 3 → rank 0 | no script in this repo ever reboots a node |
| Weight purges | owner | thin disk margin; the fetch only *prints* purge commands |
| Container image build | upstream (`tonyd2wild`) | rebuild studied and rejected — [`image-rebuild-study.md`](image-rebuild-study.md) |

---

## 3. Ledger of kept changes

Oldest first. Every row is in production today unless the ledger says otherwise. "Verified by" is
the check that catches its absence. **"Where it lives" names files and dates, not commits**: this
repository was re-initialised for publication, so its history starts at a single release commit and
no earlier revision is reachable. Every rollback below is therefore expressed as the values to
restore, never as a revert of a past commit.

| Date | Change | What it does | Where it lives | Applied by | Verified by | Measured effect | Rollback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-01 | **FP8 checkpoint** chosen over NVFP4 and EXL3 | `zai-org/GLM-5.3-Flash` FP8 becomes the only lane | `cluster.env` (`MODEL_REPO`, `MODEL_DIR`); notes `bench-results/2026-09-01-3way-final.md`; single-lane collapse 2026-09-02 | `scripts/fetch-fp8-weights.sh` | `verify-node.sh` shard census (count read from the `-of-000NN` suffix) + needle gate | NVFP4 was fastest on decode (~121-124 structured) but FP8 was the only lane that never missed a needle (3/3, 2/2) and the only one that held 524288 | none — the NVFP4 and EXL3 lanes were removed from the repo on 2026-09-02 and are not in this history |
| 2026-09-01 | **Patched NCCL, switchless ring** | `NCCL_SKIP_TREE_CONNECT=1` makes `ncclTransportTreeConnect/PatConnect` return immediately, so the uncabled diagonals are never dialled; with `NCCL_ALGO=Ring` the cluster bootstraps | `~/nccl-patched/libnccl.so.2` on the nodes; sources vendored at [`../node/nccl/`](../node/nccl/) (2026-09-04) | `node/nccl/build.sh` then `node/nccl/install-nccl.sh` (atomic, sha-verified) | `verify-node.sh` sha256 vs `node/nccl/SHA256SUMS`; `build.sh` MATCH/DIFFERS vs `expected.env` | without it the ring does not boot at all; upstream measures ring serve ~24 tok/s vs ~3.3 on the socket fallback | re-run `install-nccl.sh` — [`agents/rollback.md`](agents/rollback.md) row *Patched NCCL library* |
| 2026-09-01 | **sm_121 sparse-attn indexer K-pool patch** | fixes the GB10 48-SM / 99 KB smem top-k path; without it the boot dies at 62/62 shards with `persistent_topk … >=128KB smem` | [`../node/sparse_attn_indexer_kpool_sm121.py`](../node/sparse_attn_indexer_kpool_sm121.py) (2026-09-01) | `scripts/deploy.sh` → `~/patches/sparse_attn_indexer_kpool.py`; the launcher always mounts it | launcher preflight (`PATCH_FILE` must exist); `verify-node.sh` checks `~/patches/*.py` | enabling condition, not a speed-up | none — removing it breaks the boot |
| 2026-09-01 | **Host `/etc` set** | `ip_forward=1`, `vm.swappiness=0`, DOCKER-USER ACCEPT rules for the 4 fabric netdevs, MTU 9000 netplan | [`../node/etc/common/`](../node/etc/common/), per-node netplan (2026-09-01) | `scripts/deploy-host.sh` (additive) → activation by `bootstrap-node.sh --apply` phase 3 | `deploy-host.sh --check` (`OK`/`DRIFT`/`MISSING`); `verify-node.sh` parses the sysctl values and the iptables rules | MTU 1500 on one port costs ~2.7× on decode | per-node restore: `./scripts/deploy-host.sh --host <ALIAS_RANKn>` |
| 2026-09-01 | **Rank-0 autostart** | rank 0 boots the whole cluster on `~/tp4/cluster.env`, tested or not | [`../node/tp4-autostart.service.example`](../node/tp4-autostart.service.example) (template committed 2026-09-02; unit installed on the node 2026-09-01, verified enabled 2026-09-02) | `bootstrap-node.sh --apply` phase 6 (`enable`, never `start`) | `verify-node.sh` autostart row; `systemctl is-enabled tp4-autostart` | operational, not a perf change | `systemctl disable tp4-autostart` (owner decision) |
| 2026-09-02 | **Image `sm121-v11-dflash2`** | v1-v9 SM121 patch chain (FP8 KV / CTA patches, indexer top-k fix) + the DFlash2 overlay (vLLM PR #52816); replaces the locally built `sm121-v8` | `cluster.env.example` (`IMAGE`); chain verified with `docker history` 2026-09-02 | `sudo docker pull` on the 4 nodes | launcher preflight `docker image inspect`; `verify-node.sh` image row | enabling condition for DFlash2; a wrong image dies at 62/62 with `pe_dim must be 64 for fp8_ds_mla` | pull and pin the previous tag (`sm121-v8`) + the two v8 indexer mounts (launcher header, fallback *a*) |
| 2026-09-02 | **DFlash2 drafter replaces the in-card MTP**, `SPEC_TOKENS=3` | MTP performs *k sequential* draft steps and rebuilds attention metadata between them; DFlash2 drafts the whole 8-token block in one pass. k=3 chosen for prose and coding | `cluster.env` (`SPEC_TOKENS`, `--speculative-config` built by the launcher); note `bench-results/2026-09-02-fp8-mtp-vs-dflash.md` (measured 2026-09-02, codified the same day) | `scripts/deploy.sh` + `./tp4ctl restart` | rank-0 init line `num_speculative_tokens=…`; `docs/gate.md` §3 | vs MTP: structured +50%, per-stream at c4 +40%, sustained long decode +32%, prose −5..−10% | superseded by row *adaptive draft length* below |
| 2026-09-02 | **`ASYNC_SCHEDULING=0`** | no `--async-scheduling` flag: A/B W3 measured it worse or equal (structured 72.4 vs 74.5-76.2, prose 28.8 vs 30.7-32.9, c4 147 vs 157-165). The engine still instantiates `AsyncScheduler`, which the adaptive-k patch requires | `cluster.env.example` (`ASYNC_SCHEDULING`) | `deploy.sh` + `restart` | rank-0 `adaptive-k: … async=True` line (an `async=False` means the policy disabled itself) | −3..−10% when enabled | set `ASYNC_SCHEDULING=1` (regression, do not) |
| 2026-09-02 (evening) | **Context lane 262144 × 6** | lowered from `524288 × 4`; owner decision, no measurement window. Above 262144 the launcher would inject `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — it no longer does | `cluster.env.example` (`MAX_MODEL_LEN`, `MAX_NUM_SEQS`), codified 2026-09-02 | `deploy.sh` + `restart` | `GET /v1/models` reports `max_model_len` 262144; `docs/gate.md` § 256K gate | more concurrency, less window; every pre-2026-09-03 number belongs to the other family | edit both knobs in `cluster.env` (+ the long-len env fires again above 262144) |
| 2026-09-02 | **Agent onboarding** | `AGENTS.md` becomes the agent entry point (hosts, login, recipe, safety rules); task runbooks under [`agents/`](agents/) | [`../AGENTS.md`](../AGENTS.md) (2026-09-02, restructured 2026-09-04) | n/a (documentation) | n/a | process, not performance | n/a |
| 2026-09-03 | **`--moe-backend triton`** | with DeepGEMM E8M0 on, `triton_deep_gemm_moe` picks DeepGEMM at every M and upstream's `N ≤ 512 → Triton` fast path never fires (N=512 per rank at TP4); the default Triton fused-MoE kernel beats it on GB10 at every M. Dense FP8 linears stay on DeepGEMM | `cluster.env.example` (`EXTRA_VLLM_ARGS`), promoted 2026-09-03 and confirmed in production the same day; note `bench-results/2026-09-03-w2a-moe-triton.md` | `deploy.sh` + `restart` | rank-0 `Using TRITON Fp8 MoE backend` | decode **+7..+16%** (prose +12%, code +8%, structured +9%, c4 +15%, @1400 +9%), prefill **+5%** (30k; 100k +3%) | [`agents/rollback.md`](agents/rollback.md) row *Triton MoE backend* (drop `--moe-backend triton` from `EXTRA_VLLM_ARGS`) |
| 2026-09-03 | **Host `iommu.passthrough=1`** | GRUB drop-in puts the SMMU in identity mode, so DMA from the CX-7 fabric NICs skips translation. Named `zz-` so it is sourced after the vendor `iommu.cfg` (`=0`); the kernel honours the last occurrence | [`../node/etc/default/grub.d/zz-tp4-perf.cfg`](../node/etc/default/grub.d/zz-tp4-perf.cfg) + [`../node/host/tp4-iommu.sh`](../node/host/tp4-iommu.sh) (2026-09-03, kept after the H3 window) | `deploy-host.sh` push, then `--run tp4-iommu.sh --apply` (runs `update-grub`), then an owner-driven rolling reboot | `tp4-iommu.sh --status` must read `passthrough (cmdline) / drop-in installed / grub.cfg in sync`; `verify-node.sh` runs it | structured **+9%**, c4 **+8%**, code **+4..+7%**, @1400 +5%; prose and prefill flat | [`agents/rollback.md`](agents/rollback.md) row *Host knob* (`--revert` + rolling reboot; exit 4 = do **not** reboot) |
| 2026-09-04 | **Hybrid GB10 MoE config** | JSON produced by the Ray-free tuner (v2: skewed expert routing, persistent Triton cache, selective merge) on rank 1 the night of 2026-09-03; **hybrid** = tuned entries for batch 1,2,4,8 only, vLLM defaults for 16-48 and 1024-8192. The fully tuned file was rejected the same window (−7% at concurrency 4) and kept as `bench-results/moe-tune-2026-09-03/E=288,N=512,…json.full-tuned` | [`../node/moe-configs/`](../node/moe-configs/) (full JSON and hybrid JSON, 2026-09-04), tuner [`../node/moe-tune/`](../node/moe-tune/); promoted and confirmed in production 2026-09-04; note `bench-results/2026-09-04-w2c-moe-tuned.md` | `deploy.sh` pushes `node/moe-configs/*.json` to `~/tp4/moe-configs/`; the mount is one `-v` pair in `EXTRA_DOCKER_ENV` | rank-0 `Using configuration from …E=288,N=512,device_name=NVIDIA_GB10…json`; launcher preflight checks the `-v` source exists | single stream **+6..+10%** vs the vLLM default (code +5-9%, structured +2%); c4 −3..−5% (inside noise), prose and prefill flat | [`agents/rollback.md`](agents/rollback.md) row *Hybrid GB10 fused-MoE kernel config* (the rollback ledger keeps the long name) — remove **only** that `-v` pair from `EXTRA_DOCKER_ENV` |
| 2026-09-04 | **Kernel `6.17.0-1031-nvidia` on all four nodes** (H4) | ranks 0-1 aligned to ranks 2-3; the previous kernel stays as the GRUB fallback | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`KERNEL`, `KERNEL_PKGS`, `KERNEL_PKGS_EXTRA`), aligned 2026-09-04 | rolling reboot rank 3 → rank 0, `./tp4ctl down` first | `verify-node.sh` compares `uname -r` with `versions.env` | hygiene: one kernel everywhere, pinned; no perf claim | reboot into the previous GRUB entry — [`agents/rollback.md`](agents/rollback.md) row *Kernel version* |
| 2026-09-04 | **apt holds on the version-locked kernel packages** | `apt-mark hold` on the installed `linux-image-/linux-modules-/linux-modules-nvidia-…/linux-headers-` set for that version — **never** on the `*-hwe-24.04` metas (that makes `apt full-upgrade` unsolvable). The userspace driver is deliberately not held | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) + [`../scripts/bootstrap-node.sh`](../scripts/bootstrap-node.sh) (2026-09-04) | `bootstrap-node.sh --apply` phase 1 (`--only kernel-holds`) | `verify-node.sh` apt-holds row | prevents silent kernel/driver-module drift | `sudo apt-mark unhold <pkgs>` (owner decision) |
| 2026-09-04 | **Adaptive draft length, v1** | `SPEC_TOKENS=5` + `AdaptiveKScheduler(AsyncScheduler)`: per request, an EMA of a 0/1 position-3 acceptance signal with a hysteresis band decides whether 3 or 5 of the 5 drafted tokens are verified. Any exception degrades to base `AsyncScheduler` behaviour. The dynamic-SD table is kept only for FULL graph capture at both verify sizes | [`../node/patches/adaptive_k_scheduler.py`](../node/patches/adaptive_k_scheduler.py) + [`../node/patches/test_adaptive_k_policy.py`](../node/patches/test_adaptive_k_policy.py) (rebased on `AsyncScheduler` 2026-09-04, plus a same-day logger fix); **promoted 2026-09-04 12:05**, confirmed in production the same afternoon; design [`adaptive-k.md`](adaptive-k.md) | `deploy.sh` pushes `node/patches/*.py` to `~/patches/`; `cluster.env` mounts it at `/opt/tp4` and selects it with `--scheduler-cls` | rank-0 `adaptive-k: AdaptiveKScheduler active (enabled=1 k_lo=3 k_hi=5 …) async=True`, init line with `num_speculative_tokens=5` and the table; `python3 node/patches/test_adaptive_k_policy.py` on the workstation | vs fixed k=3: structured **+21%**, c4 **+56%** (per stream +46%), @1400 **+17%**, code **+6%**, prose **−6% harness / 0% by hand**, prefill +3%/+1.5%; **costs**: long-context essay decode −15..−20% on one size, FULL decode graphs lost on mixed-k steps | [`agents/rollback.md`](agents/rollback.md) row *Adaptive draft length* — `SPEC_TOKENS=3`, `SPEC_EXTRA_JSON=""`, drop `--scheduler-cls` and only the adaptive-k entries of `EXTRA_DOCKER_ENV`; **keep the MoE mount** |
| 2026-09-04 | **IaC package** | the repo becomes able to rebuild a node: `bootstrap-node.sh` (6 phases, `--check`/`--apply`/`--phase`/`--only`), `verify-node.sh` (read-only PASS/FAIL audit of all 4 nodes, `--live`), `deploy.sh` / `deploy-host.sh` with `--check`/`--host`, `mirror-snapshot.sh`, the vendored NCCL build, and the ordered runbook | [`../scripts/`](../scripts/) + [`../node/bootstrap/`](../node/bootstrap/) + [`../node/nccl/`](../node/nccl/) + [`install-from-zero.md`](install-from-zero.md) (2026-09-04, docs restructured the same day) | n/a | `./scripts/verify-node.sh` exit code; `deploy.sh --check` and `deploy-host.sh --check` | reproducibility, not performance | n/a |
| 2026-09-04 | **NVIDIA Container Toolkit aligned to `1.20.0-1`** | ranks 0-1 were still at `1.19.1-1` while ranks 2-3 were at `1.20.0-1`; the drift was found by the new `NVIDIA_CTK_MIN` pin during the from-zero rehearsal and closed at 14:25 | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`NVIDIA_CTK_MIN`) | `sudo apt-get install nvidia-container-toolkit=1.20.0-1 nvidia-container-toolkit-base=1.20.0-1 libnvidia-container-tools=1.20.0-1 libnvidia-container1=1.20.0-1` on ranks 0 and 1 | `verify-node.sh` NVIDIA Container Toolkit row (numeric minimum, FAIL below it) | hygiene: one toolkit version on the four nodes; no perf claim | install the previous version explicitly (owner decision — the pin then reports FAIL) |
| 2026-09-04 | **From-zero rehearsal on rank 3** (13:56 → 14:16) | rank 3 stripped back to OS + driver + docker (sysctl drop-ins, iptables unit and script, fabric netplan, grub drop-in, apt holds, `~/tp4`, `~/patches`, `~/nccl-patched`, `~/vllm-cache`, rank-0 key) and rebuilt from this repo alone: `--check` listed 11 TODO, `--apply` closed them, the re-`--check` read **22 PASS / 0 TODO / 0 FAIL**. The patched NCCL was **rebuilt** from `node/nccl/` on rank 1 (2 min; 165 exported symbols identical, +8 bytes of build id) and the rebuild **adopted as canonical** | [`install-from-zero.md`](install-from-zero.md) § Verified; [`../node/nccl/SHA256SUMS`](../node/nccl/SHA256SUMS) + [`../node/nccl/expected.env`](../node/nccl/expected.env) (updated 2026-09-04) | `bootstrap-node.sh --apply` (all phases), `deploy.sh --host <ALIAS_RANK3>`, `node/nccl/build.sh` then `install-nccl.sh --expect-sha` | `deploy-host.sh --check` (every managed file matches), `tp4ctl up` → `/health` 200 in 16 min with a cold JIT cache, sanity gate `Rome` + tool-call PASS, bench pass inside the noise band | the runbook itself, not a knob: no rollback. The pre-rebuild library is still on the build node |

### 3.1 Closed experiments — do not retry

Each row was measured and rejected; the overlay is kept only as the record, with its verdict in
the file header.

| Date | Knob / arm | Verdict | Record |
| --- | --- | --- | --- |
| 2026-09-01 | NVFP4 (RedHatAI) and EXL3 4bpw (MiaAI-Lab) lanes | **DROPPED** — NVFP4 faster on decode but missed needles; EXL3 ~40% slower on prefill | `bench-results/2026-09-01-3way-final.md` |
| 2026-09-02 | in-card MTP head (k=3, k=5) | **DROPPED** — sequential draft steps + attention-metadata rebuild | `bench-results/2026-09-02-fp8-mtp-vs-dflash.md` |
| 2026-09-02 | `--async-scheduling` on | **REJECTED** — worse or equal on every axis | `bench-results/2026-09-02-fp8-mtp-vs-dflash.md` |
| 2026-09-02 | DSpark (arXiv 2607.05147) | **NOT APPLICABLE** — no drafter exists for GLM-5.3-Flash | `README.md` § Historical context |
| 2026-09-03 | GPU SM clock lock (`nvidia-smi -lgc`), H1 | **INEFFECTIVE** — driver accepts it, clocks unchanged (2411 → 2411 idle; ~2540 MHz under load either way) | [`../node/host/README.md`](../node/host/README.md), `bench-results/2026-09-03-e0-observations.md` |
| 2026-09-03 | `VLLM_USE_DEEP_GEMM_E8M0=0` (W2b) | **FAILED on correctness** — boots, generates degenerate text. Never re-run | `experiments/2026-09-03-w2b-e8m0-off.env` |
| 2026-09-03 | GPUDirect RDMA over C2C (W4a) | **CLOSED, impossible** — `DMA_BUF_SUPPORTED=0`, `GPU_DIRECT_RDMA_SUPPORTED=0`, no peermem API, rdma-core 50 | `experiments/2026-09-03-w4a-gdr-c2c.env` |
| 2026-09-03/04 | fully tuned MoE JSON (all batch sizes) | **REJECTED** — repeatable −7% at concurrency 4; kept as `bench-results/moe-tune-2026-09-03/E=288,N=512,…json.full-tuned` | `bench-results/2026-09-04-w2c-moe-tuned.md` |
| 2026-09-04 | fixed `SPEC_TOKENS=4 / 5 / 7` | **MIXED, not promoted** — prose −12..−19% at every k | `experiments/2026-09-04-spec{4,5,7}.env` |
| 2026-09-04 | drafter TP1 (`draft_tensor_parallel_size`) | **NEUTRAL** — every axis inside the night noise band | `experiments/2026-09-04-draft-tp1.env` |
| 2026-09-04 | `BATCHED_TOKENS=16384` | **NO GAIN** — prefill +0.5..+3% (noise), decode −4..−6% | `experiments/2026-09-04-batched16k.env` |
| 2026-09-04 | `--moe-backend marlin` (W3a) | **CLOSED** — decode within noise, prefill −6% | `experiments/2026-09-04-w3a-marlin.env` |
| 2026-09-04 | `NCCL_PROTO=LL` | **CLOSED** — no gain at 8-32 KB, half the bandwidth from 1 MB up. Never set `NCCL_PROTO` | `experiments/2026-09-04-ncclbench-ll.env` |
| 2026-09-04 | default (untuned) MoE config A/B | **hybrid confirmed** — default slower on every single-stream axis | `experiments/2026-09-04-ab-default-moe.env` |
| 2026-09-04 | adaptive-k variants v2, v3, v4, v5 | **MEASURED, not promoted** — owner chose v1 (usage mix: concurrency + code) | `experiments/2026-09-04-adaptive-k-v{2,3,4,5}.env`, [`adaptive-k.md`](adaptive-k.md) § Decision |
| 2026-09-03 | rebuilding the container image | **NO-GO** — expected gain ≈ 0, breakage risk real | [`image-rebuild-study.md`](image-rebuild-study.md) |

---

## 4. Reproduction path

Fifteen steps; each one names the file that owns it. The detail — commands, timings, failure
modes — is [`install-from-zero.md`](install-from-zero.md), whose step numbers these mirror.

1. Hardware, cabling and OS/driver/docker in place → § 2 above, [`fabric.md`](fabric.md).
2. `cp cluster.env.example cluster.env`, fill topology and paths → `cluster.env.example`.
3. Create each `node/etc/<ALIAS_RANKn>/40-cx7.yaml` → [`../node/etc/40-cx7.yaml.example`](../node/etc/40-cx7.yaml.example).
4. `./scripts/bootstrap-node.sh <ALIAS_RANKn> --rank <n> --check` on all four → read-only TODO list.
5. Passwordless sudo → `--apply --phase sudoers` (the only interactive step).
6. `/etc` assets → `./scripts/deploy-host.sh --host <ALIAS_RANKn>`, activation by `--apply` phase 3.
7. Kernel pin + `apt-mark hold` → [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env), then a rolling reboot rank 3 → rank 0.
8. SSH mesh from rank 0 (rank 0 to itself included) → `bootstrap-node.sh` phase 4.
9. Patched NCCL → `node/nccl/build.sh` then `node/nccl/install-nccl.sh`, [`nccl.md`](nccl.md).
10. Container image → `docker pull` on the four nodes, [`weights.md`](weights.md) § 1.
11. Weights + drafter → `scripts/fetch-fp8-weights.sh` (fabric fan-out), [`weights.md`](weights.md) §§ 0, 2, 2a, 3.
12. Runtime files → `./scripts/deploy.sh`; host tier → `./scripts/deploy-host.sh --run tp4-iommu.sh --apply` + reboot.
13. Autostart on rank 0 → `bootstrap-node.sh --apply` phase 6.
14. `./scripts/verify-node.sh`, `./tp4ctl fabric-check`, `./tp4ctl up` (~16 min to `/health` 200).
15. Gate and benchmark → § 5 below.

## 5. Verification

| Layer | Command | Passes when |
| --- | --- | --- |
| Node audit | `./scripts/verify-node.sh` (add `--live` after boot) | every row PASS, exit 0. Covers kernel/driver vs `versions.env`, docker GPU access, `/dev/infiniband` + `ibv_devinfo`, apt holds, sysctl, iptables `DOCKER-USER`, `sudo -n`, MTU 9000 / 200000 Mb/s, `fabric-check`, ssh mesh, autostart, `deploy.sh --check` **and** `deploy-host.sh --check`, `tp4-iommu.sh --status`, rdma-core, shards, drafter, image, NCCL sha, `~/patches/*.py`, every `-v` source of `EXTRA_DOCKER_ENV` |
| Fabric | `./tp4ctl fabric-check` | MTU/IP per node + 8-way jumbo ping matrix green. Exit 0 = every fabric interface present and all jumbo pings OK, 1 = any degradation; `tp4ctl up` refuses to start on 1 |
| Boot signatures | rank-0 log, 4 lines | `adaptive-k: AdaptiveKScheduler active (… async=True)`, `Using configuration from …NVIDIA_GB10…json`, `Using TRITON Fp8 MoE backend`, container `glm53_fp8_dflash_tp4` on all 4 ranks — [`agents/status-check.md`](agents/status-check.md) |
| Sanity gate | within **2 minutes** of `/health` 200 | one coherent answer at temperature 0 with thinking off + the tool-call gate — [`agents/bench-protocol.md`](agents/bench-protocol.md) |
| Acceptance gate | [`gate.md`](gate.md) §§ 1-3 | needle 30k contains `ZULU-7741`; tool call returns `get_weather` with `Milan`; decode ≥ 30 tok/s (a floor, not a target) |
| 256K gate | [`gate.md`](gate.md) § 256K gate | `/v1/models` reports `max_model_len` 262144; needle at ~250k recovered |
| Performance | `RUNS=3 CONCURRENCY=4 LONG_DECODE=1 scripts/bench/run_ab.sh <label>` ×2 + `bench_decode.py --prompt code/prose` | within the noise band of the `BASE_*` reference values in [`../scripts/bench/run_ab.sh`](../scripts/bench/run_ab.sh) (structured 71.0, prose 40.3, c4 228.5, @1400 63.4, prefill 2189.7 / 2209.3) — ±3-5% on decode medians, ±2-3% on prefill |

A node that boots but sits outside the baseline is usually a missing piece of the `/etc` or kernel
steps (sysctl, MTU) or of the deploy step (the MoE mount, the SMMU mode): both show up first in
the four-stream aggregate and in prefill, not in single-stream decode.

---

## 6. Code-side pins

What is pinned in a file, and what enforces it.

| Pinned value | Pinned in | Enforced by |
| --- | --- | --- |
| Kernel `6.17.0-1031-nvidia` + its version-locked package set | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`KERNEL`, `KERNEL_PKGS`, `KERNEL_PKGS_EXTRA`) | `scripts/verify-node.sh` (`uname -r`), `scripts/bootstrap-node.sh` (`apt-mark hold`) |
| Driver `580.173.02` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`DRIVER`) | `verify-node.sh` / `bootstrap-node.sh` — reported **FAIL (drift)**, never auto-applied |
| `rdma-core` ≥ 50 | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`RDMA_CORE_MIN`) | `verify-node.sh` |
| OS release `Ubuntu 24.04.4 LTS` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`OS_RELEASE`) | `verify-node.sh` — `lsb_release -ds`, exact match, **FAIL** on a different release |
| Docker engine ≥ `29.2.1` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`DOCKER_MIN`) | `verify-node.sh` — `docker version --format '{{.Server.Version}}'`, numeric, **FAIL** below the minimum |
| NVIDIA Container Toolkit ≥ `1.20.0` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`NVIDIA_CTK_MIN`) | `verify-node.sh` — `nvidia-ctk --version`, numeric, **FAIL** below the minimum |
| CX-7 firmware `28.45.4028` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`CX7_FW`) | `verify-node.sh` — `ibv_devinfo` `fw_ver` per HCA, **WARN** on a mismatch (this repo never flashes an HCA; the gates are the port state and MTU/speed rows) |
| Tailscale ≥ `1.102.3` | [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) (`TAILSCALE_MIN`) | `verify-node.sh` — **WARN** only: access path, no functional dependency |
| Checkpoint identity: `config.json` sha256 + shard count | the weights themselves; revision slots `MODEL_REV` (also in `cluster.env`) and `DRAFT_REV` in [`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) | `verify-node.sh` `weights fingerprint` row: sha256 printed in full for the ledger, shard count checked against the `-of-000NN` suffix. The revision rows SKIP while the slots are empty; once `MODEL_REV` is pinned it is compared with the `.glm53-fp8-synced` fan-out marker |
| NCCL sha256 `1ddc3240396a9b3a1e4fa3e54e129d099261106ce3b9263ac3fdc3e070713bd5` (the 2026-09-04 rebuild, canonical since that day) | [`../node/nccl/SHA256SUMS`](../node/nccl/SHA256SUMS) | `verify-node.sh`, `node/nccl/install-nccl.sh` (before and after the copy) |
| NCCL size / symbol count / base tag / base commit | [`../node/nccl/expected.env`](../node/nccl/expected.env) | `node/nccl/build.sh` (MATCH/DIFFERS per field; asserts the checkout commit) |
| Image tag `sm121-v11-dflash2` | `cluster.env.example` (`IMAGE`) | launcher preflight (`docker image inspect`), `verify-node.sh` |
| MoE kernel config JSON | [`../node/moe-configs/`](../node/moe-configs/) + the `-v` pair in `EXTRA_DOCKER_ENV` | launcher preflight (every `-v` source must exist), rank-0 `Using configuration from …` signature, `deploy.sh` sha256 |
| Container patches (`adaptive_k_scheduler.py`, indexer) | [`../node/patches/`](../node/patches/), [`../node/sparse_attn_indexer_kpool_sm121.py`](../node/sparse_attn_indexer_kpool_sm121.py) | `deploy.sh` `ast.parse` + sha256 on every `.py` it pushes; launcher preflight; rank-0 `adaptive-k:` signature |
| `/etc` set, grub drop-in, sudoers, netplan | [`../node/etc/`](../node/etc/) | `deploy-host.sh --check` (`OK`/`DRIFT`/`MODE-DRIFT`/`MISSING`/`IDENTITY-MISMATCH`), `verify-node.sh` |
| apt holds | [`../scripts/bootstrap-node.sh`](../scripts/bootstrap-node.sh) phase 1, driven by `versions.env` | `bootstrap-node.sh --check`, `verify-node.sh` |
| Vendored file checksums (`flusher-unconditional.sh`, indexer patch, `node/moe-tune/vendor/benchmark_moe.py`) | [`../node/README-node-assets.md`](../node/README-node-assets.md), [`../node/moe-tune/README.md`](../node/moe-tune/README.md) | manual comparison; `deploy.sh` sha256 for what it pushes |
| Performance baseline | `BASE_*` in [`../scripts/bench/run_ab.sh`](../scripts/bench/run_ab.sh), [`gate.md`](gate.md) § Baseline, [`../bench-results/milestones.json`](../bench-results/milestones.json) | `run_ab.sh` prints the delta against them; `scripts/bench/perf-table.py` regenerates the README table from the committed JSONs |

### 6.1 NOT yet pinned in code

What is still carried by no file that a check reads. Everything else measured in § 2.2 now lives in
[`../node/bootstrap/versions.env`](../node/bootstrap/versions.env) and is checked by
`verify-node.sh` (table above).

- **Model checkpoint revision** — the `MODEL_REV` slot exists (`cluster.env`, mirrored in
  `versions.env`) but is **empty**, so the fetch follows the card's HEAD. It can only be filled by
  an owner decision: `hf download` leaves no git checkout to read a sha back from, so the value has
  to be copied from the model card's *Files and versions*. Until then the effective identity is the
  `weights fingerprint` row (`config.json` sha256 + 62 shards), identical across nodes.
- **Drafter revision** — same, `DRAFT_REV` empty, and no fetch script consumes it yet: the drafter
  is downloaded with a plain `hf download` ([`weights.md`](weights.md) § 3), which records nothing on
  the node. Its `config.json` sha256 is printed by the `drafter revision` row.
- **CX-7 firmware string suffix** — `CX7_FW` pins `28.45.4028`; the vendor part number
  `(NVD0000000087)` printed by `ethtool -i` is not compared (`ibv_devinfo` does not report it).
- **Workstation toolchain** — `timeout`/`gtimeout` is absent on the current workstation, so
  `tp4ctl`'s per-command guards do not exist there; `hf` is absent (needed only on rank 0).
