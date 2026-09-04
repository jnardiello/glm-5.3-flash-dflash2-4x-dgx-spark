# Weights, drafter and image — first install on a node

What has to exist on every node before the first `./tp4ctl up`:

| Artifact | Path on the node | Size | Source |
| --- | --- | --- | --- |
| FP8 model | `~/glm53-flash-fp8-zai` (flat dir, 62 shards) | ~306 GiB | `zai-org/GLM-5.3-Flash` |
| DFlash2 drafter | `~/glm53-dflash2-draft` | 2.34 GB | `incoai/GLM-5.3-Flash-DFlash2` |
| Container image | local docker | — | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` |
| Patched NCCL | `~/nccl-patched/libnccl.so.2` | — | `docs/nccl.md` |
| Indexer patch | `~/patches/sparse_attn_indexer_kpool.py` | — | pushed by `scripts/deploy.sh` |

The launcher preflights all of them and refuses to start if one is missing.

## 0. Disk census — do this first, and decide by hand

The nodes have 1 TB with little margin. The disk preflight in `scripts/fetch-fp8-weights.sh`
requires **≥ 330 GiB free on `$HOME` of every node** for the FP8 fetch; if the space is not
there, it only *prints* the manual purge commands and never runs them. Purging weights is never
automated: take the census, then decide.

```sh
. ./cluster.env
for n in $NODES; do
  echo "== $n"
  ssh $n 'df -h / ; du -sh ~/glm53* 2>/dev/null'
done
```

## 1. Container image

Already a superset of the older locally built `sm121-v8` chain (verified with `docker history`
on 2026-09-02: same `patch_v8_fp8`/v7/PDL/dependency-pin steps, plus the chain up to v9 and the
DFlash2 overlay), so there is no local build to do — just pull it on every node:

```bash
. ./cluster.env
for n in $NODES; do
  ssh $n 'sudo docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2'
done
```

## 2. FP8 weights (flat dir)

**The `hf` CLI has to exist on the nodes, not on the workstation**: on rank 0 for the FP8 fetch
below (the only node that talks to the Hub for the 306 GiB), and on every node for the drafter
download of § 3, which is pulled node-side:

```bash
. ./cluster.env
for n in $NODES; do ssh $n "pip install -U 'huggingface_hub[cli]'"; done
ssh <ALIAS_RANK0> 'hf version'
```

**Hugging Face authentication.** `zai-org/GLM-5.3-Flash` and `incoai/GLM-5.3-Flash-DFlash2` are
both **public** today, so an anonymous download works and no token is required. Authenticate on
**rank 0 only** — it is the single node that talks to the Hub, ranks 1..3 receive the shards over
rsync — if you hit a rate limit or if either repo ever becomes gated: `hf auth login` (stores the
token in `~/.cache/huggingface/token`) or `export HF_TOKEN=hf_…` in the shell that runs the fetch.
`scripts/fetch-fp8-weights.sh` carries no credential of its own: it just calls `hf download` and
inherits the environment. Optional speed-up for that first leg, not for the fan-out:
`pip install hf_transfer` and `HF_HUB_ENABLE_HF_TRANSFER=1`.

```bash
rsync -a scripts cluster.env <ALIAS_RANK0>:~/tp4/   # ~/tp4/scripts does not exist until you push it (deploy.sh does not copy it)
ssh <ALIAS_RANK0> "cd ~/tp4 && TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' ./scripts/fetch-fp8-weights.sh"
```

It downloads `zai-org/GLM-5.3-Flash` into `$HOME/glm53-flash-fp8-zai` (**62 shards, ~306 GiB**),
then fans it out to ranks 1..3 over rsync with a `.glm53-fp8-synced` marker. `--dry-run` lists
the df/ssh/rsync it would run without executing anything (62 shards expected). The optional
exclusions (`HF_EXCLUDE=`) are passed to `hf download` as `--exclude`, **not** as positional
arguments (hf would read those as allow_patterns → 0 files). Recommended after the download:
`sha256sum model-00022-of-00062.safetensors` against the HF page.

### 2a. Fan-out over the CX7 fabric (recommended)

The default fan-out travels on the 1GbE management LAN (~110 MB/s ⇒ ~45-50 min per peer for
~306 GiB). The switchless ring offers **direct** 200G CX7 links only towards rank1
(`<FABRIC_IP_RANK1_F0>`, link L1) and rank3 (`<FABRIC_IP_RANK3_F1>`, link L4); rank2 is reachable
only through a relay from rank1 (`<FABRIC_IP_RANK2_F1>`, link L2). With `XFER_HOSTS` the data
copy uses the fabric addresses; with `RELAY_RANK2=1` the rank2 leg starts **from rank1** (it
needs node-to-node ssh keys rank1→rank2: if the probe fails it falls back to mgmt).
Prerequisite: `~/tp4/scripts` on the head (the same `rsync -a scripts cluster.env <ALIAS_RANK0>:~/tp4/`
as the previous step, if not done yet):

```bash
ssh <ALIAS_RANK0> "cd ~/tp4 && TP4_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<MGMT_IP_RANK1> <USER>@<MGMT_IP_RANK2> <USER>@<MGMT_IP_RANK3>' XFER_HOSTS='<USER>@<MGMT_IP_RANK0> <USER>@<FABRIC_IP_RANK1_F0> <USER>@<MGMT_IP_RANK2> <USER>@<FABRIC_IP_RANK3_F1>' RELAY_RANK2=1 ./scripts/fetch-fp8-weights.sh"
```

The fabric addresses above are not examples: they are the fixed ring plan documented in
`docs/fabric.md`. `RELAY_DEST` (the destination of the relay leg, rank1→rank2 on L2) and
`FABRIC_TARGETS` both live in `cluster.env`, so with `RELAY_RANK2=1` there is nothing else to
pass on the command line.

The control plane (marker/mkdir/probe) always stays on mgmt via `TP4_HOSTS`; `XFER_HOSTS` only
drives the data copy. Expectations: **3-5×** on the direct legs (ssh cipher/NVMe bound, not 200G
line rate), rank2 through the relay ~10-15 min instead of ~45-50. Check first: `--dry-run`
prints the data-path table, the relay probe and the commands of the relay leg. What is at stake:
~306 GiB × 3 peers, i.e. ~2h in total over mgmt vs ~25-35 min over the fabric.

## 3. DFlash2 drafter

Small enough that the fabric fan-out is not worth it — download it directly on each node:

```bash
. ./cluster.env
for n in $NODES; do
  ssh $n 'hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir ~/glm53-dflash2-draft'
done
```

`incoai/GLM-5.3-Flash-DFlash2` (2.34 GB bf16, 5 layers, `block_size` 8, target layer ids
`[5,14,24,33,42]`) declares `base_model = zai-org/GLM-5.3-Flash`, i.e. exactly the FP8 checkpoint
served here. Its `block_size` 8 is the **ceiling** on the draft length: `num_speculative_tokens`
can be 1..7, never 8. Production does **not** run at the ceiling — `cluster.env` carries
`SPEC_TOKENS=5` (the drafter always drafts 5) and the `AdaptiveKScheduler` patch then verifies 3
or 5 of them per request, from that request's own acceptance history: [`adaptive-k.md`](adaptive-k.md).

> **License: CC BY-NC-ND 4.0** — non-commercial use, no derivatives. If the cluster is ever used
> commercially, the drafter has to be cleared with inco.ai first.

## 4. First-boot checklist

The `zai-org` card is **not verified drop-in** against every downstream variant of this recipe.
On the first boot, check by hand:

| Check | How |
| --- | --- |
| FP8 block in config.json | `python3 -c "import json;c=json.load(open('$HOME/glm53-flash-fp8-zai/config.json'));print(c.get('quantization_config'))"` — an FP8 quant (weight scales) is expected; if absent you would need a `--quantization` flag the recipe does not pass |
| Chat template | is `$HOME/glm53-flash-fp8-zai/chat_template*.jinja` present in the dir? The recipe does not mount one by hand; if absent, the one in the tokenizer config applies, otherwise override it with `EXTRA_VLLM_ARGS='--chat-template /model/chat_template.jinja'` |
| Vision tower (multimodal) | present: `preprocessor_config.json`, image/video processor, tower files; if missing, the server starts but `image_url` fails |
| generation_config | `generation_config.json` exists (the card's sampling defaults) |
| All 62 shards | `ls $HOME/glm53-flash-fp8-zai/model-*-of-00062.safetensors \| wc -l` |
| Drafter | `ls $HOME/glm53-dflash2-draft/config.json` and `python3 -c "import json;print(json.load(open('$HOME/glm53-dflash2-draft/config.json')).get('base_model'))"` → `zai-org/GLM-5.3-Flash` |

Expected timings on the first boot: 62 shards ≈ 10 min of weight loading, plus warmup and CUDA
graph capture — do not retry the `up` while it is running. Boot to `/health` 200 measured on
2026-09-02: **981 s**.

Once the boot is green, run the acceptance gate: `docs/gate.md`.
