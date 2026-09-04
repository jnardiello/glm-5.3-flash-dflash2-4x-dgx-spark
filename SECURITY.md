# Security

This repository is the infrastructure-as-code of a personal, four-node inference cluster. It is
designed for a **trusted LAN**, not for the internet, and it ships **no authentication**. Read this
before pointing it at anything you care about.

## Trusted LAN only, no authentication

- Rank 0 serves the OpenAI-compatible API on `0.0.0.0:<API_PORT>` (default `8000`) and every rank
  runs with `--network host`: the container shares the node's network namespace, so nothing between
  the process and the LAN filters traffic.
- **No API key, no TLS, no rate limit, no per-caller identity.** `cluster.env.example` sets no key
  and the launcher passes none; any client that can reach the port gets full use of the model,
  including the ability to exhaust the KV pool and deny service to everyone else.
- The RoCE fabric carries unauthenticated NCCL traffic on four point-to-point /24s, and
  `node/etc/common/tp4-fabric-iptables.sh` inserts blanket `ACCEPT` rules for those interfaces in
  the `DOCKER-USER` chain.
- **Deploy it on a network you control**, behind a firewall or on a private/VPN segment. If you need
  it reachable from elsewhere, put an authenticating reverse proxy in front of rank 0 and do not
  expose the fabric interfaces at all.

## Root-equivalent access on the nodes

The deploy user's sudoers drop-in (`node/etc/common/99-tp4-nopasswd.example`) grants
`NOPASSWD:ALL`. That is **root-equivalent access without a password** on all four nodes, and it is
what `tp4ctl` and the launcher rely on (docker, sysctl, `drop_caches`, `systemd-run`, netplan). Rank
0 additionally holds a passphrase-less ssh key that reaches the other three nodes, and its autostart
unit boots the whole cluster from `~/tp4/cluster.env` on power-on, tested or not. Anyone who reaches
the deploy account on any node owns the cluster.

## Threat model in five lines

1. **In scope:** accidental exposure of an unauthenticated endpoint, and operator mistakes that
   drift a node away from the recipe (detected by `scripts/verify-node.sh` and the `--check` modes).
2. **Assumed trusted:** the LAN, the fabric, the deploy account, the workstation running these
   scripts, and everyone who can already ssh to a node.
3. **Out of scope:** authenticating or authorizing API callers, multi-tenancy, prompt/output
   filtering, encrypting fabric or management traffic, and hardening the container itself.
4. **Supply chain:** the container image, model weights, drafter and NCCL sources are third-party
   artefacts; only the NCCL library and the pushed files are sha256-verified (`node/nccl/SHA256SUMS`,
   `scripts/deploy.sh`), the upstream image tag is trusted as published.
5. **Secrets:** none are committed. `cluster.env`, the per-node netplan files, the rendered sudoers
   and autostart units are gitignored, and `scripts/mirror-snapshot.sh` refuses to publish a
   snapshot that still contains a site value.

## Reporting a problem

Open a GitHub issue at
[the GitHub issue tracker](https://github.com/jnardiello/tp4-glm53-fp8-gx10/issues). If the
problem is a vulnerability that should not be public first, use GitHub's private vulnerability
reporting on the same repository. This is an unsupported personal project: there is no SLA, no
security release channel and no guarantee of a fix — expect best effort only.
