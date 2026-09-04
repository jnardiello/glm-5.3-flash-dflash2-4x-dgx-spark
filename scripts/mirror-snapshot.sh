#!/usr/bin/env bash
set -euo pipefail

# Build and scan the sanitized snapshot published to the PUBLIC mirror. It never commits,
# never pushes and never touches the repository working tree: `--build` exports HEAD with
# `git archive` into a directory OUTSIDE the repo, rewrites the site-specific values into
# placeholders there, and scans the result. The publication steps (temp index, commit-tree,
# push, fresh-clone rescan) stay outside this script, one tool call each.
#
# usage:
#   scripts/mirror-snapshot.sh --build /abs/path/to/snapshot   # export + sanitize + scan
#   scripts/mirror-snapshot.sh --scan [/abs/path]   # scan ONLY, no build: the snapshot left by
#                                 # the LAST --build (or TP4_MIRROR_DIR), unless a path is given
#   scripts/mirror-snapshot.sh --verify   # build into a throwaway directory, scan it, AND fail
#                                 # if the sanitizer had to rewrite ANY file: what is published
#                                 # must be clean at the source, not clean thanks to a rewrite
#   scripts/mirror-snapshot.sh --scan-static <dir>   # only the rules that need no cluster.env
#                                 # (IPv4, e-mail, MAC, IPv6/GID, key material, api tokens), on
#                                 # any directory, for CI on a plain checkout
#
# Exit: 0 clean, 1 at least one hit, 2 usage error.
#
# The mapping (node aliases, management IPs, fabric and link-local ranges, home directory,
# login name, tailnet names) is READ AT RUN TIME from cluster.env and from the login name
# derived from RELAY_DEST / `id -un`. It is never printed, never written to disk and never
# hard-coded here: only counters are reported, and a hit is shown as `path:line` plus a
# MASKED excerpt (first 3 characters), never as the private value itself.
#
# The node aliases of NODES are rewritten to `<ALIAS_RANKn>` everywhere, as plain
# case-insensitive substrings, so the `alias.`, `alias:`, `alias/` and `alias<N>` forms are
# all covered; the longest alias is substituted first so one alias cannot truncate another.
# Anything the mapping cannot derive (the workstation's own name, a city, a second login)
# goes into scripts/mirror-private-terms.txt — optional, gitignored, one extended regex per
# line, see scripts/mirror-private-terms.example. Those terms are NOT rewritten, only
# checked: one occurrence fails the scan, so the fix lands in the source file.
#
# Excluded from the snapshot (private, never tracked):
#   cluster.env, node/tp4-autostart.service, node/etc/common/99-tp4-nopasswd,
#   node/ssh-config, scripts/mirror-private-terms.txt, node/etc/<alias>/ (per-node
#   netplan), .claude/, __pycache__/, .ruff_cache/
#
# Sanitized and scanned set: every file the snapshot carries that `grep -I` recognises as
# text — no extension list, so `tp4ctl`, `SHA256SUMS`, `Dockerfile`, `*.patch`, `*.conf`,
# `*.json.full-tuned` and friends are covered too.
#
# The scan flags: IPv4 literals (an octet above 255 is what tells a four-part version string
# apart from an address) including the Tailscale CGNAT range (100.64/10), MAC addresses,
# IPv6/GID values, the login name, `*.ts.net` names outside the placeholder form,
# `.local`/`.lan`/`.home` host names, e-mail addresses, private-key headers and long base64
# blobs, `hf_`/`ghp_`/`sk-` tokens, any residual node alias or private term, the presence of
# any excluded path, a private file still listed by `git ls-files` (the snapshot is built
# from `git archive HEAD`, so a tracked private file would come straight back), and a
# `cluster.env.example` that does not parse as bash.
#
# The `.example` TEMPLATES are sanitized with the exact-value rewrites but NOT with the
# prefix-range ones: their dummy addresses are the documented plan and the RFC 5737
# documentation range, and they have to reach the reader intact.
# This header deliberately contains NO dotted-quad-shaped string: the snapshot carries this
# script too, so an example address written here would flag its own scan.
# Justified exceptions live in scripts/mirror-allow.txt (`<path-glob><TAB><regex>`), e.g. the
# upstream relay addresses inside the vendored NCCL patch. Nothing else is forgiven.
#
# macOS/BSD `sed` is the target: no `\b`, and `-i` needs an empty argument (GNU sed is
# detected and used with a plain `-i`).

REPO=$(cd "$(dirname "$0")/.." && pwd)

log()  { echo "[mirror] $*"; }
warn() { echo "[mirror] $*" >&2; }
die()  { echo "[mirror] ERROR: $*" >&2; exit 1; }

STATE_FILE="${TMPDIR:-/tmp}/tp4-mirror-snapshot.last"
MARKER=.tp4-mirror-snapshot
ALLOW_FILE="$REPO/scripts/mirror-allow.txt"

USAGE="usage: $0 --build <dir> | --scan [<dir>] | --verify | --scan-static <dir>"
MODE=""
DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) [ $# -ge 2 ] || { echo "--build needs a directory" >&2; exit 2; }; MODE=build; DIR=$2; shift 2 ;;
    --scan)  MODE=scan; shift; if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then DIR=$1; shift; fi ;;
    --verify) MODE=verify; shift ;;
    --scan-static) [ $# -ge 2 ] || { echo "--scan-static needs a directory" >&2; exit 2; }; MODE=scan-static; DIR=$2; shift 2 ;;
    -h|--help) sed -n '3,27p' "$0"; exit 0 ;;
    *) echo "$USAGE" >&2; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { echo "$USAGE" >&2; exit 2; }

# cluster.env holds the mapping every rewrite and every site-specific rule is derived from.
# --scan-static is the one mode that does without it, on purpose: it is what CI can run on a
# checkout that has no site configuration at all.
if [ "$MODE" != scan-static ]; then
  [ -f "$REPO/cluster.env" ] || { echo "cluster.env missing: copy cluster.env.example and fill it" >&2; exit 1; }
  # shellcheck source=../cluster.env
  . "$REPO/cluster.env"
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/tp4-mirror.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

if [ "$MODE" = verify ]; then
  # Throwaway snapshot inside our own temp dir: --verify leaves nothing behind and never
  # touches the directory --build recorded for --scan.
  DIR="$TMP/snapshot"
fi
if [ "$MODE" = scan ] && [ -z "$DIR" ]; then
  DIR=${TP4_MIRROR_DIR:-}
  [ -n "$DIR" ] || DIR=$(cat "$STATE_FILE" 2>/dev/null || true)
  [ -n "$DIR" ] || die "--scan without a directory needs a previous --build (or TP4_MIRROR_DIR)"
fi

if [ "$MODE" = scan-static ]; then
  # Read-only mode meant for CI: a relative path and the repository itself are both fine, so
  # the "absolute, outside the repository" rules of the build modes do not apply here.
  DIR=$(cd "$DIR" 2>/dev/null && pwd) || die "--scan-static: not a directory: $DIR"
else
  case "$DIR" in
    /*) ;;
    *)  die "the snapshot directory must be an absolute path (got: $DIR)" ;;
  esac
  DIR=${DIR%/}
  [ "$DIR" != "/" ] && [ "$DIR" != "$HOME" ] || die "refusing to work on $DIR"
  case "$DIR/" in
    "$REPO"/*) die "the snapshot directory must be OUTSIDE the repository ($REPO)" ;;
  esac
fi

esc() { printf '%s' "$1" | sed -e 's/[.[\*^$]/\\&/g'; }

# The mapping and the rewrite rules exist only for the modes that BUILD a snapshot and for
# the site-specific scan rules; --scan-static runs without cluster.env and without them.
if [ "$MODE" != scan-static ]; then
  # --- mapping, read from cluster.env; values are never printed ---------------------------
  read -r -a MG <<<"$MGMT_IPS"
  [ "${#MG[@]}" -gt 0 ] || die "MGMT_IPS is empty in cluster.env"
  read -r -a NODE_ALIASES <<<"${NODES:-}"
  [ "${#NODE_ALIASES[@]}" -gt 0 ] || die "NODES is empty in cluster.env"
  # login name: from RELAY_DEST (user@host) when present, otherwise the local account.
  USERNAME=${RELAY_DEST%%@*}
  case "$USERNAME" in ""|*[!A-Za-z0-9._-]*) USERNAME=$(id -un) ;; esac
  # subnet prefixes, derived (never literal): first three octets of the mgmt IPs, first two of
  # the fabric IPs, so an address of the same networks that is not in the map is caught too.
  MG_PREFIX=$(printf '%s' "${MG[0]}" | cut -d. -f1-3)
  FAB_PREFIX=$(printf '%s' "${FABRIC_TARGETS[0]%% *}" | cut -d. -f1-2)

  # Exact-value rewrites — applied to EVERY text file, templates included.
  SEDX_EXACT=()
  for i in "${!MG[@]}"; do SEDX_EXACT+=( -e "s#$(esc "${MG[$i]}")#<MGMT_IP_RANK$i>#g" ); done
  # The tailnet name goes first: it has to be collapsed while the host label in front of it is
  # still an alias and not yet the `<ALIAS_RANKn>` placeholder, which the pattern cannot match.
  SEDX_EXACT+=( -e 's#([a-z0-9-]+)\.[a-z0-9-]+\.ts\.net#\1.<TAILNET>.ts.net#g' )
  # Node aliases, LONGEST FIRST so an alias that is a prefix of another cannot truncate it, as
  # plain case-insensitive substrings so `alias.`, `alias:`, `alias/` and `alias<N>` are covered.
  for _i in $(for i in "${!NODE_ALIASES[@]}"; do printf '%s %s\n' "${#NODE_ALIASES[$i]}" "$i"; done \
              | sort -rn | cut -d' ' -f2); do
    SEDX_EXACT+=( -e "s#$(esc "${NODE_ALIASES[$_i]}")#<ALIAS_RANK$_i>#gI" )
  done
  # The PUBLIC repository URL is public information and must survive the login-name rewrite:
  # it is parked on a token that carries no private value and restored afterwards.
  # The PID is part of the token on purpose: the snapshot carries THIS script, and a token
  # written here as a fixed literal would be matched by the restore rule below and turned into
  # the real login inside this very file.
  GH_TOKEN="@@TP4-PUBLIC-GITHUB-$$@@"
  SEDX_EXACT+=( -e "s#github\.com/$(esc "$USERNAME")#$GH_TOKEN#g" )
  SEDX_EXACT+=( -e "s#/home/$(esc "$USERNAME")#/home/<USER>#g" -e "s#/Users/$(esc "$USERNAME")#/Users/<USER>#g" )
  SEDX_EXACT+=( -e "s#$(esc "$USERNAME")#<USER>#g" )
  SEDX_EXACT+=( -e "s#$GH_TOKEN#github.com/$USERNAME#g" )

  # Prefix-RANGE rewrites — everything EXCEPT the templates listed in TEMPLATES: they catch an
  # address of the same networks that is not in the map, but they would also eat the documented
  # dummy plan the templates are made of.
  SEDX_RANGE=()
  SEDX_RANGE+=( -e "s#$(esc "$MG_PREFIX")\.[0-9]{1,3}#<MGMT_IP>#g" )
  SEDX_RANGE+=( -e "s#$(esc "$FAB_PREFIX")\.[0-9]{1,3}\.[0-9]{1,3}#<FABRIC_IP>#g" )
  SEDX_RANGE+=( -e 's#169\.254\.[0-9]{1,3}\.[0-9]{1,3}#<LINK_LOCAL_IP>#g' )
fi

TEMPLATES=( cluster.env.example node/etc/40-cx7.yaml.example )

# GNU sed wants `-i`, BSD sed wants `-i ''`.
if sed --version >/dev/null 2>&1; then SED_INPLACE=( -E -i ); else SED_INPLACE=( -E -i '' ); fi

EXCLUDED=( cluster.env node/tp4-autostart.service node/etc/common/99-tp4-nopasswd
           node/ssh-config scripts/mirror-private-terms.txt )
EXCLUDED_DIRS=( .claude )
TERMS_FILE="$REPO/scripts/mirror-private-terms.txt"
# Private files that must not be TRACKED either (see the `git ls-files` check below).
PRIVATE_TRACKED_RE='(^|/)cluster\.env$|(^|/)99-tp4-nopasswd$|(^|/)tp4-autostart\.service$'
PRIVATE_TRACKED_RE="$PRIVATE_TRACKED_RE"'|^node/etc/[^/]+/40-cx7\.yaml$|^node/ssh-config$'
PRIVATE_TRACKED_RE="$PRIVATE_TRACKED_RE"'|(^|/)mirror-private-terms\.txt$'

# text_list <out-file>: NUL-separated list of every text file of the snapshot.
text_list() {
  : >"$1"
  find "$DIR" -type f -print0 \
    | while IFS= read -r -d '' f; do
        if LC_ALL=C grep -Iq . "$f" 2>/dev/null; then printf '%s\0' "$f" >>"$1"; fi
      done
}

# --- build (--build and --verify; --verify builds into the throwaway $TMP/snapshot) -------
if [ "$MODE" = build ] || [ "$MODE" = verify ]; then
  # Never delete a directory that this script did not create: it must either still carry the
  # marker of an interrupted run, or be the one recorded by the previous --build.
  if [ -e "$DIR" ]; then
    [ -e "$DIR/$MARKER" ] || [ "$(cat "$STATE_FILE" 2>/dev/null)" = "$DIR" ] \
      || die "$DIR exists and was not built by this script: remove it by hand"
    rm -rf "$DIR"
  fi
  mkdir -p "$DIR"
  : >"$DIR/$MARKER"
  ( cd "$REPO" && git archive HEAD ) | tar -x -C "$DIR"
  log "exported $(cd "$REPO" && git log --oneline -1 | cut -c1-72)"

  for f in "${EXCLUDED[@]}" "${EXCLUDED_DIRS[@]}"; do rm -rf "${DIR:?}/$f"; done
  for a in "${NODE_ALIASES[@]}"; do rm -rf "${DIR:?}/node/etc/$a"; done
  find "$DIR" \( -name __pycache__ -o -name .ruff_cache \) -type d -prune -exec rm -rf {} + 2>/dev/null || true
  log "excluded: ${EXCLUDED[*]} ${EXCLUDED_DIRS[*]} node/etc/<alias>/ __pycache__/ .ruff_cache/"
  log "mapping: ${#NODE_ALIASES[@]} aliases, ${#MG[@]} management IPs, 1 login name, mgmt/fabric/link-local prefixes (values not printed)"

  text_list "$TMP/text"
  n_text=$(tr -dc '\0' <"$TMP/text" | wc -c | tr -d ' ')
  [ "$n_text" -gt 0 ] || die "no text file in the snapshot: git archive produced nothing?"
  # The templates keep their documented dummy plan: exact rewrites yes, range rewrites no.
  : >"$TMP/text.plain"; : >"$TMP/text.tmpl"
  while IFS= read -r -d '' f; do
    rel=${f#"$DIR"/}; is_tmpl=0
    for t in "${TEMPLATES[@]}"; do
      if [ "$rel" = "$t" ]; then is_tmpl=1; break; fi
    done
    if [ "$is_tmpl" = 1 ]; then printf '%s\0' "$f" >>"$TMP/text.tmpl"
    else printf '%s\0' "$f" >>"$TMP/text.plain"; fi
  done <"$TMP/text"
  # No `xargs -r` on BSD: each list is checked for emptiness instead.
  [ ! -s "$TMP/text.plain" ] || xargs -0 sed "${SED_INPLACE[@]}" "${SEDX_EXACT[@]}" "${SEDX_RANGE[@]}" <"$TMP/text.plain"
  [ ! -s "$TMP/text.tmpl" ]  || xargs -0 sed "${SED_INPLACE[@]}" "${SEDX_EXACT[@]}" <"$TMP/text.tmpl"
  rm -f "$DIR/$MARKER"          # the snapshot is what gets published: nothing of ours stays in it
  log "sanitized $n_text text file(s) of $(find "$DIR" -type f | wc -l | tr -d ' ') in the snapshot"
  # Only a real --build leaves a snapshot behind for a later --scan; --verify's directory is
  # deleted on exit and must never become the recorded one.
  if [ "$MODE" = build ]; then printf '%s\n' "$DIR" >"$STATE_FILE"; fi
fi

# --- scan -------------------------------------------------------------------------------
[ -d "$DIR" ] || die "$DIR does not exist"
log "scan: $DIR"
text_list "$TMP/text"
[ "$(tr -dc '\0' <"$TMP/text" | wc -c | tr -d ' ')" -gt 0 ] || die "no text file under $DIR"

# allow-list: <path-glob><TAB><regex>, comments and blank lines ignored.
ALLOW_GLOB=(); ALLOW_RE=()
if [ -f "$ALLOW_FILE" ]; then
  while IFS=$'\t' read -r g r; do
    case "${g:-#}" in ''|'#'*) continue ;; esac
    [ -n "${r:-}" ] || continue
    ALLOW_GLOB+=("$g"); ALLOW_RE+=("$r")
  done <"$ALLOW_FILE"
fi
log "allow-list: ${#ALLOW_GLOB[@]} rule(s) from scripts/mirror-allow.txt"

HITS=0

# allow_filter <in> <out>: drops the hits an allow-list rule forgives and masks the rest.
# Input and output lines are `path:line:match`; the output keeps `path:line` plus the first
# three characters of the match, so a private value is never printed in full.
allow_filter() {
  local line rel rest match i allowed
  : >"$2"
  while IFS= read -r line; do
    rel=${line%%:*}; rest=${line#*:}; match=${rest#*:}
    allowed=0
    for i in "${!ALLOW_GLOB[@]}"; do
      # shellcheck disable=SC2254  # the glob must stay unquoted: it is a pattern
      case "$rel" in ${ALLOW_GLOB[$i]}) ;; *) continue ;; esac
      if printf '%s' "$match" | grep -qE "${ALLOW_RE[$i]}"; then allowed=1; break; fi
    done
    [ "$allowed" = 1 ] || printf '%s:%s  %.3s…\n' "$rel" "${rest%%:*}" "$match" >>"$2"
  done <"$1"
}

report() {   # report <label> <hits-file>
  local n; n=$(wc -l <"$2" | tr -d ' ')
  if [ "$n" = 0 ]; then
    printf '  OK   %-26s 0\n' "$1"
  else
    printf '  HIT  %-26s %s\n' "$1" "$n" >&2
    sed -e 's/^/       /' "$2" | head -20 >&2
    HITS=$((HITS + n))
  fi
}

# grep_text <grep-args...>: the whole text set, paths relative to the snapshot root.
grep_text() { xargs -0 grep -HInoE "$@" <"$TMP/text" 2>/dev/null | sed -e "s#^$DIR/##" || true; }

scan() {   # scan <label> <grep-args...>
  local label=$1; shift
  grep_text "$@" >"$TMP/raw"
  allow_filter "$TMP/raw" "$TMP/hits"
  report "$label" "$TMP/hits"
}

# 1. IPv4 literals: every octet must be ≤ 255 (that is what separates an address from a
#    four-part version string) and the well-known constants are allowed. Version strings whose
#    parts all fit in a byte still hit here and are forgiven per file in mirror-allow.txt.
grep_text '([0-9]{1,3}\.){3}[0-9]{1,3}' \
  | awk -F: '{ ip = $NF; n = split(ip, o, "."); ok = 1;
               for (i = 1; i <= n; i++) if (o[i] + 0 > 255) ok = 0;
               if (!ok) next;
               if (ip ~ /^0\.0\.0\.0$/ || ip ~ /^127\.0\.0\.1$/ || ip ~ /^255\.255\.255\.[0-9]+$/ \
                   || ip ~ /^224\.0\.0\.[0-9]+$/) next;
               print }' >"$TMP/raw" || true
allow_filter "$TMP/raw" "$TMP/hits"
report "ipv4 literal" "$TMP/hits"

# 2. the other categories. `<TAILNET>.ts.net` and `<MGMT_IP_RANKn>` are placeholders and match
#    none of these patterns (they carry no lowercase label / no digits). The three rules that
#    need the site mapping are skipped by --scan-static; an unresolved `<...>` placeholder is
#    matched by none of them, which is what makes the static set usable on a template.
if [ "$MODE" != scan-static ]; then
  scan "tailscale 100.64/10"   '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}'
  # The match is widened to the surrounding URL/path characters so that the one legitimate
  # occurrence — the PUBLIC repository URL — can be forgiven by a rule in mirror-allow.txt that
  # names no login. A sanitized snapshot has no other occurrence left.
  scan "login name"            "[A-Za-z0-9._:/-]*$(esc "$USERNAME")[A-Za-z0-9._:/-]*"
  scan "tailnet name"          '[a-z0-9-]+\.ts\.net'
fi
scan "mac address"           -i '([0-9a-f]{2}:){5}[0-9a-f]{2}'
scan "ipv6 / gid"            -i 'fe80:[0-9a-f:]+|([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|gid=[0-9a-f:]+'
scan "e-mail address"        '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
if [ "$MODE" != scan-static ]; then
  scan "local/lan/home name"   '(^|[^A-Za-z0-9._-])[a-z0-9-]+\.(local|lan|home)($|[^A-Za-z0-9._-])'
fi
scan "key material"          'BEGIN [A-Z ]*PRIVATE KEY|ssh-(rsa|ed25519) AAAA|ecdsa-sha2-[a-z0-9]+ AAAA|AAAA[A-Za-z0-9+/]{30,}'
scan "api token"             'hf_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,}'

# --scan-static stops here: everything below needs either the site mapping (residual aliases,
# private terms) or the repository the snapshot was built from (excluded paths, tracked private
# files). It is a CI gate on the SHAPE of the content, not a publication gate.
if [ "$MODE" = scan-static ]; then
  log "files scanned: $(find "$DIR" -type f | wc -l | tr -d ' ')"
  if [ "$HITS" = 0 ]; then
    log "static scan clean: 0 hits (IPv4, e-mail, MAC, IPv6/GID, key material, api tokens)"
    exit 0
  fi
  warn "static scan: $HITS hit(s) — a literal address, e-mail, MAC, GID, key or token is in the tree"
  exit 1
fi

# 3. excluded paths must not be in the snapshot at all.
: >"$TMP/excl"
for f in "${EXCLUDED[@]}" "${EXCLUDED_DIRS[@]}"; do
  [ ! -e "$DIR/$f" ] || printf '%s:0  present\n' "$f" >>"$TMP/excl"
done
# Any directory under node/etc/ that is not the shared material is a per-node netplan dir,
# whatever the aliases are called.
if [ -d "$DIR/node/etc" ]; then
  for d in "$DIR"/node/etc/*/; do
    [ -d "$d" ] || continue
    case "${d%/}" in */common|*/default) continue ;; esac
    printf '%s:0  present\n' "${d%/}" | sed -e "s#^$DIR/##" >>"$TMP/excl"
  done
fi
find "$DIR" \( -name __pycache__ -o -name .ruff_cache \) -print 2>/dev/null \
  | sed -e "s#^$DIR/##" -e 's/$/:0  present/' >>"$TMP/excl"
report "excluded path present" "$TMP/excl"

# 4. residual private terms: no node alias may have survived the rewrite, and none of the
#    extra site terms of scripts/mirror-private-terms.txt (optional, gitignored) may appear.
#    The terms are matched case-insensitively and are never printed: the report masks them.
: >"$TMP/terms"
for a in "${NODE_ALIASES[@]}"; do esc "$a" >>"$TMP/terms"; echo >>"$TMP/terms"; done
n_extra=0
if [ -f "$TERMS_FILE" ]; then
  while IFS= read -r t; do
    case "${t:-#}" in ''|'#'*) continue ;; esac
    printf '%s\n' "$t" >>"$TMP/terms"
    n_extra=$((n_extra + 1))
  done <"$TERMS_FILE"
fi
log "private terms: ${#NODE_ALIASES[@]} alias(es) + $n_extra from scripts/mirror-private-terms.txt"
scan "residual private term"  -i -f "$TMP/terms"

# 5. the private files must not be TRACKED: `--build` exports `git archive HEAD`, so a file
#    still in the index would be back in the next snapshot however carefully it is deleted
#    here. Checked against the repository, not against the snapshot.
: >"$TMP/tracked"
( cd "$REPO" && git ls-files ) | grep -E "$PRIVATE_TRACKED_RE" | sed -e 's/$/:0  tracked/' >>"$TMP/tracked" || true
report "private file tracked" "$TMP/tracked"

# 6. cluster.env.example is the template the whole runbook starts from and is meant to be
#    sourceable as-is (placeholders quoted): a snapshot that carries a template which does not
#    parse as bash is not publishable.
: >"$TMP/synt"
if [ -f "$DIR/cluster.env.example" ]; then
  if ! bash -n "$DIR/cluster.env.example" 2>"$TMP/synt.err"; then
    sed -e "s#$DIR/##g" -e 's/^/cluster.env.example:0  /' "$TMP/synt.err" | head -5 >>"$TMP/synt"
  fi
else
  printf 'cluster.env.example:0  missing from the snapshot\n' >>"$TMP/synt"
fi
report "cluster.env.example syntax" "$TMP/synt"

# 7. --verify only: the sanitizer must have had NOTHING to rewrite. What is published is
#    publishable because HEAD is already free of site values, not because the builder rewrote
#    them on the way out — every file whose sanitized copy differs from `git archive HEAD` is
#    listed here and fails the run, so the fix lands in the source file.
if [ "$MODE" = verify ]; then
  : >"$TMP/rewritten"
  mkdir -p "$TMP/pristine"
  ( cd "$REPO" && git archive HEAD ) | tar -x -C "$TMP/pristine"
  while IFS= read -r -d '' f; do
    rel=${f#"$DIR"/}
    [ -f "$TMP/pristine/$rel" ] || continue
    cmp -s "$f" "$TMP/pristine/$rel" || printf '%s:0  rewritten by the sanitizer\n' "$rel" >>"$TMP/rewritten"
  done <"$TMP/text"
  report "sanitizer rewrote file" "$TMP/rewritten"
fi

log "files in snapshot: $(find "$DIR" -type f | wc -l | tr -d ' ')"
if [ "$HITS" = 0 ] && [ "$MODE" = verify ]; then
  log "verify clean: 0 hits, 0 file rewritten — HEAD is publishable as it stands"
elif [ "$HITS" = 0 ]; then
  log "scan clean: 0 hits — the snapshot may be committed on the public mirror (separate call)"
else
  warn "scan STOP: $HITS hit(s) — do NOT publish; fix the sources, the sanitizer or (if the"
  warn "           value is genuinely public) add a justified rule to scripts/mirror-allow.txt"
fi
[ "$HITS" = 0 ]
