#!/bin/sh
# Allow TP4 RoCE fabric traffic through Docker's user hook chain.
set -e

IFACES="enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1"

iptables -L DOCKER-USER -n >/dev/null 2>&1 || exit 0

for IF in $IFACES; do
    iptables -C DOCKER-USER -i "$IF" -j ACCEPT 2>/dev/null || \
        iptables -I DOCKER-USER -i "$IF" -j ACCEPT
    iptables -C DOCKER-USER -o "$IF" -j ACCEPT 2>/dev/null || \
        iptables -I DOCKER-USER -o "$IF" -j ACCEPT
done

exit 0
