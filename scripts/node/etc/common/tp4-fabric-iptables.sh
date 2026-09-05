#!/bin/sh
# Allow TP4 RoCE fabric traffic through Docker's user hook chain.
set -e

: "${TP4_FABRIC_IFACES:?set TP4_FABRIC_IFACES via /etc/default/tp4-fabric-iptables}"

iptables -L DOCKER-USER -n >/dev/null 2>&1 || exit 0

for IF in $TP4_FABRIC_IFACES; do
    iptables -C DOCKER-USER -i "$IF" -j ACCEPT 2>/dev/null || \
        iptables -I DOCKER-USER -i "$IF" -j ACCEPT
    iptables -C DOCKER-USER -o "$IF" -j ACCEPT 2>/dev/null || \
        iptables -I DOCKER-USER -o "$IF" -j ACCEPT
done

exit 0
