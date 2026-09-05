FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
RUN for i in 1 2 3 4 5; do apt-get update && break || sleep 10; done && \
    apt-get install -y --no-install-recommends build-essential libibverbs-dev python3 git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python3 --version && git --version && test -f /usr/include/infiniband/verbs.h
RUN git config --global --add safe.directory /nccl
