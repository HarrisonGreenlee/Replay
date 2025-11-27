#!/usr/bin/env bash

set -euo pipefail

mkdir -p networks

for POP in 500 1000 1500 2000 2500 3000 3500; do
    OUT_FILE="networks/dense_${POP}.txt"
    echo "Generating ${OUT_FILE} with population ${POP}..."

    ./contact_network \
        "2000-01-01T00:00:00" \
        100 \
        3600 \
        "${POP}" \
        "${POP}" \
        1 \
        1 \
        "${OUT_FILE}" \
        12345
done

echo "All networks generated in 'networks/'."

