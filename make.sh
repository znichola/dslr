#!/bin/bash

fetch() {
    wget https://cdn.intra.42.fr/document/document/36253/datasets.tgz
    tar -xf datasets.tgz
    rm datasets.tgz
}

venv() {
    uv sync
    echo "ℹ️ To activate the venv run : source .venv/bin/activate"
    echo "ℹ️ To activate the venv run : .\.venv\Scripts\activate"
}

jupyter() {
    source .venv/bin/activate \
        && python3 -m jupyter notebook
}

usage() {
    cmds=$(declare -F | awk '{print $3}' | paste -s -d'|' -)
    echo "Usage: ./make.sh [$cmds]"
}


# ENTRY POINT

cmd="$1"

if [[ -z "$cmd" || "$cmd" == "help" ]]; then
    usage
    exit 0
fi

# Check if function exists, then call
if declare -F "$cmd" > /dev/null; then
    "$cmd"
else
    echo "Unknown command: $cmd"
    echo
    usage
    exit 1
fi
