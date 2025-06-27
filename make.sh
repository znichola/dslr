#!/bin/bash

fetch() {
    wget https://cdn.intra.42.fr/document/document/34001/datasets.tgz
    tar -xf datasets.tgz
    rm datasets.tgz
}

venv() {

    IN_SCHOOL="$(echo $SESSION_MANAGER | grep 42 | wc --lines)"

    VENV="python3 -m venv"

    if [[ IN_SCHOOL == 1 ]]; then
        VENV="virtualenv .venv"
    fi 

    $VENV .venv \
        && source .venv/bin/activate \
        && pip install --upgrade pip \
        && pip install -r requirements.txt

    echo "ℹ️ To activate the venv: run : source .venv/bin/activate"
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
