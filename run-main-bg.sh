#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="/tmp/kiro-gateway.pid"
LOG_FILE="/tmp/kiro-gateway.log"

cd "${SCRIPT_DIR}"

if [[ -f "${PID_FILE}" ]]; then
    EXISTING_PID="$(cat "${PID_FILE}")"
    if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "main.py is already running with PID ${EXISTING_PID}" >&2
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

nohup python main.py "$@" >"${LOG_FILE}" 2>&1 &
PID=$!

echo "${PID}" >"${PID_FILE}"
echo "${PID}"
