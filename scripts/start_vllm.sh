#!/usr/bin/env bash
# Operational control script for local vLLM serving container.
# Usage: ./scripts/start_vllm.sh [start|stop|restart|status|logs|wait] [timeout_seconds]

set -euo pipefail

ACTION="${1:-start}"
TIMEOUT_SECONDS="${2:-180}"
COMPOSE_FILE="docker-compose.vllm.yml"
HEALTH_URL="http://localhost:8000/health"

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "[ERROR] Docker engine is not reachable. Ensure Docker daemon is running with NVIDIA GPU support." >&2
        exit 1
    fi
}

start_serving() {
    check_docker
    echo "==> Starting vLLM serving container ($COMPOSE_FILE)..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "[OK] Container started in background."
    echo "==> Run './scripts/start_vllm.sh wait' to monitor readiness."
}

stop_serving() {
    check_docker
    echo "==> Stopping vLLM serving container..."
    docker compose -f "$COMPOSE_FILE" down
    echo "[OK] Container stopped."
}

show_status() {
    check_docker
    docker compose -f "$COMPOSE_FILE" ps
}

show_logs() {
    check_docker
    docker compose -f "$COMPOSE_FILE" logs -f --tail=100
}

wait_readiness() {
    echo "==> Waiting for vLLM server readiness at $HEALTH_URL (timeout: ${TIMEOUT_SECONDS}s)..."
    local start_time
    start_time=$(date +%s)

    while true; do
        local now
        now=$(date +%s)
        local elapsed=$((now - start_time))

        if [ "$elapsed" -ge "$TIMEOUT_SECONDS" ]; then
            echo "[ERROR] Timed out waiting for vLLM server readiness after ${TIMEOUT_SECONDS}s." >&2
            exit 1
        fi

        if curl -s -f "$HEALTH_URL" >/dev/null 2>&1; then
            echo "[OK] vLLM server is healthy and ready to serve requests! (took ${elapsed}s)"
            return 0
        fi

        echo "  ... initializing / loading weights (${elapsed}s elapsed)"
        sleep 5
    done
}

case "$ACTION" in
    start)   start_serving ;;
    stop)    stop_serving ;;
    restart) stop_serving; start_serving ;;
    status)  show_status ;;
    logs)    show_logs ;;
    wait)    wait_readiness ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|wait} [timeout_seconds]" >&2
        exit 1
        ;;
esac
