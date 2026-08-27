#!/usr/bin/env bash
# Cinch Kubernetes Deployment Script for Local k3d Cluster (Bash)
set -euo pipefail

CLUSTER_NAME="${1:-cinch-cluster}"
AGENTS="${2:-2}"
HOST_PORT="${3:-8081}"

echo "=== Deploying Cinch Platform to k3d Cluster ==="

# 1. Create k3d cluster if not exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Creating k3d cluster '$CLUSTER_NAME' (1 server, $AGENTS agents, port $HOST_PORT:80)..."
    k3d cluster create "$CLUSTER_NAME" --servers 1 --agents "$AGENTS" --port "${HOST_PORT}:80@loadbalancer"
else
    echo "Using existing k3d cluster '$CLUSTER_NAME'."
fi

# 2. Build and import image
echo "Building local gateway image 'cinch-gateway:latest'..."
docker build -t cinch-gateway:latest -f docker/Dockerfile.gateway .

echo "Importing image into k3d..."
k3d image import cinch-gateway:latest -c "$CLUSTER_NAME"

# 3. Apply manifests
echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/

# 4. Wait for rollout
echo "Waiting for rollout..."
kubectl rollout status deployment/cinch-gateway -n cinch --timeout=120s

# 5. Display status
echo "=== Deployment Status ==="
kubectl get pods,svc,endpoints -n cinch -o wide
echo "Access Gateway at http://localhost:$HOST_PORT"
