# Milestone 7: Kubernetes Orchestration and Local k3d Deployment

This document details the Kubernetes orchestration architecture, multi-node cluster topology, and local deployment validation for the Cinch LLM serving platform.

---

## 1. Architecture and Cluster Topology

The Cinch platform uses a decoupled, two-tier architecture:
1. **Stateless Gateway Tier**: Managed inside Kubernetes across worker nodes.
2. **GPU Inference Backend Tier**: Single-container vLLM server with direct hardware passthrough.

```
+-----------------------------------------------------------------------------------------+
|                                    k3d Multi-Node Cluster                               |
|                                                                                         |
|  +-----------------------------+           +-----------------------------+              |
|  | Node: cinch-cluster-agent-0 |           | Node: cinch-cluster-agent-1 |              |
|  |  [Pod: cinch-gateway-1]     |           |  [Pod: cinch-gateway-2]     |              |
|  |   - CPU: 200m / 500m        |           |   - CPU: 200m / 500m        |              |
|  |   - Mem: 256Mi / 512Mi      |           |   - Mem: 256Mi / 512Mi      |              |
|  |   - Health: /health         |           |   - Health: /health         |              |
|  +-----------------------------+           +-----------------------------+              |
|                 ^                                         ^                             |
|                 +-------------------+---------------------+                             |
|                                     |                                                   |
|                         [Service: cinch-gateway]                                        |
|                         [Ingress: Traefik :8081]                                        |
+-------------------------------------|---------------------------------------------------+
                                      | (Internal Cluster DNS / host.k3d.internal:8000)
                                      v
                      +----------------------------------+
                      | Host GPU Docker Daemon           |
                      |  [Container: cinch-vllm]         |
                      |   - RTX 3060 Ti (8GB VRAM)       |
                      |   - Qwen2.5-7B-AWQ (Marlin)      |
                      +----------------------------------+
```

---

## 2. Kubernetes Manifest Specifications

All core manifests reside under `k8s/` and bundle via `k8s/kustomization.yaml`.

### Resource Manifests
- **Namespace (`k8s/namespace.yaml`)**: `cinch` isolation domain.
- **ConfigMap (`k8s/configmap.yaml`)**: Defines `VLLM_BASE_URL` (`http://host.k3d.internal:8000`), `RATE_LIMIT_RPM` (`120`), and `REQUEST_TIMEOUT_SECONDS` (`120.0`).
- **Secret (`k8s/secret.yaml`)**: Stores `GATEWAY_API_KEY` (`cinch-prod-key`).
- **Deployment (`k8s/gateway-deployment.yaml`)**:
  - Replicas: 2 (distributed across agent nodes).
  - Rolling update strategy: `maxSurge: 1`, `maxUnavailable: 0` (zero-downtime deployments).
  - Resource requests: `cpu: 200m`, `memory: 256Mi`.
  - Resource limits: `cpu: 500m`, `memory: 512Mi`.
  - Liveness probe: HTTP `GET /health` on port 8080 (initial delay: 5s, period: 10s).
  - Readiness probe: HTTP `GET /health` on port 8080 (initial delay: 3s, period: 5s).
- **Service & Ingress (`k8s/gateway-service.yaml`, `k8s/gateway-ingress.yaml`)**: Exposes the gateway via Traefik Ingress on cluster host port 8081.

---

## 3. Production Extension Path (PRD §6)

As defined in PRD.md §6, single-GPU developer hardware limits real multi-pod GPU placement to one device. Production multi-node cloud clusters scale GPU workers through the dedicated manifests in `k8s/production-extension/`:
1. **GPU Worker Deployment (`k8s/production-extension/vllm-gpu-deployment.yaml`)**: Defines NVIDIA GPU node selectors, `nvidia.com/gpu: 1` limits, `/dev/shm` shared memory mounts (4Gi), and HuggingFace cache persistent volume claims.
2. **KEDA Autoscaler (`k8s/production-extension/keda-scaledobject.yaml`)**: Defines queue-depth autoscaling triggers on `vllm:num_requests_waiting > 4` with custom scale-up (100% per 15s) and scale-down stabilization windows (300s).

---

## 4. Local Deployment Execution

### Cluster Deployment Command
```powershell
powershell -ExecutionPolicy Bypass -File scripts/k8s_deploy.ps1
```

### Live Status Verification
```
NAMESPACE     NAME                                 READY   STATUS    IP          NODE
cinch         pod/cinch-gateway-55fbb4f997-2fdsq   1/1     Running   10.42.2.6   k3d-cinch-cluster-server-0
cinch         pod/cinch-gateway-55fbb4f997-9gwtc   1/1     Running   10.42.1.4   k3d-cinch-cluster-agent-1
```

### Verification Client Execution
```powershell
python scripts/test_gateway_live.py --gateway-url http://localhost:8081 --api-key cinch-prod-key
```
Result: `[SUCCESS] Milestone 3 live verification passed: FastAPI Gateway operational.`
