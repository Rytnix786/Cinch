# Milestone 8: Gateway Horizontal Pod Autoscaler (HPA) & Load Testing

This document details the configuration, load-testing execution, and empirical scaling behavior of the Kubernetes HorizontalPodAutoscaler (HPA) for the stateless FastAPI Gateway in the local multi-node k3d cluster.

---

## 1. HPA Configuration Specifications

Target: `Deployment/cinch-gateway` (Namespace: `cinch`)  
Manifest: `k8s/gateway-hpa.yaml`  

- **Scaling Bounds**:
  - `minReplicas`: 2
  - `maxReplicas`: 6
- **Scaling Metric**:
  - `type`: Resource (CPU)
  - `target`: `averageUtilization: 50%` (relative to container CPU request of `200m`, triggering when average pod CPU exceeds `100m`).
- **Autoscaling Behavior Policies**:
  - **Scale-Up**: `stabilizationWindowSeconds: 0`, policy: $+100\%$ or $+2$ pods per 15s (immediate scaling under load spikes).
  - **Scale-Down**: `stabilizationWindowSeconds: 60`, policy: $-50\%$ per 30s (prevents thrashing/flapping during brief request pauses).

---

## 2. Empirical Load Test & Scaling Timeline

Data source: `benchmarks/results/hpa_scaling.json`  
Load Profile: 24 concurrent worker threads generating continuous traffic  
Total Requests Processed: **26,296 requests**  

```
Replicas & CPU Utilization Over Time
  6 Replicas |                             +---------------------------+
             |                            / (223% CPU Peak)            |
  4 Replicas |                   +-------+                             |
             |                  / (88% CPU)                            |
  2 Replicas | +---------------+                                       +--------------+
  (Min)      | (1-4% CPU)                                              (1% CPU Cooldown)
             +-----------------+---------+-----------------------------+--------------+
             0s               50s       60s                           95s            135s
             [----- BASELINE -----] [----- LOAD INJECTION -----] [----- COOLDOWN -----]
```

### Milestone Timeline Breakdown

| Time Window | Operational Phase | Replica Count | Ready Replicas | CPU Utilization (%) | Description |
|---|---|---|---|---|---|
| **0.0s – 48.0s** | Baseline / Initial Load | 2 | 2 | 1% – 4% | Initial request ingress warming up container socket pools. |
| **51.1s** | Scale-Up Event 1 | 4 | 2 | **88%** (Threshold: 50%) | HPA detects CPU breach (88% > 50%), emitting scale-up event from 2 to 4 desired replicas. |
| **57.7s** | Ready Convergence 1 | 4 | 4 | **88%** | Newly scheduled gateway pods pass readiness probes across cluster agent nodes. |
| **66.5s** | Scale-Up Event 2 | 6 | 4 | **223%** | Sustained load drives CPU to 223%, triggering HPA to scale to maxReplicas (6). |
| **73.1s** | Max Scale Convergence | 6 | 6 | **223%** | All 6 replicas fully ready and actively serving ingress traffic. |
| **95.1s** | Traffic Cessation / Cooldown | 6 | 6 | **19%** | Load generation stops. CPU drops below 50% target to 19%. |
| **110.5s – 135s** | Scale-Down Stabilization | 6 | 6 | **1%** | CPU drops to idle (1%). 60-second stabilization window elapses, initiating gradual scale-down back to 2 replicas. |

---

## 3. Multi-Node Pod Distribution

During peak scaling ($N=6$), Kubernetes scheduler distributed pods across all cluster nodes:
- `k3d-cinch-cluster-server-0`: 2 pods
- `k3d-cinch-cluster-agent-0`: 2 pods
- `k3d-cinch-cluster-agent-1`: 2 pods

All pods passed in-cluster readiness probes (`/health`) and received load-balanced traffic through the Traefik Ingress controller on port 8081 without dropped requests or connection errors.

---

## 4. Honest Scope Boundary (Architectural Design Scope)

- **Tested in Cluster**: Real Horizontal Pod Autoscaling of the stateless FastAPI gateway layer across a multi-node local k3d cluster using CPU utilization triggers under load.
- **Production Extension Path**: Cloud GPU worker scaling (`k8s/production-extension/`) using KEDA and queue depth metrics (`vllm:num_requests_waiting`) is formally architected and documented as the cloud path, not claimed as executed on single-GPU developer hardware.
