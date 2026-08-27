# Milestone 14: Circuit Breaker & Chaos Resilience Engineering

This document details the Finite State Machine (FSM) architecture, fast-fail protection latency, and empirical chaos resilience evaluations of the Circuit Breaker in the Cinch Gateway.

---

## 1. Cascading GPU Failure Mitigation

When upstream LLM inference instances experience CUDA Out-Of-Memory (OOM) panics, deadlocks, or Kubernetes worker restarts:
- Unmitigated gateways continue buffering client requests until connection timeouts trigger ($30.0\text{s}$ per request).
- Concurrent client requests saturate the gateway event loop, filling queue memory and crashing the routing tier.
- **The Solution**: An automated **Three-State Circuit Breaker** that detects consecutive upstream failures, immediately fast-fails new ingress traffic to protect gateway resources, and automatically probes backend health via single-flight canary requests upon cooldown.

---

## 2. Circuit Breaker State Machine Architecture (`gateway/circuit_breaker.py`)

```
             +-----------------------+
             |        CLOSED         |<-----------------------+
             |   (Normal Routing)    |                        |
             +-----------------------+                        |
                         |                                    |
          Consecutive Failures >= 3                           |
                         |                                    |
                         v                                    |
             +-----------------------+                        |
             |         OPEN          |                 Canary Probe
             | (Fast-Fail with 503)  |                   Succeeds
             +-----------------------+                        |
                         |                                    |
             Cooldown (10s) Expired                           |
                         |                                    |
                         v                                    |
             +-----------------------+                        |
             |       HALF-OPEN       |------------------------+
             | (1 Canary In-Flight)  |
             +-----------------------+
                         |
                   Canary Fails
                         |
                         v
                    (Re-trip to OPEN)
```

---

## 3. Empirical Chaos Benchmark Results

Data source: `benchmarks/results/chaos_resilience.json`  
Target: In-Cluster Gateway on `http://localhost:8081`  
Failure Injection: Burst of upstream error probes tripping gateway replicas, followed by concurrent ingress traffic and automated canary recovery.

### Resilience & Latency Breakdown

| Evaluation Phase | Injected Condition | Cluster Breaker State | Response Status | Observed Latency | System Impact |
|---|---|---|---|---|---|
| **Phase 1** | Healthy Baseline | `CLOSED` | `200 OK` | $21.5\text{ ms}$ | Full inference serving |
| **Phase 2** | Fault Burst (Probes 1–8) | `CLOSED` $\to$ `OPEN` | `404` / `503` | $46.8\text{ ms}$ | Replicas trip to `OPEN` on 3rd failure |
| **Phase 3** | Fast-Fail Protection (10 reqs) | `OPEN` | `503 Unavailable` | **$45.56\text{ ms}$** | **$658\times$ faster rejection** vs $30\text{s}$ hang |
| **Phase 4** | Canary Probing ($T=10.5\text{s}$) | `HALF_OPEN` $\to$ `CLOSED` | `200 OK` | $19.8\text{ ms}$ | Full recovery restored (**MTTR: $10.85\text{s}$**) |

```
Degraded Worker Latency Protection
30,000ms |  [ Unmitigated Connection Timeout: 30,000ms ]
         |  |========================================================================|
         |
    45ms |  [ Circuit Breaker Fast-Fail: 45.56ms ]
         |  |=|
     0ms +---------------------------------------------------------------------------+
```

### Key Performance Findings
- **Fast-Fail Rejection Latency**: **$45.56\text{ ms}$** total HTTP round-trip (compared to a $30,000\text{ ms}$ unmitigated TCP connection timeout).
- **Protection Factor**: **$658\times$ faster rejection**, preventing gateway queue saturation and memory exhaustion during GPU worker outages.
- **Mean Time To Recover (MTTR)**: **$10.85\text{ s}$** automated self-healing upon upstream recovery with zero operator intervention.
