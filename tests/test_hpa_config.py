"""Unit test suite for Kubernetes HPA configuration and telemetry parsing."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch
import yaml
from scripts.load_test_hpa import query_k8s_hpa_status


def test_hpa_manifest_structure() -> None:
    """Verify gateway HPA manifest schema, scaling triggers, and bounds."""
    hpa_file = pathlib.Path(__file__).parent.parent / "k8s" / "gateway-hpa.yaml"
    with open(hpa_file, "r", encoding="utf-8") as f:
        hpa = yaml.safe_load(f)

    assert hpa["kind"] == "HorizontalPodAutoscaler"
    assert hpa["metadata"]["name"] == "cinch-gateway-hpa"
    assert hpa["metadata"]["namespace"] == "cinch"

    spec = hpa["spec"]
    assert spec["scaleTargetRef"]["kind"] == "Deployment"
    assert spec["scaleTargetRef"]["name"] == "cinch-gateway"
    assert spec["minReplicas"] == 2
    assert spec["maxReplicas"] == 6

    # Metric check
    metric = spec["metrics"][0]
    assert metric["type"] == "Resource"
    assert metric["resource"]["name"] == "cpu"
    assert metric["resource"]["target"]["averageUtilization"] == 50

    # Behavior check
    behavior = spec["behavior"]
    assert behavior["scaleUp"]["stabilizationWindowSeconds"] == 0
    assert behavior["scaleDown"]["stabilizationWindowSeconds"] == 60


def test_kustomization_includes_hpa() -> None:
    """Verify kustomization bundle includes gateway-hpa.yaml."""
    kust_file = pathlib.Path(__file__).parent.parent / "k8s" / "kustomization.yaml"
    with open(kust_file, "r", encoding="utf-8") as f:
        kust = yaml.safe_load(f)

    assert "gateway-hpa.yaml" in kust["resources"]


def test_query_k8s_hpa_status_mock() -> None:
    """Verify HPA and deployment telemetry parsing logic."""
    mock_deployment = json.dumps({"status": {"replicas": 4, "readyReplicas": 4}})
    mock_hpa = json.dumps({
        "status": {
            "desiredReplicas": 4,
            "currentMetrics": [
                {"type": "Resource", "resource": {"name": "cpu", "current": {"averageUtilization": 78}}}
            ],
        }
    })
    mock_top = "cinch-gateway-abc 150m 45Mi\ncinch-gateway-def 140m 42Mi\n"

    def side_effect(cmd: list[str], *args: object, **kwargs: object) -> str:
        if "deployment" in cmd:
            return mock_deployment
        elif "hpa" in cmd:
            return mock_hpa
        elif "top" in cmd:
            return mock_top
        return "{}"

    with patch("subprocess.check_output", side_effect=side_effect):
        state = query_k8s_hpa_status()
        assert state["replicas"] == 4
        assert state["ready_replicas"] == 4
        assert state["desired_replicas"] == 4
        assert state["current_cpu_percent"] == 78
        assert len(state["pod_metrics"]) == 2
        assert state["pod_metrics"][0]["cpu"] == "150m"
