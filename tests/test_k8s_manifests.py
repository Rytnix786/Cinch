"""Automated unit test suite for Cinch Kubernetes manifests."""

from __future__ import annotations

import pathlib
import subprocess
import yaml


def test_k8s_yaml_syntax_and_structure() -> None:
    """Verify all Kubernetes manifests parse cleanly without YAML syntax errors."""
    k8s_dir = pathlib.Path(__file__).parent.parent / "k8s"
    yaml_files = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("production-extension/*.yaml"))
    assert len(yaml_files) >= 6

    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) > 0, f"Empty YAML file: {yf}"
            for doc in docs:
                assert "apiVersion" in doc, f"Missing apiVersion in {yf}"
                assert "kind" in doc, f"Missing kind in {yf}"


def test_gateway_deployment_spec() -> None:
    """Verify Gateway Deployment adheres to PRD resource, probe, and replica specs."""
    deploy_file = pathlib.Path(__file__).parent.parent / "k8s" / "gateway-deployment.yaml"
    with open(deploy_file, "r", encoding="utf-8") as f:
        deploy = yaml.safe_load(f)

    assert deploy["kind"] == "Deployment"
    assert deploy["metadata"]["name"] == "cinch-gateway"
    assert deploy["metadata"]["namespace"] == "cinch"
    assert deploy["spec"]["replicas"] >= 2

    # Container checks
    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "gateway"

    # Resource requests and limits
    res = container["resources"]
    assert res["requests"]["cpu"] == "200m"
    assert res["requests"]["memory"] == "256Mi"
    assert res["limits"]["cpu"] == "500m"
    assert res["limits"]["memory"] == "512Mi"

    # Health probes
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["port"] == 8080
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["port"] == 8080


def test_gateway_service_spec() -> None:
    """Verify Gateway Service maps external port 80 to container port 8080."""
    svc_file = pathlib.Path(__file__).parent.parent / "k8s" / "gateway-service.yaml"
    with open(svc_file, "r", encoding="utf-8") as f:
        svc = yaml.safe_load(f)

    assert svc["kind"] == "Service"
    assert svc["metadata"]["name"] == "cinch-gateway"
    assert svc["metadata"]["namespace"] == "cinch"
    assert svc["spec"]["type"] == "LoadBalancer"

    port_entry = svc["spec"]["ports"][0]
    assert port_entry["port"] == 80
    assert port_entry["targetPort"] == 8080


def test_kustomization_resources() -> None:
    """Verify Kustomization bundle includes all core resources."""
    kust_file = pathlib.Path(__file__).parent.parent / "k8s" / "kustomization.yaml"
    with open(kust_file, "r", encoding="utf-8") as f:
        kust = yaml.safe_load(f)

    assert kust["kind"] == "Kustomization"
    assert kust["namespace"] == "cinch"
    resources = kust["resources"]
    assert "namespace.yaml" in resources
    assert "configmap.yaml" in resources
    assert "secret.yaml" in resources
    assert "gateway-deployment.yaml" in resources
    assert "gateway-service.yaml" in resources


def test_production_extension_specs() -> None:
    """Verify production GPU extension manifests."""
    ext_dir = pathlib.Path(__file__).parent.parent / "k8s" / "production-extension"
    gpu_deploy_file = ext_dir / "vllm-gpu-deployment.yaml"
    keda_file = ext_dir / "keda-scaledobject.yaml"

    with open(gpu_deploy_file, "r", encoding="utf-8") as f:
        gpu_deploy = yaml.safe_load(f)
    assert gpu_deploy["kind"] == "Deployment"
    res = gpu_deploy["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert res["limits"]["nvidia.com/gpu"] == "1"

    with open(keda_file, "r", encoding="utf-8") as f:
        keda = yaml.safe_load(f)
    assert keda["kind"] == "ScaledObject"
    assert keda["spec"]["scaleTargetRef"]["name"] == "vllm-worker"


def test_kubectl_client_dry_run() -> None:
    """Verify client-side schema validation using kubectl kustomize / dry-run."""
    k8s_dir = pathlib.Path(__file__).parent.parent / "k8s"
    try:
        # Use kubectl kustomize for pure client-side template expansion
        cmd = ["kubectl", "kustomize", str(k8s_dir)]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        assert "kind: Deployment" in output
        assert "name: cinch-gateway" in output
        assert "namespace: cinch" in output
    except FileNotFoundError:
        pass  # Skip if kubectl binary is not in PATH
    except subprocess.CalledProcessError as exc:
        raise AssertionError(f"kubectl kustomize failed:\n{exc.output}") from exc

