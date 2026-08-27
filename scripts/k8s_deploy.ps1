# Cinch Kubernetes Deployment Script for Local k3d Cluster (PowerShell)
param(
    [string]$ClusterName = "cinch-cluster",
    [int]$Agents = 2,
    [int]$HostPort = 8081
)

$ErrorActionPreference = "Stop"

Write-Host "=== Deploying Cinch Platform to k3d Cluster ===" -ForegroundColor Cyan

# 1. Check if cluster already exists
$clusterList = k3d cluster list -o json | ConvertFrom-Json
$existing = $clusterList | Where-Object { $_.name -eq $ClusterName }

if (-not $existing) {
    Write-Host "Creating k3d cluster '$ClusterName' (1 server, $Agents agents, port $HostPort:80)..." -ForegroundColor Yellow
    k3d cluster create $ClusterName --servers 1 --agents $Agents --port "${HostPort}:80@loadbalancer"
} else {
    Write-Host "Using existing k3d cluster '$ClusterName'." -ForegroundColor Green
}

# 2. Build and import Gateway Docker Image
Write-Host "`nBuilding local gateway image 'cinch-gateway:latest'..." -ForegroundColor Yellow
docker build -t cinch-gateway:latest -f docker/Dockerfile.gateway .

Write-Host "Importing 'cinch-gateway:latest' into k3d cluster '$ClusterName'..." -ForegroundColor Yellow
k3d image import cinch-gateway:latest -c $ClusterName

# 3. Apply Kubernetes Manifests
Write-Host "`nApplying Kubernetes manifests from k8s/..." -ForegroundColor Yellow
kubectl apply -k k8s/

# 4. Wait for rollout
Write-Host "`nWaiting for Deployment/cinch-gateway rollout..." -ForegroundColor Yellow
kubectl rollout status deployment/cinch-gateway -n cinch --timeout=120s

# 5. Display Status
Write-Host "`n=== Deployment Status (Namespace: cinch) ===" -ForegroundColor Green
kubectl get pods,svc,endpoints -n cinch -o wide

Write-Host "`n[SUCCESS] Cinch Gateway deployed successfully on cluster." -ForegroundColor Green
Write-Host "Access Gateway at: http://localhost:$HostPort" -ForegroundColor Cyan
