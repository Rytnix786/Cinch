<#
.SYNOPSIS
    Operational control script for local vLLM serving container.
.DESCRIPTION
    Starts, stops, monitors, or waits for readiness of the vLLM Docker container.
.PARAMETER Action
    Action to perform: 'start', 'stop', 'restart', 'status', 'logs', or 'wait' (default: 'start').
.PARAMETER TimeoutSeconds
    Timeout in seconds when waiting for container readiness (default: 180).
.EXAMPLE
    .\scripts\start_vllm.ps1 -Action start
    .\scripts\start_vllm.ps1 -Action wait
    .\scripts\start_vllm.ps1 -Action logs
#>

param (
    [ValidateSet("start", "stop", "restart", "status", "logs", "wait")]
    [string]$Action = "start",

    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$ComposeFile = "docker-compose.vllm.yml"
$HealthUrl = "http://localhost:8000/health"

function Check-Docker {
    try {
        docker info | Out-Null
    } catch {
        Write-Error "[ERROR] Docker engine is not reachable. Please start Docker Desktop and ensure WSL2 / GPU integration is enabled."
        exit 1
    }
}

function Start-Serving {
    Check-Docker
    Write-Host "==> Starting vLLM serving container ($ComposeFile)..."
    docker compose -f $ComposeFile up -d
    Write-Host "[OK] Container started in background."
    Write-Host "==> Run '.\scripts\start_vllm.ps1 -Action wait' to monitor readiness."
}

function Stop-Serving {
    Check-Docker
    Write-Host "==> Stopping vLLM serving container..."
    docker compose -f $ComposeFile down
    Write-Host "[OK] Container stopped."
}

function Show-Status {
    Check-Docker
    docker compose -f $ComposeFile ps
}

function Show-Logs {
    Check-Docker
    docker compose -f $ComposeFile logs -f --tail=100
}

function Wait-Readiness {
    Write-Host "==> Waiting for vLLM server to become healthy at $HealthUrl (timeout: ${TimeoutSeconds}s)..."
    $startTime = Get-Date

    while ($true) {
        $elapsed = (New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds
        if ($elapsed -gt $TimeoutSeconds) {
            Write-Error "[ERROR] Timed out waiting for vLLM server readiness after ${TimeoutSeconds}s."
            exit 1
        }

        try {
            $resp = Invoke-WebRequest -Uri $HealthUrl -Method Get -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) {
                Write-Host "[OK] vLLM server is healthy and ready to serve requests! (took $([math]::Round($elapsed, 1))s)"
                return
            }
        } catch {
            # Expected while model weights are downloading or initializing
        }

        Write-Host "  ... initializing / loading weights ($([math]::Round($elapsed, 0))s elapsed)"
        Start-Sleep -Seconds 5
    }
}

switch ($Action) {
    "start"   { Start-Serving }
    "stop"    { Stop-Serving }
    "restart" { Stop-Serving; Start-Serving }
    "status"  { Show-Status }
    "logs"    { Show-Logs }
    "wait"    { Wait-Readiness }
}
