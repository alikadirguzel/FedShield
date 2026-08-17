# FedShield Windows helper (Make substitute for PowerShell)
# Usage:
#   .\dev.ps1 install
#   .\dev.ps1 data
#   .\dev.ps1 test
#   .\dev.ps1 up
#   .\dev.ps1 down
#   .\dev.ps1 logs
#   .\dev.ps1 e2e

param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "lint", "test", "test-unit", "test-integration", "data", "build", "up", "start", "down", "logs", "health", "e2e", "clean")]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Assert-Docker {
    try {
        docker info 1>$null 2>$null
        if ($LASTEXITCODE -ne 0) { throw "docker info failed" }
    } catch {
        Write-Host @"
Docker Desktop is not running (or not installed).

1) Start Docker Desktop and wait until it says 'Engine running'
2) Then re-run:  .\dev.ps1 up

MLflow UI will be at http://localhost:5000 only after containers start.
"@ -ForegroundColor Yellow
        exit 1
    }
}

function Get-Python {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        return ".\.venv\Scripts\python.exe"
    }
    return "python"
}

function Invoke-Install {
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\pip.exe install -r requirements.txt
}

function Invoke-Data {
    $py = Get-Python
    & $py scripts\generate_non_iid_data.py
}

function Invoke-Test {
    $py = Get-Python
    $env:FEDSHIELD_DEV_CERTS = "true"
    & $py -m pytest tests -q
}

function Invoke-TestUnit {
    $py = Get-Python
    $env:FEDSHIELD_DEV_CERTS = "true"
    & $py -m pytest tests\unit -q
}

function Invoke-TestIntegration {
    $py = Get-Python
    $env:FEDSHIELD_DEV_CERTS = "true"
    & $py -m pytest tests\integration -q
}

function Invoke-Lint {
    $py = Get-Python
    & $py -m ruff check src tests scripts
}

function Invoke-Build {
    Assert-Docker
    docker compose build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Up {
    Assert-Docker
    Invoke-Data
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "docker compose up failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ""
    Write-Host "Stack starting. Check status with:  docker compose ps" -ForegroundColor Green
    Write-Host "MLflow UI: http://localhost:5000"
    Write-Host "Vault:     http://localhost:8200"
    Write-Host "Logs:      .\dev.ps1 logs"
    Write-Host "Next time (fast):  .\dev.ps1 start"
}

function Invoke-Start {
    # Fast restart: no rebuild, no data regen — use after first successful .\dev.ps1 up
    Assert-Docker
    docker compose up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "docker compose start failed. If images are missing, run: .\dev.ps1 up" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Stack started (no rebuild)." -ForegroundColor Green
    Write-Host "MLflow UI: http://localhost:5000"
}

function Invoke-Down {
    Assert-Docker
    docker compose down -v
}

function Invoke-Logs {
    Assert-Docker
    docker compose logs -f --tail=200
}

function Invoke-E2E {
    Assert-Docker
    Invoke-Data
    Invoke-Build
    docker compose up --abort-on-container-exit --exit-code-from server
    docker compose ps
    Write-Host "MLflow UI: http://localhost:5000"
}

function Invoke-Clean {
    Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    @(".pytest_cache", ".ruff_cache", "htmlcov", "mlruns") | ForEach-Object {
        if (Test-Path $_) { Remove-Item -Recurse -Force $_ }
    }
}

function Invoke-Health {
    Assert-Docker
    Write-Host "`n=== Container status ===" -ForegroundColor Cyan
    docker compose ps -a

    $checks = @()

    # Expected running services
    $psJson = docker compose ps --format json 2>$null
    $running = @()
    if ($psJson) {
        $lines = $psJson -split "`n" | Where-Object { $_.Trim() -ne "" }
        foreach ($line in $lines) {
            try { $running += ($line | ConvertFrom-Json) } catch { }
        }
    }

    function Test-Http([string]$Url) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
        } catch { return $false }
    }

    function Test-Tcp([string]$HostName, [int]$Port) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.ReceiveTimeout = 2000
            $tcp.SendTimeout = 2000
            $tcp.Connect($HostName, $Port)
            $ok = $tcp.Connected
            $tcp.Close()
            return $ok
        } catch { return $false }
    }

    function Test-ContainerState([string]$Name) {
        $state = docker inspect -f "{{.State.Status}}" $Name 2>$null
        $health = docker inspect -f "{{if .State.Health}}{{.State.Health.Status}}{{end}}" $Name 2>$null
        if ($health) { return ($health -eq "healthy") }
        return ($state -eq "running")
    }

    $mlflowOk = Test-Http "http://localhost:5000"
    $vaultOk = Test-ContainerState "fedshield-vault"
    $serverOk = Test-ContainerState "fedshield-server"

    $vaultInit = docker inspect -f "{{.State.ExitCode}}" fedshield-vault-init 2>$null
    $vaultInitOk = ($vaultInit -eq "0")

    $need = @("vault", "mlflow", "server", "client-alpha", "client-beta", "client-gamma")
    $upNames = @()
    foreach ($c in $running) {
        if ($c.State -eq "running" -or $c.Status -like "Up*") {
            $upNames += $c.Service
        }
    }

    Write-Host "`n=== Health checks ===" -ForegroundColor Cyan
    $rows = @(
        @{ Name = "Vault container"; Ok = $vaultOk }
        @{ Name = "Vault-init exit 0"; Ok = $vaultInitOk }
        @{ Name = "MLflow UI :5000"; Ok = $mlflowOk }
        @{ Name = "FL Server container"; Ok = $serverOk }
    )
    foreach ($svc in $need) {
        $rows += @{ Name = "Container $svc running"; Ok = ($upNames -contains $svc) }
    }

    $failed = 0
    foreach ($row in $rows) {
        if ($row.Ok) {
            Write-Host ("  [OK]  " + $row.Name) -ForegroundColor Green
        } else {
            Write-Host ("  [FAIL] " + $row.Name) -ForegroundColor Red
            $failed++
        }
    }

    Write-Host "`n=== Recent FL activity (server logs) ===" -ForegroundColor Cyan
    docker compose logs server --tail=15 2>$null

    Write-Host "`n=== How to read this ===" -ForegroundColor Cyan
    Write-Host "1) All [OK] + MLflow opens in browser => infra is up"
    Write-Host "2) Server logs show 'fit_round' / 'Aggregated fit metrics' => training works"
    Write-Host "3) Open http://localhost:5000 and look for experiment 'fedshield-federated-dp'"
    Write-Host "4) If vault-init FAIL: run  .\dev.ps1 down  then  .\dev.ps1 up"

    if ($failed -gt 0) {
        Write-Host "`nResult: NOT healthy ($failed checks failed)" -ForegroundColor Red
        exit 1
    }
    Write-Host "`nResult: healthy" -ForegroundColor Green
}

function Show-Help {
    @"
FedShield targets (PowerShell):
  .\dev.ps1 install            Create venv + install deps (once)
  .\dev.ps1 up                 First run / after code changes (build)
  .\dev.ps1 start              Daily restart (fast, no rebuild)
  .\dev.ps1 health             Verify stack is really working
  .\dev.ps1 down               Stop stack + volumes
  .\dev.ps1 logs               Tail compose logs
  .\dev.ps1 test               Unit + integration tests
  .\dev.ps1 data               Generate Non-IID partitions
  .\dev.ps1 build              Build docker images only
  .\dev.ps1 lint               Run ruff
  .\dev.ps1 e2e                data + build + foreground compose
  .\dev.ps1 clean              Remove caches
"@
}

switch ($Command) {
    "help" { Show-Help }
    "install" { Invoke-Install }
    "lint" { Invoke-Lint }
    "test" { Invoke-Test }
    "test-unit" { Invoke-TestUnit }
    "test-integration" { Invoke-TestIntegration }
    "data" { Invoke-Data }
    "build" { Invoke-Build }
    "up" { Invoke-Up }
    "start" { Invoke-Start }
    "down" { Invoke-Down }
    "logs" { Invoke-Logs }
    "health" { Invoke-Health }
    "e2e" { Invoke-E2E }
    "clean" { Invoke-Clean }
}
