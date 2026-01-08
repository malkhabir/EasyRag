# Helper script to run commands with virtual environment
# Usage: .\run.ps1 <command>

$ErrorActionPreference = "Stop"

# Activate virtual environment
$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    & $venvPath
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    Write-Host "Virtual environment created and dependencies installed!" -ForegroundColor Green
} else {
    & $venvPath
}

# Run the provided command
if ($args.Count -gt 0) {
    $command = $args -join " "
    Write-Host "Running: $command" -ForegroundColor Cyan
    Invoke-Expression $command
} else {
    Write-Host "Virtual environment activated. You can now run Python commands." -ForegroundColor Green
}
