param(
    [string]$Python = "python",
    [string]$Config = "configs/phase4_5_lightrag.yaml"
)

$ErrorActionPreference = "Stop"

Write-Host "Running Phase 4.5 LightRAG controlled integration..."
& $Python -m src.run_experiment --config $Config
