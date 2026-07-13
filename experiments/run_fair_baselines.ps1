param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $BundledPython = "C:\Users\happy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $BundledPython) {
        $Python = $BundledPython
    } else {
        $Python = "python"
    }
}

$Configs = @(
    "configs/phase2_vector_rag_fair.yaml",
    "configs/phase2_bm25_rag_fair.yaml",
    "configs/phase2_hybrid_rag_fair.yaml"
)

foreach ($Config in $Configs) {
    Write-Host "Running fair baseline: $Config"
    & $Python -m src.run_experiment --config $Config
}
