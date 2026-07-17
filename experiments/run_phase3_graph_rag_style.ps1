param(
    [string]$Config = "configs/phase3_graph_rag_style.yaml",
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

& $Python -m src.run_experiment --config $Config
