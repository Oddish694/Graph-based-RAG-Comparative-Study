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
    "configs/phase2_hybrid_rag_fair.yaml",
    "configs/phase3_graph_rag_style.yaml",
    "configs/phase4_improved_lightrag.yaml",
    "configs/phase5_improved_lightrag_no_aliases.yaml",
    "configs/phase5_improved_lightrag_no_graph_expansion.yaml",
    "configs/phase5_improved_lightrag_no_coverage_reranking.yaml",
    "configs/phase5_improved_lightrag_no_entity_coverage.yaml"
)

foreach ($Config in $Configs) {
    Write-Host "Running ablation config: $Config"
    & $Python -m src.run_experiment --config $Config
}

& $Python -m src.evaluation.summarize_results `
    --output "results/ablation_table.csv" `
    "results/phase2_5_fair_hybrid_rag" `
    "results/phase3_graph_rag_style" `
    "results/phase4_improved_lightrag" `
    "results/phase5_improved_lightrag_no_aliases" `
    "results/phase5_improved_lightrag_no_graph_expansion" `
    "results/phase5_improved_lightrag_no_coverage_reranking" `
    "results/phase5_improved_lightrag_no_entity_coverage"
