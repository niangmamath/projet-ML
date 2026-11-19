# Interpretation de chaque modele a horizon 96 (le plus parlant visuellement).
# Genere : erreur par horizon, exemples de predictions, importance permutation,
# et une carte d'attention pour le Vanilla Transformer.

$ErrorActionPreference = "Continue"

$python = ".\venv\Scripts\python.exe"
$models = @("DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer")
$logDir = "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($m in $models) {
    $extra = @()
    if ($m -eq "Transformer") { $extra = @("--attention") }

    Write-Host ""
    Write-Host "====================================================================="
    Write-Host "Interpretation de $m (h=96)"
    Write-Host "====================================================================="

    & $python -m src.interpret --model $m --data ETTh1 --pred_len 96 --n_repeats 3 @extra 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "interpret_$m.log")

    if ($LASTEXITCODE -ne 0) {
        Write-Host "$m : ECHEC (exit=$LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Toutes les interpretations sont dans results\figures\"
