# Grille d'experiences complete : 6 modeles x 4 horizons sur ETTh1.
# Chaque run est entraine 15 epochs avec early stopping (patience=5).
# Les resultats sont ajoutes a results/metrics/comparison.csv.
# Les logs par run sont ecrits dans results/logs/<model>_h<horizon>.log
#
# Usage :
#   cd c:\Users\lenovo\projet7-transformers-ts
#   .\scripts\run_grid.ps1

# Important : PowerShell traite par defaut stderr comme une "erreur".
# Notre logger Python ecrit INFO sur stderr -> ne pas utiliser "Stop".
$ErrorActionPreference = "Continue"

$models = @("DLinear", "Transformer", "PatchTST", "Informer", "Autoformer", "FEDformer")
$horizons = @(96, 192, 336, 720)
$python = ".\venv\Scripts\python.exe"

$logDir = "results\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$total = $models.Count * $horizons.Count
$i = 0
$startAll = Get-Date
$failed = @()

foreach ($m in $models) {
    foreach ($h in $horizons) {
        $i++
        $tag = "$m" + "_h$h"
        $logFile = Join-Path $logDir "$tag.log"
        $startRun = Get-Date

        Write-Host ""
        Write-Host "====================================================================="
        Write-Host "[$i/$total] $tag   demarre a $(Get-Date -Format HH:mm:ss)"
        Write-Host "====================================================================="

        # 2>&1 fusionne stderr+stdout. Tee-Object ecrit a l'ecran ET dans le log.
        & $python -m src.train --model $m --data ETTh1 --pred_len $h --epochs 15 2>&1 | Tee-Object -FilePath $logFile

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[$i/$total] $tag ECHEC (exit=$LASTEXITCODE) -> on continue"
            $failed += $tag
        } else {
            $dur = (Get-Date) - $startRun
            Write-Host ("[$i/$total] $tag termine en {0:N1} min" -f $dur.TotalMinutes)
        }
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Runs en echec : $($failed -join ', ')"
}

$totalDur = (Get-Date) - $startAll
Write-Host ""
Write-Host "====================================================================="
Write-Host ("Grille complete : {0} runs en {1:N1} h" -f $total, $totalDur.TotalHours)
Write-Host "Resultats dans results\metrics\comparison.csv"
Write-Host "====================================================================="
