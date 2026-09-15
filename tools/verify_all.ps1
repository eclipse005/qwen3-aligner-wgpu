# Regression gate + baseline table: 6 fixtures x 2 models, text compared with the
# frozen python-hf baselines, phase timings printed.
#
#   powershell -File tools/verify_all.ps1                 # all 12 runs
#   powershell -File tools/verify_all.ps1 -Models 0.6B   # one model
#   powershell -File tools/verify_all.ps1 -SkipTiming    # text only
#
# Records the SM clock next to every number: this machine's P104 drops to
# 1151 MHz (from 1911) after a while and everything scales with it, so timings
# from different runs are only comparable when the clock is.

param(
    [string]$Models = "0.6B,1.7B",
    [string]$Fixtures = "15s_en,30s_zh,90s_en,90s_ja,180s_en,180s_zh",
    [switch]$SkipTiming,
    [string]$Out = ""
)

$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo "target\release\transcribe.exe"
$fxDir = "D:\qwen3-asr-rs\tests\fixtures"
$baseDir = "D:\qwen3-asr-rs\docs\baseline\texts"

# Same rule as the reference gold harness: max(256, min(2048, audio_s*8 + 64)).
function MaxNew([string]$name) {
    $sec = [int]($name -replace 's_.*', '')
    [Math]::Max(256, [Math]::Min(2048, $sec * 8 + 64))
}

$clock = (nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader)
Write-Host "GPU: $clock`n"
$rows = @()
foreach ($model in $Models.Split(',')) {
    $dir = "D:\Qwen3-ASR\models\Qwen3-ASR-$model-hf"
    foreach ($fx in $Fixtures.Split(',')) {
        $wav = Join-Path $fxDir "$fx.wav"
        $base = Join-Path $baseDir "python-hf_${model}_$fx.txt"
        if (-not (Test-Path $wav)) { continue }
        $mn = MaxNew $fx
        $err = & $exe --model $dir --wav $wav --adapter nvidia --max-new $mn --baseline $base 2>&1
        $verdict = (($err | Select-String -Pattern 'MATCH|MISMATCH').Line -join " ").Trim()
        $phases = if ($SkipTiming) { "" } else { ($err | Select-String -Pattern 'mel=').Line }
        $tot = if ($SkipTiming) { "" } else { ($err | Select-String -Pattern 'elapsed=').Line }
        $rows += [pscustomobject]@{ Model = $model; Fixture = $fx; Verdict = $verdict; Phases = $phases; Total = $tot }
        Write-Host ("{0,-5} {1,-9} {2}" -f $model, $fx, $verdict)
        if ($phases) { Write-Host "      $phases" }
        if ($tot) { Write-Host "      $tot" }
    }
}
$bad = ($rows | Where-Object { $_.Verdict -notmatch 'MATCH' -or $_.Verdict -match 'MISMATCH' }).Count
Write-Host "`n=== $($rows.Count) runs, $bad not matching ==="
if ($Out) { $rows | Export-Csv -NoTypeInformation -Encoding utf8 $Out; Write-Host "wrote $Out" }
