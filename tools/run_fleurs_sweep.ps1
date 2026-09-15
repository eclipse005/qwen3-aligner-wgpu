# Run the port over every available FLEURS language and dump hypotheses.
#
#   powershell -File tools/run_fleurs_sweep.ps1 -Model D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf -Tag 0p6
#
# Writes, per language: eval_data/fleurs/<config>.<tag>.hyps.tsv + .sweep.log.
# Re-running skips languages whose hypothesis dump already has enough lines.

param(
    [string]$Model = "D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf",
    [string]$Tag = "0p6",
    [int]$Limit = 200,
    # Evenly spaced clips across the test set (sorted by size ~ duration) instead
    # of the head; 0 = use -Limit.  Both systems must use the same value.
    [int]$Even = 0,
    [string]$Root = "D:\qwen3-asr-wgpu\eval_data\fleurs",
    [string]$Langs = "",
    [int]$MaxNew = 256,
    # Force the language into the prompt instead of letting the model detect it
    # (the "does telling it help?" experiment; the tag keeps it separate).
    [switch]$ForceLang
)

# Windows PowerShell 5.1 turns a native command's stderr into a terminating error
# under `Stop`, and eval_asr prints its GPU banner there — keep `Continue` and
# judge each run by $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo "target\release\eval_asr.exe"
if (-not (Test-Path $exe)) { throw "build it first: cargo build --release --bin eval_asr" }

$table = Get-Content (Join-Path $PSScriptRoot "fleurs_langs.tsv") |
    Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' } |
    ForEach-Object { $c = $_ -split "`t"; [pscustomobject]@{ Lang = $c[0]; Config = $c[1]; Metric = $c[2] } } |
    Where-Object { $_.Config -ne '--' }
if ($Langs) {
    $want = $Langs.Split(',')
    $table = $table | Where-Object { $want -contains $_.Config }
}

$t0 = Get-Date
foreach ($row in $table) {
    $cfg = $row.Config
    $hyps = Join-Path $Root "$cfg.$Tag.hyps.tsv"
    $dir = Join-Path $Root "audio\$cfg\test"
    $want = if ($Even -gt 0) { $Even } else { $Limit }
    $sel = if ($Even -gt 0) { @("--even", "$Even") } else { @("--limit", "$Limit") }
    if (-not (Test-Path $dir)) { Write-Host "[skip] $cfg (no audio)"; continue }
    if ((Test-Path $hyps) -and ((Get-Content $hyps | Measure-Object -Line).Lines -ge $want)) {
        Write-Host "[skip] $cfg (hypotheses present)"; continue
    }
    $log = Join-Path $Root "$cfg.$Tag.sweep.log"
    Write-Host "[run ] $cfg  $($row.Lang)"
    # Two attempts: a second GPU client (the Python reference sweep) can make the
    # wgpu device creation fail transiently, and that is not worth a restart.
    $ok = $false
    $langArg = if ($ForceLang) { @("--lang", $row.Lang) } else { @() }
    foreach ($attempt in 1..2) {
        & $exe --model $Model `
            --tsv (Join-Path $Root "tsv\$cfg.test.tsv") `
            --audio-dir $dir --mode $row.Metric --max-new $MaxNew @sel `
            --adapter nvidia --quiet --hyps-out $hyps `
            --lang-out (Join-Path $Root "$cfg.$Tag.langs.tsv") `
            --summary-out (Join-Path $Root "$cfg.$Tag.summary.tsv") @langArg 2> $log |
            Out-File -Encoding utf8 (Join-Path $Root "$cfg.$Tag.summary.txt")
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Host "       attempt $attempt failed (exit $LASTEXITCODE), retrying"
        Start-Sleep -Seconds 10
    }
    if (-not $ok) { Write-Host "[FAIL] $cfg"; continue }
    $rtfx = (Select-String -Path (Join-Path $Root "$cfg.$Tag.summary.txt") -Pattern 'RTFx' -ErrorAction SilentlyContinue)
    Write-Host "       $(if ($rtfx) { $rtfx.Line } else { 'no RTFx line' })"
}
Write-Host "`nsweep $Tag finished in $([int]((Get-Date) - $t0).TotalMinutes) min"
