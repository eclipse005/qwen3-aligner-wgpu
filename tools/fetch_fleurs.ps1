# Fetch + extract the FLEURS test split for every language we can evaluate, from
# the ModelScope mirror (huggingface.co is not reachable from this machine).
#
#   powershell -File tools/fetch_fleurs.ps1
#
# Layout under eval_data/fleurs/ (git-ignored):
#   tsv/<config>.test.tsv          id, filename, raw, normalized, ...
#   audio/<config>/test/*.wav      one file per tsv row
# Re-running resumes a partial download and skips finished languages.

param(
    [string]$Root = "D:\qwen3-asr-wgpu\eval_data\fleurs",
    [string]$Base = "https://modelscope.cn/datasets/pengzhendong/fleurs/resolve/master",
    [int]$LimitPerLang = 200
)

# Same reason as run_fleurs_sweep.ps1: curl/tar write progress and warnings to
# stderr, which PowerShell 5.1 would turn into a terminating error under `Stop`.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
$langsFile = Join-Path $PSScriptRoot "fleurs_langs.tsv"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "tsv"), (Join-Path $Root "audio") | Out-Null

$configs = Get-Content $langsFile |
    Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() -ne '' } |
    ForEach-Object { $c = $_ -split "`t"; [pscustomobject]@{ Lang = $c[0]; Config = $c[1]; Metric = $c[2] } } |
    Where-Object { $_.Config -ne '--' }

foreach ($row in $configs) {
    $cfg = $row.Config
    $tsv = Join-Path $Root "tsv\$cfg.test.tsv"
    $ext = Join-Path $Root "audio\$cfg\test"
    if (-not (Test-Path $tsv)) {
        Invoke-WebRequest -Uri "$Base/data/$cfg/test.tsv" -OutFile $tsv -TimeoutSec 120
    }
    if ((Test-Path $ext) -and (Get-ChildItem "$ext\*.wav" -ErrorAction SilentlyContinue).Count -ge $LimitPerLang) {
        Write-Host "[skip] $cfg ($((Get-ChildItem "$ext\*.wav").Count) wavs already extracted)"
        continue
    }
    $tar = Join-Path $Root "audio\$cfg.test.tar.gz"
    Write-Host "[get ] $cfg"
    curl.exe -sL --retry 3 -C - -o $tar "$Base/data/$cfg/audio/test.tar.gz"
    if ($LASTEXITCODE -ne 0) { Write-Warning "download failed for $cfg (exit $LASTEXITCODE)"; continue }
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "audio\$cfg") | Out-Null
    tar.exe -xzf $tar -C (Join-Path $Root "audio\$cfg")
    Remove-Item $tar
    $n = (Get-ChildItem "$ext\*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host "[done] $cfg : $n wavs"
}

Write-Host "`n=== available ==="
Get-ChildItem (Join-Path $Root "audio") -Directory | ForEach-Object {
    $n = (Get-ChildItem "$($_.FullName)\test\*.wav" -ErrorAction SilentlyContinue).Count
    if ($n -gt 0) { "  {0,-12} {1,5} wavs" -f $_.Name, $n }
}
