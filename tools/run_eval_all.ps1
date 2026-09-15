# The FLEURS comparison, STRICTLY ONE JOB AT A TIME.
#
# Four runs over all 29 languages, 20 clips each (evenly spaced by file size, so
# short and long clips are both covered), then the final table:
#
#   wgpu   0.6B   tag w06     python 0.6B   tag py06
#   wgpu   1.7B   tag w17     python 1.7B   tag py17
#
# Eight GB of VRAM cannot hold two ASR clients (1.7B fp16 + a Python run + the
# wgpu buffers went to 7.6/8.0 GB and one run died with
# `Error occurred when trying to async map a buffer`), so every step below waits
# for the previous one, and every step resumes from what is already on disk.
#
#   powershell -File tools/run_eval_all.ps1

$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$py = "C:\Users\ADMIN\miniconda3\envs\asr\python.exe"
$env:PYTHONUTF8 = "1"
$model06 = "D:\Qwen3-ASR\models\Qwen3-ASR-0.6B-hf"
$model17 = "D:\Qwen3-ASR\models\Qwen3-ASR-1.7B-hf"
$n = 20

function Step($name, $block) {
    Write-Host "`n=== $name ===  $((Get-Date).ToString('HH:mm:ss'))"
    & $block
    Write-Host "=== $name done ===  $((Get-Date).ToString('HH:mm:ss'))"
}

Step "wgpu 0.6B" {
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_fleurs_sweep.ps1 `
        -Model $model06 -Tag w06 -Even $n
}
Step "wgpu 1.7B" {
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_fleurs_sweep.ps1 `
        -Model $model17 -Tag w17 -Even $n
}
Step "python 0.6B" {
    & $py tools\run_python_hf.py --model $model06 `
        --root eval_data/fleurs --even $n --tag py06 --max-new 256
}
Step "python 1.7B" {
    & $py tools\run_python_hf.py --model $model17 `
        --root eval_data/fleurs --even $n --tag py17 --max-new 256
}
Step "final table" {
    & $py tools\score_final.py --tags "w06=w06,w17=w17,py06=py06,py17=py17" `
        --out docs\eval-fleurs-final.md
}
Write-Host "`nRUN_EVAL_ALL_DONE"
