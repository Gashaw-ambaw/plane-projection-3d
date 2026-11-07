# run_examples.ps1 - run Ex0.py on every nrrd/csv pair of the class-drive folder.
#
# Run from anywhere, in PowerShell:
#   .\scripts\run_examples.ps1 -Drive "C:\path\to\the\unzipped\folder"
# If PowerShell blocks the script (execution policy), run it like this instead:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_examples.ps1 -Drive "C:\path\to\the\unzipped\folder"
# The folder must hold the .nrrd files and a csv\ subfolder with the lecturer's CSV files
# (they are not in this repo).
# Add -IncludeEdgeCases to also run edge_cases.csv (a robustness check; its frame 16, a plane
# with a 5102 x 10199 image, is above the image size limit of 2048 per side and becomes an
# error frame, so the video is still written).
#
# Writes one mp4 per pair into out_examples\ in the repo root (ignored by git, see .gitignore).
param(
    [Parameter(Mandatory = $true)][string]$Drive,
    [string]$Out,
    [string]$Python,
    [switch]$IncludeEdgeCases
)

# $PSScriptRoot is empty inside param(), so the defaults are set here.
$Root = Split-Path $PSScriptRoot -Parent
if (-not $Out) { $Out = Join-Path $Root "out_examples" }
if (-not $Python) { $Python = Join-Path $Root ".venv\Scripts\python.exe" }
$Ex0 = Join-Path $Root "Ex0.py"
if (-not (Test-Path $Python)) { Write-Error "Python not found: $Python"; exit 1 }
if (-not (Test-Path (Join-Path $Drive "csv"))) { Write-Error "No csv folder in: $Drive"; exit 1 }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# volume name, csv name
$pairs = @(
    @("MRHead_1",     "lecturer_example"),
    @("MRHead_1",     "head"),
    @("onion",        "onion"),
    @("swiss_roll",   "swiss_roll"),
    @("biscuit_cake", "biscuit_cake"),
    @("croissant",    "croissant")
)
if ($IncludeEdgeCases) { $pairs += ,@("MRHead_1", "edge_cases") }

$results = @()
foreach ($p in $pairs) {
    $nrrd = Join-Path $Drive ($p[0] + ".nrrd")
    $csv  = Join-Path $Drive ("csv\" + $p[1] + ".csv")
    $mp4  = Join-Path $Out ($p[0] + "__" + $p[1] + ".mp4")
    Write-Host ""
    Write-Host "=== $($p[0]) + $($p[1]) ==="
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Python $Ex0 $nrrd $csv $mp4
    $code = $LASTEXITCODE
    $sw.Stop()
    # A failed run can leave a half-written mp4; never report it as a result.
    if ($code -ne 0 -and (Test-Path $mp4)) { Remove-Item $mp4 -Force }
    $size = if (Test-Path $mp4) { [math]::Round((Get-Item $mp4).Length / 1MB, 2) } else { 0 }
    $results += [pscustomobject]@{
        Volume = $p[0]; CSV = $p[1]; ExitCode = $code
        Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); MP4_MB = $size
    }
}

Write-Host ""
Write-Host "=== Summary ==="
$results | Format-Table -AutoSize
