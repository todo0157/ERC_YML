param(
  [string]$OutDir = "release_assets",
  [string]$Name = "",
  [string[]]$ExcludePatterns = @(
    "\\.git\\",
    "\\.venv\\",
    "\\__pycache__\\",
    "\\.vscode\\",
    "\\release_assets\\",
    "\\snapshots\\",
    "\\dist\\"
  )
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($Name)) {
  $Name = "YML-snapshot-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$outDirFull = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Force -Path $outDirFull | Out-Null

$zipPath = Join-Path $outDirFull ($Name + ".zip")
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }

Write-Host ("Repo root:  " + $repoRoot)
Write-Host ("Output ZIP: " + $zipPath)

$files = Get-ChildItem -Path $repoRoot -Recurse -File -Force |
  Where-Object {
    $full = $_.FullName
    foreach ($p in $ExcludePatterns) {
      if ($full -match $p) { return $false }
    }
    return $true
  } |
  ForEach-Object { $_.FullName }

if (-not $files -or $files.Count -eq 0) {
  throw "No files found to archive. Check ExcludePatterns."
}

Compress-Archive -LiteralPath $files -DestinationPath $zipPath -CompressionLevel Optimal

$sizeMB = [math]::Round(((Get-Item $zipPath).Length / 1MB), 2)
Write-Host ("Done. ZIP size: " + $sizeMB + " MB")


