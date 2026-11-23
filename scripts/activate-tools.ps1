# Dot-source this file to use the workspace-local Node/npm/Neon installation:
# . .\scripts\activate-tools.ps1
$projectToolsRoot = Split-Path -Parent $PSScriptRoot
$projectNodeFile = Join-Path $projectToolsRoot '.tools/node-path.txt'
if (Test-Path -LiteralPath $projectNodeFile) {
    $projectNodeDirectory = Get-Content -LiteralPath $projectNodeFile
    $projectNeonDirectory = Join-Path $projectToolsRoot '.tools/npm-global'
    $env:Path = "$projectNodeDirectory;$projectNeonDirectory;$env:Path"
}
