<#
.SYNOPSIS
    Publish a new Herbivora release with Windows, macOS, and Linux installers.

.DESCRIPTION
    1. Checks: on main, up to date with origin, gh logged in, tag unused,
       version newer than VERSION, CHANGELOG.md has a "## [X.Y.Z]" section,
       Python sources compile.
    2. Writes VERSION, rebuilds packaging/installer_license.txt, commits, tags
       vX.Y.Z, and pushes main + tag.
    3. Creates the GitHub Release (notes = CHANGELOG section + download table).
    4. Waits for .github/workflows/release.yml, which builds from the tag:
         Herbivora-Setup-vX.Y.Z.exe, Herbivora-vX.Y.Z.dmg,
         Herbivora-vX.Y.Z-source-linux.tar.gz, SHA256SUMS
    5. Fails unless all four files are attached to the Release.

    Untracked files are never committed. Modified tracked files abort the run
    unless -CommitTracked is given.

.EXAMPLE
    .\packaging\release.ps1 -Version 1.4.4 -DryRun
.EXAMPLE
    .\packaging\release.ps1 -Version 1.4.4 -CommitTracked
.EXAMPLE
    # Rebuild installers for an existing tag (e.g. after a CI failure)
    .\packaging\release.ps1 -Version 1.4.3 -RebuildOnly
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    # Also commit modified tracked files (git add -u). Untracked files stay out.
    [switch]$CommitTracked,

    # Run every check and print the release notes, but change nothing.
    [switch]$DryRun,

    # Skip commit/tag/release; re-run the build workflow for an existing tag.
    [switch]$RebuildOnly
)

$ErrorActionPreference = 'Stop'
$Workflow = 'release.yml'
$Branch = 'main'

function Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }

# No param() block on purpose: native flags such as -m / -f must reach $args untouched.
function Invoke-Native {
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($rest -join ' ')" }
}

function Get-Output {
    $exe = $args[0]
    $rest = @($args | Select-Object -Skip 1)
    $out = & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $exe $($rest -join ' ')" }
    return $out
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Find-Python {
    $candidates = @(
        (Join-Path $Root '.venv\Scripts\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Herbivora\Python\python.exe')
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    throw 'Python not found (.venv, %LOCALAPPDATA%\Herbivora\Python, or PATH).'
}

$Version = $Version.Trim().TrimStart('v')
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version must look like 1.4.4 (got '$Version')." }
$Tag = "v$Version"
$ExpectedAssets = @(
    "Herbivora-Setup-$Tag.exe",
    "Herbivora-$Tag.dmg",
    "Herbivora-$Tag-source-linux.tar.gz",
    'SHA256SUMS'
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

# ---------------------------------------------------------------- checks
Step 'Checking GitHub CLI login'
Invoke-Native gh auth status | Out-Null
$Repo = (Get-Output gh repo view --json nameWithOwner --jq .nameWithOwner).Trim()
Write-Host "Repository: $Repo"

Step "Checking branch and sync with origin/$Branch"
$current = (Get-Output git branch --show-current).Trim()
if ($current -ne $Branch) { throw "Switch to '$Branch' first (current: '$current')." }
Invoke-Native git fetch origin $Branch --tags --quiet
$behind = [int](Get-Output git rev-list --count "HEAD..origin/$Branch")
if ($behind -gt 0) { throw "Local $Branch is $behind commit(s) behind origin. Run: git pull --rebase" }

if (-not $RebuildOnly) {
    Step 'Checking working tree'
    $dirty = @(Get-Output git status --porcelain --untracked-files=no)
    if ($dirty.Count -gt 0) {
        Write-Host ($dirty -join "`n")
        if (-not $CommitTracked) {
            throw 'Tracked files have uncommitted changes. Commit them, or re-run with -CommitTracked to include them in the release commit.'
        }
        Write-Host 'These tracked changes will be included in the release commit (-CommitTracked).' -ForegroundColor Yellow
    }
    $untracked = @(Get-Output git ls-files --others --exclude-standard)
    if ($untracked.Count -gt 0) {
        Write-Host "$($untracked.Count) untracked file(s) will NOT be in the release:" -ForegroundColor Yellow
        $untracked | ForEach-Object { Write-Host "  $_" }
    }

    Step "Checking tag $Tag is unused"
    & git rev-parse -q --verify "refs/tags/$Tag" *> $null
    if ($LASTEXITCODE -eq 0) { throw "Tag $Tag already exists locally. Use -RebuildOnly to rebuild its installers." }
    $remoteTag = Get-Output git ls-remote --tags origin "refs/tags/$Tag"
    if ($remoteTag) { throw "Tag $Tag already exists on origin. Use -RebuildOnly to rebuild its installers." }

    $oldVersion = (Get-Content (Join-Path $Root 'VERSION') -Raw).Trim()
    if ([version]$Version -le [version]$oldVersion) {
        throw "New version $Version must be greater than current VERSION $oldVersion."
    }
    Write-Host "VERSION: $oldVersion -> $Version"

    Step "Reading CHANGELOG.md section [$Version]"
    $changelog = Get-Content (Join-Path $Root 'CHANGELOG.md') -Raw
    $pattern = '(?ms)^## \[' + [regex]::Escape($Version) + '\][^\r\n]*\r?\n(.*?)(?=^## \[|\z)'
    $m = [regex]::Match($changelog, $pattern)
    if (-not $m.Success -or -not $m.Groups[1].Value.Trim()) {
        throw "CHANGELOG.md has no '## [$Version] - YYYY-MM-DD' section with content. Add one above the previous release, then re-run."
    }
    $notesBody = $m.Groups[1].Value.Trim()

    $firstBullet = ($notesBody -split "`r?`n" | Where-Object { $_ -match '^\s*-\s+' } | Select-Object -First 1)
    $summary = if ($firstBullet) { ($firstBullet -replace '^\s*-\s+', '' -replace '\*\*|`', '').Trim() } else { 'see CHANGELOG' }
    if ($summary.Length -gt 90) { $summary = $summary.Substring(0, 87) + '...' }
    $commitMessage = "Release ${Version}: $summary"

    $notes = @"
$notesBody

## Downloads

| System | File | How to install |
|--------|------|----------------|
| Windows 10/11 | ``Herbivora-Setup-$Tag.exe`` | Double-click and follow the wizard. If SmartScreen appears: **More info**, then **Run anyway**. |
| macOS 11+ | ``Herbivora-$Tag.dmg`` | Open it, drag **Herbivora** to **Applications**, open it from Applications. |
| Linux | ``Herbivora-$Tag-source-linux.tar.gz`` | ``tar -xzf Herbivora-$Tag-source-linux.tar.gz && cd Herbivora-$Tag && ./install.sh`` |

Verify downloads with ``SHA256SUMS``. Full steps: [USER_GUIDE.md](https://github.com/$Repo/blob/$Tag/USER_GUIDE.md).
"@

    Step 'Compiling Python sources'
    $python = Find-Python
    Write-Host "Python: $python"
    Invoke-Native $python -m compileall -q -x '[\\/](\.venv|\.git|hf_cache|dist|build|train_contour[\\/]runs)[\\/]' .

    if ($DryRun) {
        Step 'Dry run - nothing was changed'
        Write-Host "Commit message: $commitMessage"
        Write-Host "Tag:            $Tag"
        Write-Host "Assets:         $($ExpectedAssets -join ', ')"
        Write-Host "`n----- Release notes -----`n$notes"
        exit 0
    }

    # ---------------------------------------------------------- publish
    Step "Writing VERSION and installer license"
    Write-Utf8NoBom (Join-Path $Root 'VERSION') "$Version`n"
    Invoke-Native $python packaging/build_installer_license.py

    Step 'Committing release'
    if ($CommitTracked) { Invoke-Native git add -u }
    Invoke-Native git add VERSION CHANGELOG.md packaging/installer_license.txt
    Invoke-Native git commit -m $commitMessage

    Step "Tagging and pushing $Tag"
    Invoke-Native git tag -a $Tag -m "Herbivora $Tag"
    Invoke-Native git push origin $Branch
    Invoke-Native git push origin $Tag

    Step 'Creating GitHub Release'
    $notesFile = Join-Path ([System.IO.Path]::GetTempPath()) "herbivora-notes-$Tag.md"
    Write-Utf8NoBom $notesFile $notes
    Invoke-Native gh release create $Tag --title "Herbivora $Tag" --notes-file $notesFile --verify-tag
    Remove-Item $notesFile -ErrorAction SilentlyContinue
    $runEvent = 'release'
}
else {
    Step "Rebuilding installers for existing $Tag"
    Invoke-Native gh release view $Tag --json tagName | Out-Null
    if ($DryRun) {
        Write-Host "Dry run: would run '$Workflow' from $Branch with tag=$Tag."
        exit 0
    }
    Invoke-Native gh workflow run $Workflow --ref $Branch -f "tag=$Tag"
    $runEvent = 'workflow_dispatch'
}

# ---------------------------------------------------------------- wait for CI
Step "Waiting for workflow '$Workflow' to start"
$runId = $null
$branchFilter = if ($runEvent -eq 'release') { $Tag } else { $Branch }
for ($i = 0; $i -lt 40 -and -not $runId; $i++) {
    Start-Sleep -Seconds 6
    $runs = Get-Output gh run list --workflow $Workflow --event $runEvent --branch $branchFilter --limit 1 --json 'databaseId,status,createdAt'
    $run = $runs | ConvertFrom-Json | Select-Object -First 1
    if ($run -and ([datetime]$run.createdAt).ToUniversalTime() -gt (Get-Date).ToUniversalTime().AddMinutes(-10)) {
        $runId = $run.databaseId
    }
}
if (-not $runId) { throw "No '$Workflow' run appeared. Check https://github.com/$Repo/actions" }
Write-Host "Run: https://github.com/$Repo/actions/runs/$runId"

Step 'Building Windows, macOS, and Linux installers (usually 5-10 min)'
& gh run watch $runId --exit-status --interval 20
if ($LASTEXITCODE -ne 0) {
    throw "Build failed. Logs: https://github.com/$Repo/actions/runs/$runId  (fix, then: .\packaging\release.ps1 -Version $Version -RebuildOnly)"
}

# ---------------------------------------------------------------- verify
Step 'Verifying release assets'
$assets = @(Get-Output gh release view $Tag --json assets --jq '.assets[].name')
$missing = @($ExpectedAssets | Where-Object { $assets -notcontains $_ })
$assets | ForEach-Object { Write-Host "  $_" }
if ($missing.Count -gt 0) { throw "Missing on release: $($missing -join ', ')" }

$url = (Get-Output gh release view $Tag --json url --jq .url).Trim()
Write-Host "`nRelease $Tag is complete: $url" -ForegroundColor Green
