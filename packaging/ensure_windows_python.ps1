# Ensures a private per-user CPython for HerbivoR (Windows) with working Tcl/Tk.
# Primary: Astral python-build-standalone (portable tar.gz extract - reliable).
# Fallback: official python.org silent EXE (often fails to honor TargetDir).
$ErrorActionPreference = "Stop"
$PyVersion = "3.12.13"
$StandaloneTag = "20260303"
$Target = Join-Path $env:LOCALAPPDATA "HerbivoR\Python"
$PythonExe = Join-Path $Target "python.exe"

function Test-HerbivoRPython {
    param([string]$Exe)
    if (-not (Test-Path $Exe)) { return $false }
    & $Exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -ne 0) { return $false }

    $probeFile = Join-Path $env:TEMP ("herbivor-tk-probe-" + [guid]::NewGuid().ToString("N") + ".py")
    $probeCode = @(
        "import os, sys"
        "from pathlib import Path"
        "base = Path(sys.base_prefix)"
        "pairs = ["
        "    (base / 'tcl' / 'tcl8.6', base / 'tcl' / 'tk8.6'),"
        "    (base / 'lib' / 'tcl8.6', base / 'lib' / 'tk8.6'),"
        "    (base / 'Library' / 'lib' / 'tcl8.6', base / 'Library' / 'lib' / 'tk8.6'),"
        "]"
        "for tcl_dir, tk_dir in pairs:"
        "    if (tcl_dir / 'init.tcl').is_file() and (tk_dir / 'tk.tcl').is_file():"
        "        os.environ['TCL_LIBRARY'] = str(tcl_dir)"
        "        os.environ['TK_LIBRARY'] = str(tk_dir)"
        "        break"
        "import tkinter as t"
        "r = t.Tk()"
        "r.withdraw()"
        "r.destroy()"
    ) -join "`n"
    try {
        Set-Content -Path $probeFile -Value $probeCode -Encoding UTF8
        & $Exe $probeFile
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -Force $probeFile -ErrorAction SilentlyContinue
    }
}

function Clear-TargetDir {
    if (Test-Path $Target) {
        Write-Host "Removing previous private Python at $Target ..."
        Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
}

function Install-FromStandalone {
    $asset = "cpython-$PyVersion+$StandaloneTag-x86_64-pc-windows-msvc-install_only.tar.gz"
    $urls = @(
        "https://github.com/astral-sh/python-build-standalone/releases/download/$StandaloneTag/$asset",
        "https://github.com/indygreg/python-build-standalone/releases/download/$StandaloneTag/$asset"
    )
    $tmpRoot = Join-Path $env:TEMP ("herbivor-pbs-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
    try {
        $archive = Join-Path $tmpRoot $asset
        $downloaded = $false
        foreach ($url in $urls) {
            Write-Host "Downloading portable Python $PyVersion ..."
            Write-Host $url
            try {
                Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
                if ((Test-Path $archive) -and ((Get-Item $archive).Length -gt 10MB)) {
                    $downloaded = $true
                    break
                }
            } catch {
                Write-Host "  download failed: $($_.Exception.Message)"
            }
        }
        if (-not $downloaded) {
            throw "Could not download python-build-standalone archive"
        }

        Write-Host "Extracting portable Python ..."
        tar -xf $archive -C $tmpRoot
        if ($LASTEXITCODE -ne 0) {
            throw "tar extract failed (exit $LASTEXITCODE)"
        }

        $candidate = Get-ChildItem -Path $tmpRoot -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\python\\python\.exe$' -or $_.Directory.Name -eq 'python' } |
            Select-Object -First 1
        if ($null -eq $candidate) {
            $candidate = Get-ChildItem -Path $tmpRoot -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
                Select-Object -First 1
        }
        if ($null -eq $candidate) {
            throw "python.exe not found inside standalone archive"
        }

        $srcDir = $candidate.Directory.FullName
        Clear-TargetDir
        Write-Host "Installing portable Python to $Target ..."
        Copy-Item -Path (Join-Path $srcDir "*") -Destination $Target -Recurse -Force

        if (-not (Test-Path $PythonExe)) {
            throw "python.exe missing after extract: $PythonExe"
        }
        Write-Host "Portable Python ready: $PythonExe"
    }
    finally {
        Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
    }
}

function Install-FromOfficialExe {
    $Url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    $Tmp = Join-Path $env:TEMP "herbivor-python-3.12.10-amd64.exe"
    Write-Host "Downloading official Python installer (fallback) ..."
    Write-Host $Url
    Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing

    Clear-TargetDir
    Write-Host "Running official installer into $Target ..."
    $argLine = @(
        "/quiet"
        "InstallAllUsers=0"
        "PrependPath=0"
        "Include_doc=0"
        "Include_launcher=0"
        "InstallLauncherAllUsers=0"
        "Include_test=0"
        "Include_tools=0"
        "Include_pip=1"
        "Include_tcltk=1"
        "Include_lib=1"
        "Include_exe=1"
        "Shortcuts=0"
        "AssociateFiles=0"
        "SimpleInstall=1"
        "TargetDir=`"$Target`""
    ) -join " "

    $p = Start-Process -FilePath $Tmp -ArgumentList $argLine -Wait -PassThru
    Remove-Item -Force $Tmp -ErrorAction SilentlyContinue

    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) {
        throw "Python installer exited with code $($p.ExitCode)"
    }
    if (-not (Test-Path $PythonExe)) {
        $fallback = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        if (Test-Path $fallback) {
            Write-Host "Official installer used default path; copying from $(Split-Path $fallback) ..."
            $src = Split-Path $fallback
            Clear-TargetDir
            Copy-Item -Path (Join-Path $src "*") -Destination $Target -Recurse -Force
        }
    }
    if (-not (Test-Path $PythonExe)) {
        throw "python.exe not found at $PythonExe after official installer"
    }
}

if (Test-HerbivoRPython $PythonExe) {
    Write-Host "Private Python already present: $PythonExe"
    exit 0
}

$errors = @()
try {
    Install-FromStandalone
} catch {
    $errors += "Standalone: $($_.Exception.Message)"
    Write-Host "WARNING: portable Python install failed - $($_.Exception.Message)"
    try {
        Install-FromOfficialExe
    } catch {
        $errors += "Official EXE: $($_.Exception.Message)"
        Write-Host "ERROR: $($_.Exception.Message)"
        Write-Error ("Private Python install failed.`n - " + ($errors -join "`n - "))
        exit 1
    }
}

if (-not (Test-HerbivoRPython $PythonExe)) {
    Write-Error "Private Python at $PythonExe is present but tkinter/Tcl-Tk is not usable"
    exit 1
}

Write-Host "Private Python ready: $PythonExe"
exit 0
