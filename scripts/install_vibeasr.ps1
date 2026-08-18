param(
    [string]$SourceDir = "",
    [string]$MinGWBin = "",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ToolsRoot = Join-Path $ProjectRoot ".tools"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PortableToolsRoot = Join-Path $env:LOCALAPPDATA "OtterLocalTools"
$WinLibsRoot = Join-Path $PortableToolsRoot "winlibs"
$WinLibsUrl = "https://github.com/brechtsanders/winlibs_mingw/releases/download/16.2.0posix-14.0.0-ucrt-r1/winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r1.zip"
$WinLibsSha256 = "c1f52294597c0b73786b2a78eb5d176d89226d2f21875eab75e783a8b1cefcc4"

if (-not (Test-Path $VenvPython)) {
    throw "Missing app virtual environment: $VenvPython"
}

if (-not $SourceDir) {
    $SourceDir = Join-Path $ToolsRoot "vibeasr"
}
$SourceDir = [IO.Path]::GetFullPath($SourceDir)

if (-not (Test-Path (Join-Path $SourceDir "CMakeLists.txt")) -and $Bootstrap) {
    New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
    & git clone --recursive https://github.com/microsoft/VibeASR.cpp.git $SourceDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $MinGWBin) {
    $portableBin = Join-Path $WinLibsRoot "mingw64\bin"
    if (Test-Path (Join-Path $portableBin "gcc.exe")) {
        $MinGWBin = $portableBin
    }
}
if (-not $MinGWBin) {
    $gcc = Get-Command gcc -ErrorAction SilentlyContinue
    if ($gcc) {
        $MinGWBin = Split-Path -Parent $gcc.Source
    }
}
if ((-not $MinGWBin -or -not (Test-Path (Join-Path $MinGWBin "gcc.exe"))) -and $Bootstrap) {
    $archive = Join-Path $ToolsRoot "winlibs-x86_64.zip"
    New-Item -ItemType Directory -Force -Path $ToolsRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $PortableToolsRoot | Out-Null
    if (-not (Test-Path $archive)) {
        Write-Host "Downloading portable WinLibs MinGW-w64 toolchain..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $WinLibsUrl -OutFile $archive
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $WinLibsSha256) {
        throw "WinLibs archive checksum mismatch. Expected $WinLibsSha256; got $actualHash."
    }
    Write-Host "Extracting portable compiler..." -ForegroundColor Cyan
    Expand-Archive -LiteralPath $archive -DestinationPath $WinLibsRoot -Force
    $MinGWBin = Join-Path $WinLibsRoot "mingw64\bin"
}
if (-not $MinGWBin -or -not (Test-Path (Join-Path $MinGWBin "gcc.exe"))) {
    throw "VibeASR.cpp needs MinGW-w64. Re-run with -Bootstrap or set -MinGWBin to the folder containing gcc.exe and mingw32-make.exe."
}

$env:PATH = "$MinGWBin;$env:PATH"
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    & $VenvPython -m pip install cmake
    $cmakePath = Join-Path $ProjectRoot ".venv\Scripts\cmake.exe"
} else {
    $cmakePath = $cmake.Source
}

if (-not (Test-Path (Join-Path $SourceDir "CMakeLists.txt"))) {
    throw "VibeASR.cpp source is missing at $SourceDir. Re-run with -Bootstrap or clone the official repository there, including submodules."
}

$substDrive = ""
$buildSourceDir = $SourceDir
$buildMinGWBin = $MinGWBin
if ($ProjectRoot -match "\s") {
    foreach ($letter in @("V", "W", "X", "Y", "Z")) {
        if (-not (Get-PSDrive -Name $letter -ErrorAction SilentlyContinue)) {
            $candidate = "$letter`:"
            & subst $candidate $ProjectRoot
            if ($LASTEXITCODE -eq 0) {
                $substDrive = $candidate
                $projectPrefix = $ProjectRoot.TrimEnd("\") + "\"
                if (-not $SourceDir.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                    & subst $substDrive /D
                    throw "VibeASR source must be inside the project when its path contains spaces."
                }
                $sourceRelative = $SourceDir.Substring($projectPrefix.Length)
                $buildSourceDir = Join-Path "$substDrive\" $sourceRelative
                if ($MinGWBin.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                    $mingwRelative = $MinGWBin.Substring($projectPrefix.Length)
                    $buildMinGWBin = Join-Path "$substDrive\" $mingwRelative
                } elseif ($MinGWBin -match "\s") {
                    & subst $substDrive /D
                    throw "MinGW must be installed in a path without spaces. Re-run with -Bootstrap."
                }
                break
            }
        }
    }
    if (-not $substDrive) {
        throw "Could not allocate a temporary drive alias required for MinGW paths containing spaces."
    }
}

$buildDir = Join-Path $buildSourceDir "build-otter"
$gccPath = Join-Path $buildMinGWBin "gcc.exe"
$gxxPath = Join-Path $buildMinGWBin "g++.exe"
$makePath = Join-Path $buildMinGWBin "mingw32-make.exe"
$buildExitCode = 1
try {
    & $cmakePath -S $buildSourceDir -B $buildDir -G "MinGW Makefiles" `
        -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_C_COMPILER=$gccPath" `
        "-DCMAKE_CXX_COMPILER=$gxxPath" `
        "-DCMAKE_MAKE_PROGRAM=$makePath"
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -eq 0) {
        & $cmakePath --build $buildDir --target asr_infer -j 4
        $buildExitCode = $LASTEXITCODE
    }
} finally {
    if ($substDrive) {
        & subst $substDrive /D
    }
}
if ($buildExitCode -eq 0) {
    $runtimeBin = Join-Path $SourceDir "build-otter\bin"
    foreach ($dllName in @("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "libgomp-1.dll")) {
        Copy-Item -LiteralPath (Join-Path $MinGWBin $dllName) -Destination $runtimeBin -Force
    }
    Write-Host "VibeASR.cpp runtime ready at $runtimeBin" -ForegroundColor Green
}
exit $buildExitCode
