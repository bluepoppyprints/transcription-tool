param(
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7877,
    [int]$Workers = 8,
    [switch]$NoBrowser,
    [int]$BrowserTimeoutSeconds = 20,
    [Alias("BrowserUrl", "OpenUrl")]
    [string]$LaunchUrl = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not available in PATH."
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg was not found in PATH. Transcription will fail until ffmpeg is installed."
}

# Accept GNU-style arguments as well (e.g. --host, --port, --workers).
if ($ExtraArgs) {
    for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
        $arg = $ExtraArgs[$i]
        if (-not $arg) { continue }
        $next = if ($i + 1 -lt $ExtraArgs.Count) { $ExtraArgs[$i + 1] } else { $null }

        switch -Regex ($arg.ToLowerInvariant()) {
            '^--?host$' {
                if ($next) {
                    $BindHost = $next
                    $i++
                }
                continue
            }
            '^--?port$' {
                if ($next -and ($next -as [int])) {
                    $Port = [int]$next
                    $i++
                }
                continue
            }
            '^--?workers$' {
                if ($next -and ($next -as [int])) {
                    $Workers = [int]$next
                    $i++
                }
                continue
            }
            '^--?nobrowser$' {
                $NoBrowser = $true
                continue
            }
            '^--?browsertimeoutseconds$' {
                if ($next -and ($next -as [int])) {
                    $BrowserTimeoutSeconds = [int]$next
                    $i++
                }
                continue
            }
            '^--?(browserurl|openurl|launchurl)$' {
                if ($next) {
                    $LaunchUrl = $next
                    $i++
                }
                continue
            }
        }
    }
}

# Run from source in this project directory.
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if ($LaunchUrl) {
    $BrowserUrl = $LaunchUrl
} else {
    if ($BindHost -in @("0.0.0.0", "::", "[::]")) {
        $BrowserHost = "127.0.0.1"
    } else {
        $BrowserHost = $BindHost
    }
    $BrowserUrl = "http://$BrowserHost`:$Port"
}

Write-Host "Starting Otter Local Web from $ProjectRoot" -ForegroundColor Cyan
Write-Host "URL: $BrowserUrl  Workers: $Workers" -ForegroundColor Cyan

# Prevent accidentally opening an old server if the chosen port is already in use.
$existingListeners = @()
try {
    $existingListeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
} catch {
    $existingListeners = @()
}
if ($existingListeners.Count -gt 0) {
    $pids = ($existingListeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    Write-Error "Port $Port is already in use (PID: $pids). Stop the old process or choose another port."
    exit 1
}

$openBrowserJob = $null
if (-not $NoBrowser) {
    $timeout = [Math]::Max(1, [int]$BrowserTimeoutSeconds)
    $openBrowserJob = Start-Job -ScriptBlock {
        param(
            [string]$Url,
            [int]$TimeoutSeconds
        )

        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            try {
                $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                    Start-Process $Url | Out-Null
                    return
                }
            } catch {
                # Keep polling until timeout.
            }
            Start-Sleep -Milliseconds 500
        }
    } -ArgumentList $BrowserUrl, $timeout
}

$exitCode = 0
try {
    python -m otter_local.web --host $BindHost --port $Port --workers $Workers
    $exitCode = $LASTEXITCODE
} finally {
    if ($openBrowserJob) {
        if ($openBrowserJob.State -eq "Running") {
            Stop-Job $openBrowserJob -ErrorAction SilentlyContinue | Out-Null
        }
        Receive-Job $openBrowserJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $openBrowserJob -ErrorAction SilentlyContinue | Out-Null
    }
}

exit $exitCode
