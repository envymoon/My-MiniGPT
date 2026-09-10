param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [string]$MonitorLogPath
)

while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
    if (Test-Path -LiteralPath $LogPath) {
        $tail = Get-Content -LiteralPath $LogPath -Tail 40 -ErrorAction SilentlyContinue
        if ($tail -match "starting training; metrics are also written") {
            Add-Content -LiteralPath $MonitorLogPath -Value "$(Get-Date -Format o) data preparation finished; stopping process tree PID=$TargetPid"
            taskkill.exe /PID $TargetPid /T /F | Out-Null
            break
        }
    }
    Start-Sleep -Seconds 15
}
