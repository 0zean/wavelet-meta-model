<#
Run experiment specs one after another, detached, on Windows (U11).

    powershell -ExecutionPolicy Bypass -File scripts\run_specs.ps1 experiments\specs\u11_a1.yaml experiments\specs\u11_a_screen.yaml

- Detaches itself (hidden) and returns at once; progress: results\experiments\run_specs.log, each run's output in
  results\experiments\<spec>.<timestamp>.out.log / .err.log, per-symbol fits in results\experiments\signals\*.log.
- Windows 11 puts a windowless background process under EcoQoS power throttling, which confines it to the E-cores
  (on the i9-14900KF: 16 E-cores busy, 8 P-cores idle). Every minute this opts the run's python processes out
  (SetProcessInformation / ProcessPowerThrottling, per process, nothing persists).
- LOKY_MAX_CPU_COUNT=1: rf_ldp's n_jobs=-1 then fits single-threaded inside each cell process.
- A spec whose run exits non-zero stops the chain. Stop everything: taskkill /T /F /PID <pid in run_specs.log>;
  finished per-symbol fits are kept and skipped on the next run.
#>
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Specs,
    [int]$Jobs = 30,
    [string]$Ledger = "",  # default: results\ledger.jsonl
    [string]$Root = "",  # default: results\experiments
    [switch]$Attached
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repo "results\experiments"
$status = Join-Path $logDir "run_specs.log"

if (-not $Attached) {
    $args_ = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Jobs", $Jobs, "-Attached")
    if ($Ledger) { $args_ += @("-Ledger", "`"$Ledger`"") }
    if ($Root) { $args_ += @("-Root", "`"$Root`"") }
    $args_ += $Specs | ForEach-Object { "`"$_`"" }
    $p = Start-Process powershell -ArgumentList $args_ -WindowStyle Hidden -PassThru
    "run_specs started (pid $($p.Id)); progress in $status"
    exit 0
}

Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class Unthrottle {
  [StructLayout(LayoutKind.Sequential)] struct PPTS { public uint Version, ControlMask, StateMask; }
  [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint a, bool i, int pid);
  [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetProcessInformation(IntPtr h, int c, ref PPTS s, int n);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
  public static bool Apply(int pid) {
    IntPtr h = OpenProcess(0x0200, false, pid); if (h == IntPtr.Zero) return false;
    PPTS s = new PPTS { Version = 1, ControlMask = 1, StateMask = 0 };  // execution speed: never throttle
    bool ok = SetProcessInformation(h, 4, ref s, Marshal.SizeOf(typeof(PPTS))); CloseHandle(h); return ok;
  }
}
"@

function Log($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content -Path $status -Encoding utf8 }

$env:LOKY_MAX_CPU_COUNT = "1"
$env:PYTHONIOENCODING = "utf-8"
$global_ = @()
if ($Ledger) { $global_ += @("--ledger", "`"$Ledger`"") }
if ($Root) { $global_ += @("--root", "`"$Root`"") }
Log "launcher pid $($PID): $($Specs -join ', ') with $Jobs jobs"
foreach ($spec in $Specs) {
    $stem = "$([IO.Path]::GetFileNameWithoutExtension($spec)).$(Get-Date -Format 'yyyyMMdd-HHmm')"
    $argv = @("run", "python", "-m", "experiments") + $global_ + @("run", "`"$spec`"", "--jobs", $Jobs)
    $run = Start-Process uv -ArgumentList $argv `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "$stem.out.log") -RedirectStandardError (Join-Path $logDir "$stem.err.log")
    $null = $run.Handle  # keeps ExitCode readable after the process exits
    Log "started $spec (uv pid $($run.Id)) -> $stem.out.log"
    $seen = @{}
    while (-not $run.HasExited) {
        Get-CimInstance Win32_Process -Filter "name='python.exe'" |
            Where-Object { ($_.CommandLine -like '*-m experiments*' -or $_.CommandLine -like '*joblib*') -and -not $seen.ContainsKey($_.ProcessId) } |
            ForEach-Object { $seen[$_.ProcessId] = [Unthrottle]::Apply($_.ProcessId) }
        Start-Sleep 60
    }
    $code = $run.ExitCode
    Log "finished $spec with exit code $code ($($seen.Count) processes unthrottled)"
    if ($code -ne 0) { Log "stopping the chain"; exit $code }
}
Log "all specs done"
