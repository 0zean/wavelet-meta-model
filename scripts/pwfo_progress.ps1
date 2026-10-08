<#
Progress of a running PWFO spec (U11 Stage C): one line per started cell with windows walked / expected over its
combos, plus the launcher status and the spec's ledger rows. Read-only.

    powershell -ExecutionPolicy Bypass -File scripts\pwfo_progress.ps1            # u11_c
    powershell -ExecutionPolicy Bypass -File scripts\pwfo_progress.ps1 -Spec u11_c -Cells 14
#>
param([string]$Spec = "u11_c", [int]$Cells = 14)
$repo = Split-Path -Parent $PSScriptRoot
$exp = Join-Path $repo "results\experiments"

# windows per combo over the 2016-01 -> 2025-10 development range (2,450 sessions; feasibility probe 2026-10-06)
$expected = @{
    252 = @{ 5 = 439; 10 = 219; 21 = 104; 63 = 34 }; 378 = @{ 5 = 414; 10 = 207; 21 = 98; 63 = 32 }
    504 = @{ 5 = 389; 10 = 194; 21 = 92; 63 = 30 }; 756 = @{ 5 = 338; 10 = 169; 21 = 80; 63 = 26 }
    1260 = @{ 5 = 238; 10 = 119; 21 = 56; 63 = 18 }; 1512 = @{ 5 = 187; 10 = 93; 21 = 44; 63 = 14 }
}

Get-Content (Join-Path $exp "run_specs.log") -Tail 3
$rows = (Select-String -Path (Join-Path $repo "results\ledger.jsonl") -Pattern "`"spec_name`": `"$Spec`"" -SimpleMatch).Count
"ledger rows for ${Spec}: $rows / $Cells (a row appears only when a whole cell finishes)"
$out = Get-ChildItem $exp -Filter "$Spec.*.out.log" | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $out) { "no $Spec run log yet"; exit 0 }
$since = $out.CreationTime.AddMinutes(-1)

Get-ChildItem (Join-Path $exp "cells") -Directory | ForEach-Object {
    $logs = Get-ChildItem (Join-Path $_.FullName "logs") -Filter "IS*_OOS*.log" -ErrorAction SilentlyContinue |
        Where-Object CreationTime -gt $since
    if (-not $logs) { return }
    $s = (Get-Content (Join-Path $_.FullName "spec.json") -Raw | ConvertFrom-Json).spec
    $walked = 0; $total = 0; $combosDone = 0
    foreach ($l in $logs) {
        if ($l.BaseName -notmatch '^IS(\d+)_OOS(\d+)$') { continue }
        $n = $expected[[int]$Matches[1]][[int]$Matches[2]]
        $total += $n
        if (Select-String -Path $l.FullName -Pattern '\[PWFO\].*windows, status' -Quiet) { $walked += $n; $combosDone++; continue }
        $last = Select-String -Path $l.FullName -Pattern 'window (\d+):' | Select-Object -Last 1
        if ($last) { $walked += [int]$last.Matches[0].Groups[1].Value }
    }
    $finished = Test-Path (Join-Path $_.FullName "daily_returns.csv")
    [pscustomobject]@{
        cell     = "$($s.symbols[0]) $($s.timeframe) $($s.primary.name) $($s.model.meta) $($s.sizer)"
        windows  = "$walked / $total"
        pct      = [math]::Round(100 * $walked / [math]::Max($total, 1))
        combos   = "$combosDone / $($logs.Count) done"
        state    = $(if ($finished) { "finished" } else { "running" })
        updated  = ($logs | Measure-Object LastWriteTime -Maximum).Maximum.ToString("HH:mm")
        hash     = $_.Name
    }
} | Sort-Object state, pct -Descending | Format-Table -AutoSize
"python processes: $((Get-Process python -ErrorAction SilentlyContinue).Count) (2 are the hermes gateway)"
