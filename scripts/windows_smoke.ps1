# Windows smoke test for an installed anygrep (Windows PowerShell 5.1 or PowerShell 7).
#
#   python -m venv venv; venv\Scripts\pip install anygrep
#   $env:PATH = "$PWD\venv\Scripts;$env:PATH"
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows_smoke.ps1 -WorkDir C:\anygrep-smoke
#
# Generates the demo corpus under a folder whose name has a space and Japanese
# characters, indexes it into a separate index file, and checks searches,
# filters, JSON Lines, interactive input from a pipe, output encoding through
# PowerShell pipes, exit codes, incremental updates, a dry run, a background
# index job, remove and reset. The model is
# downloaded on first use. Runs at low priority with 2 threads.
# This file must stay ASCII-only (Windows PowerShell reads it as the ANSI code page).

param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [string]$Repo = '',
    [int]$Threads = 2
)

$ErrorActionPreference = 'Continue'
# Windows PowerShell 5.1 leaves $PSScriptRoot empty in param defaults.
if (-not $Repo) { $Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:ANYGREP_THREADS = "$Threads"
$fails = 0
function Check([string]$name, [bool]$ok, [string]$detail = '') {
    if ($ok) { Write-Host "PASS  $name" } else { Write-Host "FAIL  $name  $detail"; $script:fails++ }
}
function Score($hits, [string]$pattern) {
    $m = $hits | Where-Object { $_.path -match $pattern } | Select-Object -First 1
    if ($m) { [double]$m.score } else { -1 }
}
function JsonLines($text) { $text | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json } }

$memo = [string][char]0x30E1 + [char]0x30E2           # "memo" in katakana
$C = Join-Path $WorkDir ("demo corpus\" + $memo)
$env:ANYGREP_DB = Join-Path $WorkDir 'smoke-index.sqlite'
New-Item -ItemType Directory -Force $WorkDir | Out-Null
Remove-Item -Recurse -Force (Join-Path $WorkDir 'demo corpus') -ErrorAction SilentlyContinue
Remove-Item "$($env:ANYGREP_DB)*" -ErrorAction SilentlyContinue

python (Join-Path $Repo 'scripts\make_demo_corpus.py') $C | Out-Null
Check 'demo corpus generated' ((Get-ChildItem -Recurse -File $C).Count -eq 23)

anygrep index $C --threads $Threads -q
Check 'index' ($LASTEXITCODE -eq 0)

# Capture output as UTF-8 so the score checks do not depend on the console code page.
$prevEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$q1 = JsonLines (anygrep 'when is the tax return due' $C -m 3 --json)
$q2 = JsonLines (anygrep 'running out of disk space' $C -m 3 --json)
[Console]::OutputEncoding = $prevEnc
foreach ($t in @(@($q1, 'tax-checklist\.md', 0.732), @($q1, 'kakutei-shinkoku\.md', 0.691), @($q2, 'error-dialog\.png', 0.767))) {
    $s = Score $t[0] $t[1]
    Check "score $($t[1]) ~ $($t[2])" ([math]::Abs($s - $t[2]) -le 0.01) "got $s"
}
Check 'Japanese path in output' ($q1[0].path.Contains($memo))

# Plain PowerShell pipe with the default console code page: no mojibake.
$piped = anygrep 'when is the tax return due' $C -m 2 | Out-String
Check 'PowerShell pipe keeps Japanese' ($piped.Contains($memo)) ($piped -split "`n" | Select-Object -Last 3)
Check 'pipe format PATH:LINE:SNIPPET' ($piped -match 'tax-checklist\.md:\d+:') $piped

$json = JsonLines (anygrep 'when is the tax return due' $C -m 2 --json)
Check '--json lines parse, Japanese path intact' ($json[0].path.Contains($memo) -and $json.Count -eq 2)

$files = anygrep -l 'when is the tax return due' $C -m 2 | Out-String
Check '-l' ($files -match 'tax-checklist\.md' -and -not ($files -match ':\d+:'))

$pdf = anygrep 'wifi password reset' $C -t pdf -m 1 | Out-String
Check '-t pdf' ($pdf -match 'home-network-manual\.pdf:1:\[p\.')
foreach ($g in @("$C\notes\finance", 'notes/finance/*', 'notes\finance\*')) {
    $r = anygrep 'tax' $C -g $g -m 5 | Out-String
    Check "-g $g" (($r -match 'tax-checklist') -and -not ($r -match 'pdf|png'))
}
$img = anygrep --image "$C\photos\red-bicycle.jpg" $C -m 3 --weak | Out-String
Check '--image' ($LASTEXITCODE -eq 0 -and $img -match ':1:\[image\]')

$prevOut = $OutputEncoding
$OutputEncoding = [Text.Encoding]::UTF8                  # adds a BOM to piped input
$inter = "when is the tax return due`n`n" | anygrep -Q $C -m 1 | Out-String
$OutputEncoding = $prevOut
Check '-Q with piped UTF-8 input' ($LASTEXITCODE -eq 0 -and $inter -match 'tax-checklist') $inter

anygrep 'qwzx plorb fnord' $C 2>$null | Out-Null
Check 'no match exits 1' ($LASTEXITCODE -eq 1) "got $LASTEXITCODE"
anygrep how many vacation days 2>$null | Out-Null
Check 'missing PATH exits 2' ($LASTEXITCODE -eq 2) "got $LASTEXITCODE"
anygrep 'when is the tax return due' $C -n -m 1 | Out-Null
Check '-n is accepted' ($LASTEXITCODE -eq 0) "got $LASTEXITCODE"

$st = anygrep status --json | Out-String | ConvertFrom-Json
Check 'status --json' ($st.counts.text.ok -eq 12 -and $st.roots[0].Contains($memo))

Move-Item "$C\notes\sourdough.md" "$C\notes\recipes\"        # moved: index rows reused
Remove-Item "$C\notes\dev\docker-cheatsheet.md"
Set-Content "$C\notes\vet.txt" 'The cat needs a vet appointment for vaccinations in March'
$inc = anygrep index $C --threads $Threads 2>&1 | Out-String
Check 'incremental index' ($inc -match '1 new, 0 updated, 21 unchanged, 1 removed, 1 moved') ($inc -split "`n" | Select-String 'Done')
$r = anygrep 'vet appointment for the cat' $C -m 1 | Out-String
Check 'new file found' ($r -match 'vet\.txt')

anygrep remove ($C.ToUpper() + '\NOTES\FINANCE') | Out-Null
$st = anygrep status --json | Out-String | ConvertFrom-Json
Check 'remove (other letter case)' ($st.counts.text.ok -eq 10)

# Large folders: dry run, then a background job that is waited for.
$B = Join-Path $WorkDir 'bg notes'
Remove-Item -Recurse -Force $B -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $B | Out-Null
1..3 | ForEach-Object { Set-Content (Join-Path $B "note$_.txt") "Background note $_ about watering the garden plants" }
$dry = anygrep index --dry-run $B 2>&1 | Out-String
Check 'index --dry-run' ($LASTEXITCODE -eq 0 -and $dry -match '3 new or changed files') $dry
anygrep index --background $B --threads $Threads 2>&1 | Out-Null
Check 'index --background starts' ($LASTEXITCODE -eq 0) "got $LASTEXITCODE"
$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep 2
    $st = anygrep status --json | Out-String | ConvertFrom-Json
} while ($st.job -and (Get-Date) -lt $deadline)
Check 'background job finished' (-not $st.job)
$r = anygrep 'watering the garden' $B -m 1 --no-index | Out-String
Check 'background job indexed' ($r -match 'note\d\.txt') $r
anygrep index --stop 2>&1 | Out-Null
Check 'index --stop without a job' ($LASTEXITCODE -eq 0) "got $LASTEXITCODE"

# reset while another process has the index open fails cleanly, then works.
$py = (Get-Command python).Source
$holder = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList '-c', "`"import sqlite3,time;c=sqlite3.connect(r'$($env:ANYGREP_DB)');c.execute('select 1');time.sleep(30)`""
Start-Sleep 2
$errFile = Join-Path $WorkDir 'reset-stderr.txt'
anygrep reset --yes 2> $errFile
$code = $LASTEXITCODE
$err = (Get-Content -Raw $errFile) -replace '\s', ''   # PowerShell 5.1 wraps stderr lines
Check 'reset while index is open' ($code -eq 2 -and $err -match 'Closeotheranygrep') $err
Stop-Process -Id $holder.Id -Force
Start-Sleep 1
anygrep reset --yes | Out-Null
Check 'reset' ($LASTEXITCODE -eq 0 -and -not (Test-Path $env:ANYGREP_DB))

Write-Host ""
if ($fails) { Write-Host "$fails check(s) failed"; exit 1 } else { Write-Host 'All checks passed' }
