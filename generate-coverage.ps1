param(
    [Parameter(Mandatory = $true)]
    [string]$InstrProfile
)

$llvmCovPath = "$env:USERPROFILE\.rustup\toolchains\nightly-x86_64-pc-windows-msvc\lib\rustlib\x86_64-pc-windows-msvc\bin\llvm-cov.exe"
$targetExe = ".\target\x86_64-pc-windows-msvc\coverage\x86_64-pc-windows-msvc\release\fuzz_sieve.exe"
$outputFile = "index.html"
$covData = ".\fuzz\coverage\$InstrProfile\coverage.profdata"

& $llvmCovPath show -format=html -instr-profile $covData -Xdemangler=rustfilt $targetExe -ignore-filename-regex=".*cargo\\registry" > $outputFile

Write-Host "Coverage report generated: $outputFile"