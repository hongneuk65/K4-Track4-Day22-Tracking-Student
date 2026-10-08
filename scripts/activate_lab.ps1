# Kích hoạt môi trường và đặt biến dữ liệu cho phiên PowerShell hiện tại.
$taskCondaHook = Join-Path $env:USERPROFILE 'miniconda3\shell\condabin\conda-hook.ps1'
if (-not (Test-Path -LiteralPath $taskCondaHook)) {
    throw 'Không thấy Miniconda trong thư mục người dùng. Xem HUONG_DAN.md.'
}
. $taskCondaHook
conda activate cv_robotics_lab21
if ($LASTEXITCODE -ne 0) {
    throw 'Không kích hoạt được môi trường cv_robotics_lab21.'
}
$env:PYTHONUTF8 = '1'
$taskRepoRoot = Split-Path -Parent $PSScriptRoot
$taskDataRoot = Join-Path (Split-Path -Parent $taskRepoRoot) 'data_lab21'
if (-not $env:LAB_DATA -and (Test-Path -LiteralPath (Join-Path $taskDataRoot 'video_1\img1'))) {
    $env:LAB_DATA = $taskDataRoot
}
Write-Host "Đã kích hoạt cv_robotics_lab21. LAB_DATA=$env:LAB_DATA"
