@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
set "REPO_DIR=%~dp0.."
set "TARGET_DIR=%REPO_DIR%\engines\llama.cpp"

if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

echo ============================================================================
echo  Moris Roleplay - llama.cpp 本地引擎获取助手
echo ============================================================================
echo 目标引擎库目录: "%TARGET_DIR%"
echo.
echo 正在查询 bml/ggml-org llama.cpp 最新 Windows (x64 CPU/AVX2) 预编译包...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop'; " ^
  "$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers @{'User-Agent'='MorisRoleplay'}; " ^
  "$asset = $release.assets | Where-Object { $_.name -match 'bin-win-(avx2|cpu)-x64\.zip$' } | Select-Object -First 1; " ^
  "if (-not $asset) { Write-Host '[提示] 未匹配到自动下载资产，请手动前往 https://github.com/ggml-org/llama.cpp/releases 下载并解压至 engines/llama.cpp/' -ForegroundColor Yellow; exit 1 }; " ^
  "$zipPath = Join-Path '%TARGET_DIR%' $asset.name; " ^
  "Write-Host ('正在下载: ' + $asset.name + ' ...'); " ^
  "Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath; " ^
  "Write-Host '正在解压到 engines/llama.cpp/ ...'; " ^
  "Expand-Archive -Path $zipPath -DestinationPath '%TARGET_DIR%' -Force; " ^
  "Remove-Item $zipPath -Force; " ^
  "Write-Host '[完成] llama.cpp 已就绪！请在 Web 界面点击 [刷新库] 选择引擎。' -ForegroundColor Green"

exit /b %ERRORLEVEL%
