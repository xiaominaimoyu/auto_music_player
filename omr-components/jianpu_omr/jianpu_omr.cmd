@echo off
setlocal
set "NODE=%~dp0node.exe"
if not exist "%NODE%" if exist "%~dp0runtime\node.exe" set "NODE=%~dp0runtime\node.exe"
if not exist "%NODE%" (
  echo [AutoMusicPlayer] bundled Node runtime is missing 1>&2
  exit /b 2
)
"%NODE%" "%~dp0amp_worker.mjs" %*
exit /b %ERRORLEVEL%
