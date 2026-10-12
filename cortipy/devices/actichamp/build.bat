@echo off
rem Build EEG_SharedMemoryProducer.exe with the Microsoft Visual C++ compiler (x64).
rem
rem Requires the BrainVision Amplifier SDK from Brain Products (not part of this
rem repository, see README). It is expected in "C:\Vision\Amplifier SDK". If it is
rem installed somewhere else, set the variable AMPLIFIER_SDK to that folder first:
rem     set "AMPLIFIER_SDK=D:\My SDKs\Amplifier SDK"
rem
rem Usage:  build.bat            builds into this folder (overwrites the .exe)
rem         build.bat <outdir>   builds into <outdir> (keeps the shipped .exe)
setlocal
cd /d "%~dp0"

set "OUTDIR=%~1"
if "%OUTDIR%"=="" set "OUTDIR=%~dp0."
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

rem Locate the Amplifier SDK.
if not defined AMPLIFIER_SDK set "AMPLIFIER_SDK=C:\Vision\Amplifier SDK"
if not exist "%AMPLIFIER_SDK%\Include\SDK.h" goto nosdk
if not exist "%AMPLIFIER_SDK%\Samples\C++\RawDataHandlerExample.h" goto nosdk
if not exist "%AMPLIFIER_SDK%\Bin\x64\AmplifierSDK.lib" goto nosdk

rem Already inside an "x64 Native Tools Command Prompt"? Then cl.exe is on the PATH.
where cl >nul 2>nul
if not errorlevel 1 goto compile

rem Otherwise locate Visual Studio and load its x64 build environment.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto novs
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto novs
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
where cl >nul 2>nul
if errorlevel 1 goto novs

:compile
rem The producer must not be running while the .exe is replaced.
cl /nologo /EHsc /O2 /MT /I"%AMPLIFIER_SDK%\Include" /I"%AMPLIFIER_SDK%\Samples\C++" EEG_SharedMemoryProducer.cpp "%AMPLIFIER_SDK%\Bin\x64\AmplifierSDK.lib" /Fo"%OUTDIR%\\" /Fe"%OUTDIR%\EEG_SharedMemoryProducer.exe"
if errorlevel 1 goto failed
echo Built "%OUTDIR%\EEG_SharedMemoryProducer.exe"
exit /b 0

:nosdk
echo ERROR: BrainVision Amplifier SDK not found in "%AMPLIFIER_SDK%".
echo Download and install it from https://www.brainproducts.com/downloads/more-software/
echo or set AMPLIFIER_SDK to the folder that contains Include, Bin and Samples.
exit /b 1

:novs
echo ERROR: Visual Studio with the workload "Desktop development with C++" was not found.
exit /b 1

:failed
echo BUILD FAILED
exit /b 1
