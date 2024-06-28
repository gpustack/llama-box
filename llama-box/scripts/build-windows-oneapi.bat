set DIR=%1
set ARCH=%2

@call "%ONEAPI_ROOT%\setvars.bat" intel64 --force
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)

if "%ARCH%"=="amd64" (
    cmake -G "Ninja" -S %DIR% -B %DIR%\build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DLLAMA_SYCL=on -DLLAMA_SYCL_F16=on -DLLAMA_NATIVE=off -DLLAMA_OPENMP=off
) else (
    cmake -G "Ninja" -S %DIR% -B %DIR%\build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DLLAMA_SYCL=on -DLLAMA_SYCL_F16=on -DLLAMA_NATIVE=on -DLLAMA_OPENMP=off
)
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)

cmake --build %DIR%\build --target llama-box --config Release
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)
