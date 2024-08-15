set DIR=%1
set ARCH=%2

@call "%ONEAPI_ROOT%\setvars.bat" intel64 --force
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)

if "%ARCH%"=="amd64" (
    cmake -G "Ninja" -S %DIR% -B %DIR%\build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DGGML_SYCL=on -DGGML_SYCL_F16=on -DGGML_NATIVE=off -DGGML_OPENMP=off -DGGML_RPC=on
) else (
    cmake -G "Ninja" -S %DIR% -B %DIR%\build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DGGML_SYCL=on -DGGML_SYCL_F16=on -DGGML_NATIVE=on -DGGML_OPENMP=off -DGGML_RPC=on
)
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)

cmake --build %DIR%\build --target llama-box --config Release
if %ERRORLEVEL% neq 0 (exit /B %ERRORLEVEL%)
