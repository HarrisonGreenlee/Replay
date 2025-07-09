REM Run benchmark.py
py benchmark.py || echo benchmark.py failed

REM Run benchmark_parallelism.py
py benchmark_parallelism.py || echo benchmark_parallelism.py failed

REM Run grid_benchmark.py
py grid_benchmark.py || echo grid_benchmark.py failed

echo All scripts attempted.
