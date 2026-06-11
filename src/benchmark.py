import time
import numpy as np
from matrix_power_cpu import matrix_power_cpu
from matrix_power_gpu import matrix_power_gpu

N = 1024
POWER = 100

A = np.random.rand(N, N).astype(np.float32)

start = time.perf_counter()
cpu_result = matrix_power_cpu(A.copy(), POWER)
cpu_time = time.perf_counter() - start

start = time.perf_counter()
gpu_result = matrix_power_gpu(A.copy(), POWER)
gpu_time = time.perf_counter() - start

print(f'CPU Time: {cpu_time:.4f}s')
print(f'GPU Time: {gpu_time:.4f}s')
print(f'Speedup : {cpu_time/gpu_time:.2f}x')
print(f'Max Error: {np.max(np.abs(cpu_result-gpu_result))}')
