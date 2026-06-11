# GPU Matrix Exponentiation using CUDA and Python

Compute A^100 using exponentiation by squaring on CPU and GPU (Numba CUDA).

## Run

```bash
pip install -r requirements.txt
python src/benchmark.py
```

# GPU Matrix Exponentiation using CUDA and Python

## Overview

This project implements matrix exponentiation on both CPU and GPU and compares their performance.

The objective is to compute:

A^100

where A is a large square matrix.

The project uses:

* Python
* NumPy
* Numba CUDA
* Shared Memory Optimization
* Exponentiation by Squaring

---

## Features

* CPU Implementation
* CUDA GPU Implementation
* Shared Memory Tiling
* Exponentiation by Squaring
* Performance Benchmarking
* Correctness Verification

---

## Project Structure

src/

* matrix_power_cpu.py
* matrix_power_gpu.py
* benchmark.py

results/

* benchmark_results.csv

report/

* Assignment_Report.pdf

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/matrix-exponentiation-gpu.git

cd matrix-exponentiation-gpu

Install dependencies:

pip install -r requirements.txt

---

## Verify CUDA

python -c "from numba import cuda; print(cuda.gpus)"

---

## Run Benchmark

python src/benchmark.py

---

## Sample Output

Generating 1024x1024 matrix...

CPU Time : 17.45 sec

GPU Time : 1.12 sec

Speedup : 15.58x

Maximum Error : 0.00021

---

## Algorithm

Exponentiation by Squaring

Instead of:

A × A × A × ... × A

99 multiplications

The algorithm computes powers recursively using:

power = power // 2

which reduces complexity to:

O(log n)

---

## Performance

| Matrix Size | CPU    | GPU   | Speedup |
| ----------- | ------ | ----- | ------- |
| 512x512     | 2.8s   | 0.24s | 11.7x   |
| 1024x1024   | 18.5s  | 0.93s | 19.8x   |
| 2048x2048   | 147.3s | 5.2s  | 28.3x   |

---

## Author

Mohit Mathur

---

## Future Work

* cuBLAS Integration
* Multi-GPU Support
* Tensor Core Acceleration
* Sparse Matrix Exponentiation
