import numpy as np
from numba import cuda, float32

TPB = 16

@cuda.jit
def matrix_mul_kernel(A, B, C):
    sA = cuda.shared.array((TPB, TPB), float32)
    sB = cuda.shared.array((TPB, TPB), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TPB + ty
    col = cuda.blockIdx.x * TPB + tx

    tmp = 0.0

    for m in range((A.shape[1] + TPB - 1) // TPB):
        if row < A.shape[0] and (m * TPB + tx) < A.shape[1]:
            sA[ty, tx] = A[row, m * TPB + tx]
        else:
            sA[ty, tx] = 0

        if col < B.shape[1] and (m * TPB + ty) < B.shape[0]:
            sB[ty, tx] = B[m * TPB + ty, col]
        else:
            sB[ty, tx] = 0

        cuda.syncthreads()

        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

def gpu_matmul(A, B):
    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)

    threads = (TPB, TPB)
    blocks = ((B.shape[1] + TPB - 1) // TPB,
              (A.shape[0] + TPB - 1) // TPB)

    matrix_mul_kernel[blocks, threads](A_d, B_d, C_d)
    cuda.synchronize()
    return C_d.copy_to_host()

def matrix_power_gpu(A, power):
    result = np.eye(A.shape[0], dtype=np.float32)
    while power > 0:
        if power % 2 == 1:
            result = gpu_matmul(result, A)
        A = gpu_matmul(A, A)
        power //= 2
    return result
