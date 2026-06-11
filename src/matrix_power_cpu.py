import numpy as np

def matrix_power_cpu(A, power):
    result = np.eye(A.shape[0], dtype=np.float32)
    while power > 0:
        if power % 2 == 1:
            result = np.matmul(result, A)
        A = np.matmul(A, A)
        power //= 2
    return result
