import numpy as np
from sympy import factorint

def _valuation(n: int, p: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e

def _egcd(a: int, b: int):
    if b == 0:
        return (1, 0, a)
    x1, y1, g = _egcd(b, a % b)
    return (y1, x1 - (a // b) * y1, g)

def _inv_mod(a: int, m: int) -> int:
    a %= m
    try:
        return pow(a, -1, m)
    except ValueError:
        x, y, g = _egcd(a, m)
        if g != 1:
            raise ZeroDivisionError(f"{a} has no inverse mod {m}")
        return x % m

def _crt_pair(a1: int, m1: int, a2: int, m2: int):
    inv = _inv_mod(m1 % m2, m2)
    t = ((a2 - a1) % m2) * inv % m2
    x = a1 + m1 * t
    return x, m1 * m2

def factor_as_2a3_3b(n: int):
    if n <= 0:
        raise ValueError("n must be positive")
    e2 = _valuation(n, 2)
    m  = n >> e2
    e3 = _valuation(m, 3)
    leftover = m // (3**e3)
    if leftover != 1 or e2 < 3:
        raise ValueError("n must be of the form 2^(a+3)*3^b")
    return e2 - 3, e3

def solve_two_crts(n: int):
    a, b = factor_as_2a3_3b(n)
    m1 = 2**(a+3)
    m2 = 3**(b+1)
    x, M = _crt_pair(5, m1, 1, m2)
    y, _ = _crt_pair(1, m1, 2, m2)
    return a, b, x % M, y % M, M

def exponent_matrix_mul(n: int):
    a, b, x, y, M = solve_two_crts(n)
    rows = 2**(a+1)
    cols = 2 * (3**b)
    mat = []
    for i in range(rows):
        row = []
        xi = pow(x, i, M)
        for j in range(cols):
            val = (xi * pow(y, j, M)) % M
            row.append(val)
        mat.append(row)
    return mat, M

def exponent_matrix_mul_transposed(n: int):
    a, b, x, y, M = solve_two_crts(n)
    rows = 2 * (3**b)
    cols = 2**(a+1)
    matT = []
    y_pows = [pow(y, j, M) for j in range(rows)]
    x_pows = [pow(x, i, M) for i in range(cols)]
    for j in range(rows):
        row = []
        yj = y_pows[j]
        for i in range(cols):
            row.append((x_pows[i] * yj) % M)
        matT.append(row)
    return matT, M

def flatten_rows_to_vector(mat):
    vals = []
    for row in mat:
        vals.extend(row)
    return vals

def exponent_vector_mul(n: int):
    mat, M = exponent_matrix_mul(n)
    return flatten_rows_to_vector(mat), M

def exponent_vector_mul_transposed(n: int):
    matT, M = exponent_matrix_mul_transposed(n)
    return flatten_rows_to_vector(matT), M

def naive_dft24(coeffs):
    index, N = exponent_vector_mul(24)
    out = np.zeros(12, dtype=np.complex128)
    for i in range(12):
        for j in range(len(coeffs)):
            out[i] += coeffs[j] * np.exp(-2j * np.pi * index[i] * j / N)
    return out

def fft_input_permutation(n: int):
    factors = factorint(n)
    if not set(factors).issubset({2, 3}):
        raise ValueError("n must be of the form 2*3^a*2^b")
    if factors.get(2, 0) < 1:
        raise ValueError("n must be even")
    a = factors.get(3, 0)
    b = factors.get(2, 0) - 1
    index = [0] * n
    length = 1
    index[0] = 0
    shift = n >> 1
    for i in range(length):
        index[length + i] = index[i] + shift
    length *= 2
    for _ in range(a):
        shift //= 3
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len
        for i in range(base_len):
            index[length + i] = index[i] + 2*shift
        length += base_len
    for _ in range(b):
        shift //= 2
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len
    return index

def inverse_permutation(perm):
    inv = [0]*len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

index, M = exponent_vector_mul(24);
w = np.exp(-2j * np.pi / M)
W12 = w**index[0:6]
W6 = W12**2

A2 = w**np.array([4, 44])
B2 = A2*w**np.array([48, 24])
C2 = A2*w**np.array([24, 48])

A2_sq = A2**2
B2_sq = B2**2
C2_sq = C2**2

z = w**12
zminusz5inv = 1.0 / (z - z**5)

def dft2(coeffs):
    T = z * coeffs[1]
    coeffs[1] = coeffs[0] + coeffs[1] - T
    coeffs[0] = coeffs[0] + T

def idft2(coeffs):
    T1 = coeffs[0] + coeffs[1]
    T2 = (coeffs[0] - coeffs[1]) * zminusz5inv
    coeffs[0] = (T1 - T2) / 2.0
    coeffs[1] = T2

def dft6(coeffs):
    dft2(coeffs[0:2])
    dft2(coeffs[2:4])
    dft2(coeffs[4:6])

    T1 = coeffs[0:2] + A2 * coeffs[2:4] + A2_sq * coeffs[4:6]
    T2 = coeffs[0:2] + B2 * coeffs[2:4] + B2_sq * coeffs[4:6]
    T3 = coeffs[0:2] + C2 * coeffs[2:4] + C2_sq * coeffs[4:6]

    coeffs[0:2] = T1
    coeffs[2:4] = T2
    coeffs[4:6] = T3

def dft12(coeffs, verbose=False):
    dft6(coeffs[0:6])
    dft6(coeffs[6:12])
    T = W6 * coeffs[6:12]
    coeffs[0:6] = coeffs[0:6] + T

def dft24(coeffs, verbose=False):
    dft12(coeffs[0:12])
    dft12(coeffs[12:24])
    T = W12 * coeffs[12:18]
    coeffs[6:12] = coeffs[0:6] - T
    coeffs[0:6]  = coeffs[0:6] + T

def compare_dft24(verbose=True):
    np.random.seed(42)
    x24 = (np.random.randint(-5, 6, size=24) + 1j*np.random.randint(-5, 6, size=24)).astype(np.complex128)
    y_naive = naive_dft24(x24)
    perm = fft_input_permutation(24)
    x24_perm = x24[perm].copy()
    dft24(x24_perm)
    diff = x24_perm[0:12] - y_naive
    max_abs_diff = np.max(np.abs(diff))
    print("\n=== Comparison (dft8 vs naive_dft8) ===")
    if verbose:
        print("Input x24:         ", x24)
        print("Permutation idx:  ", perm)
    print("radix dft24 output: ", x24_perm[0:12])
    print("naive dft24 output: ", y_naive)
    print("max |diff|:        ", max_abs_diff)

if __name__ == "__main__":
    compare_dft24(verbose=True)
