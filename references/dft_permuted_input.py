import numpy as np
from sympy import factorint

def naive_dft6(coeffs):
    N = 18
    index = [1, 5, 7, 11, 13, 17]
    out = np.zeros(6, dtype=np.complex128)
    for i in range(6):
        for j in range(len(coeffs)):
            out[i] += coeffs[j] * np.exp(-2j * np.pi * index[i] * j / N)           
    return out

def naive_dft12(coeffs):
    N = 36
    index = [1, 5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35]
    out = np.zeros(12, dtype=np.complex128)
    for i in range(12):
        for j in range(len(coeffs)):
            out[i] += coeffs[j] * np.exp(-2j * np.pi * index[i] * j / N)
    return out

# ---------- Permutation ----------
def fft_input_permutation_2then3then2(n: int):
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

def fft_input_permutation_2then3(n: int):
    factors = factorint(n)
    if not set(factors).issubset({2, 3}):
        raise ValueError("n must be of the form 2 * 2^b * 3^a")
    if factors.get(2, 0) < 1:
        raise ValueError("n must be even")

    a = factors.get(3, 0)
    b = factors.get(2, 0) - 1

    index = [0] * n
    length = 1
    index[0] = 0

    shift = None
    for k in range(b + 1):
        if k == 0:
            shift = n >> 1
        else:
            shift //= 2
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len

    for _ in range(a):
        shift //= 3
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len
        for i in range(base_len):
            index[length + i] = index[i] + 2 * shift
        length += base_len

    return index

# ---------- Globals for DFT/IDFT (M=36 cyclotomic) ----------
M = 36
w = np.exp(-2j * np.pi / M)
W6 = w**np.array([1, 5, 7, 11, 13, 17])
W6_rev = w**np.array([17, 13, 11, 7, 5, 1])
A2 = w**np.array([2, 10])
A2_rev = w**np.array([10, 2])
A2_sq = A2**2
A2_sq_rev = A2_rev**2
omega3_1 = w**12
omega3_2 = w**24
w6 = w**6
zminusz5inv = 1.0 / (w6 - w6**5)

# ---------- Small blocks ----------
def dft2(coeffs):
    T = w6 * coeffs[1]
    
    coeffs[1] = coeffs[0] + coeffs[1] - T;
    coeffs[0] = coeffs[0]             + T;

def idft2(coeffs):
    T1 = coeffs[0] + coeffs[1]
    T2 = (coeffs[0] - coeffs[1]) * zminusz5inv

    coeffs[0] = (T1 - T2) / 2.0
    coeffs[1] = T2

def dft6(coeffs):
    dft2(coeffs[0:2])
    dft2(coeffs[2:4])
    dft2(coeffs[4:6])

    T1 = A2    * coeffs[2:4]
    T2 = A2_sq * coeffs[4:6]
    T3 = (T1 - T2) * omega3_1

    coeffs[4:6] = coeffs[0:2] - T1 - T3
    coeffs[2:4] = coeffs[0:2] - T2 + T3
    coeffs[0:2] = coeffs[0:2] + T1 + T2

def idft6(coeffs):
    T1 = (coeffs[2:4] - coeffs[0:2])*omega3_1
    T2 = (coeffs[4:6] - coeffs[0:2] + T1)*A2_rev
    T3 = (coeffs[4:6] - coeffs[2:4] - T1)*A2_sq_rev
    
    coeffs[0:2] = (coeffs[0:2] + coeffs[2:4] + coeffs[4:6]) / 3.0
    coeffs[2:4] = T2 / 3.0
    coeffs[4:6] = T3 / 3.0

    idft2(coeffs[0:2])
    idft2(coeffs[2:4])
    idft2(coeffs[4:6])

def dft12(coeffs, verbose=False):  
    dft6(coeffs[0:6])
    dft6(coeffs[6:12])
    
    T = W6 * coeffs[6:12]

    coeffs[6:12] = coeffs[0:6] - T
    coeffs[0:6]  = coeffs[0:6] + T

def idft12(coeffs):
    T = coeffs[6:12].copy()
    
    coeffs[6:12] = W6_rev*(T - coeffs[0:6]) / 2.0
    coeffs[0:6] =         (T + coeffs[0:6]) / 2.0
    
    idft6(coeffs[0:6])
    idft6(coeffs[6:12])

def compare_dft6(verbose=True):
    np.random.seed(42)
    x6 = (np.random.randint(-5, 6, size=6) + 1j*np.random.randint(-5, 6, size=6)).astype(np.complex128)

    y_naive = naive_dft6(x6)
    
    perm = fft_input_permutation_2then3then2(6)
    x6_perm = x6[perm]
    
    dft6(x6_perm)

    diff = x6_perm - y_naive
    max_abs_diff = np.max(np.abs(diff))
    print("\n=== Comparison (radix-3 vs naive) ===")
    if verbose:
        print("Input x6:", x6)
    
    print("radix-3 dft6:", x6_perm)
    print("naive   dft6:", y_naive)
    print("max |diff|  :", max_abs_diff)

def compare_dft12(verbose=True):
    np.random.seed(123)
    x12 = (np.random.randint(-5, 6, size=12) + 1j*np.random.randint(-5, 6, size=12)).astype(np.complex128)

    y_naive = naive_dft12(x12)

    perm = fft_input_permutation_2then3then2(12)
    x12_perm = x12[perm]
    dft12(x12_perm, verbose=verbose)

    diff = x12_perm - y_naive
    max_abs_diff = np.max(np.abs(diff))
    print("\n=== Comparison (radix-2/3 dft12 vs naive) ===")
    if verbose:
        print("Input x12:", x12)
    
    print("radix dft12:", x12_perm)
    print("naive dft12:", y_naive)
    print("max |diff|  :", max_abs_diff)

def test_dft6_idft6():
    print("\n================ END-TO-END 6-POINT TEST ================")
    np.random.seed(42)
    x = (np.random.randint(-5, 6, size=6) + 1j*np.random.randint(-5, 6, size=6)).astype(np.complex128)
    print("Original Coeffs:\n", x, "\n")

    perm = fft_input_permutation_2then3then2(6)
    invperm = inverse_permutation(perm)
    x_perm = x[perm].copy()   # protect original & ensure contiguous buffer

    dft6(x_perm)
    print("DFT Result (normal order):\n", x_perm, "\n")

    idft6(x_perm)
    x_rec = x_perm[invperm]

    print("Recovered Coeffs:\n", x_rec, "\n")
    print("Allclose:", np.allclose(x, x_rec, atol=1e-12))
    print("Max abs error:", np.max(np.abs(x - x_rec)))


# ---------- End-to-end test (permute-in / inverse-permute-out) ----------
def test_dft12_idft12():
    print("\n================ END-TO-END 12-POINT TEST ================")
    np.random.seed(42)
    x = (np.random.randint(-5, 6, size=12) + 1j*np.random.randint(-5, 6, size=12)).astype(np.complex128)
    print("Original Coeffs:\n", x, "\n")

    perm = fft_input_permutation_2then3then2(12)
    invperm = inverse_permutation(perm)
    x_perm = x[perm].copy()

    dft12(x_perm)
    print("DFT Result (normal order):\n", x_perm, "\n")

    idft12(x_perm)
    x_rec = x_perm[invperm]

    print("Recovered Coeffs:\n", x_rec, "\n")
    print("Allclose:", np.allclose(x, x_rec))
    print("Max abs error:", np.max(np.abs(x - x_rec)))

if __name__ == "__main__":
    test_dft6_idft6()
    test_dft12_idft12()
    compare_dft6(verbose=True)
    compare_dft12(verbose=True)
