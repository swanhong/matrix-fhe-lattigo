# --- Naive 6-point DFT and IDFT ---
def naive_dft6(coeffs, verbose=False):
    N = 18
    index = [1, 5, 7, 11, 13, 17]
     
    out = np.zeros(6, dtype=np.complex128)

    for i in range(6): # index
        for j in range(len(coeffs)): # degree
            out[i] += coeffs[j] * np.exp(-2j * np.pi * index[i] * j / N)
    if verbose:
        print(f"  <- Naive DFT-6 Output: {out}")
    return out

##########################################
#  For pure radix-2 NTTs, the transform matrix W satisfies W W* = nI,
# so the inverse can be obtained natively by conjugating twiddles and
# reversing the flow. However, this property does not hold for x^6 − x^3 + 1,
# where the mixed 2×3 structure breaks the simple W W* = nI relation.
##########################################
def naive_idft6(coeffs, verbose=False):
    N = 6
    out = np.zeros(6, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            out[n] += coeffs[k] * np.exp(2j * np.pi * k * n / N)
    out /= 6.0
    if verbose:
        print(f"  <- Naive IDFT-6 Output: {out}")
    return out
# --- 3-point DFT and IDFT ---
def dft3(coeffs, verbose=False):
    """Naive 3-point DFT."""
    if verbose:
        print(f"  -> DFT-3 Input: {coeffs}")
    N = 3
    out = np.zeros(3, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            out[k] += coeffs[n] * np.exp(-2j * np.pi * k * n / N)
    if verbose:
        print(f"  <- DFT-3 Output: {out}")
    return out

def idft3(coeffs, verbose=False):
    """Naive 3-point IDFT."""
    if verbose:
        print(f"  -> IDFT-3 Input: {coeffs}")
    N = 3
    out = np.zeros(3, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            out[n] += coeffs[k] * np.exp(2j * np.pi * k * n / N)
    out /= 3.0
    if verbose:
        print(f"  <- IDFT-3 Output: {out}")
    return out
import numpy as np

# Set a consistent seed for reproducible random numbers
np.random.seed(42)

# --- Global Parameters & Pre-computed Twiddle Factors ---

# We are working with the 36th cyclotomic polynomial, so M=36.
M = 36
w = np.exp(- 2j * np.pi / M)

# Twiddles for Radix-2 step (DFT-12 -> DFT-6)
# These are the specific roots w^k for k in {1, 5, 7, 11, 13, 17}
W6 = w**np.array([1, 5, 7, 11, 13, 17])

# Twiddles for Radix-3 step (DFT-6 -> DFT-2)
# A2 corresponds to y = x^2, with roots {w^2, w^10}
A2 = w**np.array([2, 10])
A2_sq = A2**2
# 3rd roots of unity
omega3_1 = w**12
omega3_2 = w**24

# Twiddle for Base Case (DFT-2)
w6 = w**6

# print parameters for DFT:
print("=========================================")
print("         DFT Parameters                  ")
print(f"M = {M}")
print(f"w = {w}")
print(f"W6 = {W6}")
print(f"A2 = {A2}")
print(f"A2_sq = {A2_sq}")
print(f"omega3_1 = {omega3_1}")
print(f"omega3_2 = {omega3_2}")
print(f"w6 = {w6}")
print("=========================================")

# --- Forward FFT (DFT) Implementation ---

def dft2(coeffs, verbose=False):
    """Base case: 2-point DFT using the special S2 matrix."""
    if verbose:
        print(f"  -> DFT-2 Input: {coeffs}")
        
    out = np.zeros(2, dtype=np.complex128)
    out[0] = coeffs[0] + w6 * coeffs[1]
    # The special relation from the ring: w^30 = 1 - w^6
    out[1] = coeffs[0] + (1 - w6) * coeffs[1]
    
    if verbose:
        print(f"  <- DFT-2 Output: {out}")
    return out

def dft6(coeffs, verbose=False):
    """Radix-3 decomposition of a 6-point DFT."""
    if verbose:
        print(f" -> DFT-6 Input: {coeffs}")


    # 1. Split into 3 groups of 2 (stride-3)
    b0, b1, b2 = coeffs[::3], coeffs[1::3], coeffs[2::3]

    # 2. Perform DFT-2 on each subgroup
    B0 = dft2(b0, verbose)
    B1 = dft2(b1, verbose)
    B2 = dft2(b2, verbose)

    # 3. Apply twiddles to B1, B2
    X0 = B0
    X1 = A2 * B1
    X2 = A2_sq * B2

    # 4. Radix-3 butterfly
    out = np.zeros(6, dtype=np.complex128)
    for k in range(2):
        out[k]   = X0[k] + X1[k] + X2[k]
        out[k+2] = X0[k] + omega3_1 * X1[k] + omega3_2 * X2[k]
        out[k+4] = X0[k] + omega3_2 * X1[k] + omega3_1 * X2[k]

    if verbose:
        print(f" <- DFT-6 Output: {out}")
    return out
    
def dft12(coeffs, verbose=False):
    """Top-level Radix-2 decomposition of a 12-point DFT."""
    if verbose:
        print(f"-> DFT-12 Input: {coeffs}")

    # 1. Split into even and odd coefficients
    even_coeffs, odd_coeffs = coeffs[::2], coeffs[1::2]
    
    # 2. Perform DFT-6 on each subgroup
    E = dft6(even_coeffs, verbose)
    O = dft6(odd_coeffs, verbose)
    
    # 3. Combine using Radix-2 butterfly
    T = W6 * O
    Y_top = E + T
    Y_bottom = E - T
    
    out = np.concatenate([Y_top, Y_bottom])
    if verbose:
        print(f"<- DFT-12 Output: {out}")
    return out

# --- Backward FFT (IDFT) Implementation ---

def idft2(coeffs, verbose=False):
    """Inverse of the special 2-point DFT."""
    if verbose:
        print(f"  -> IDFT-2 Input: {coeffs}")
        
    # Inverse of the S2 matrix [1, w6; 1, 1-w6]
    # det = (1 - w6) - w6 = 1 - 2*w6
    inv_det = 1.0 / (1 - 2 * w6)
    
    out = np.zeros(2, dtype=np.complex128)
    out[0] = inv_det * ((1 - w6) * coeffs[0] - w6 * coeffs[1])
    out[1] = inv_det * (-coeffs[0] + coeffs[1])
    
    if verbose:
        print(f"  <- IDFT-2 Output: {out}")
    return out
    
def idft6(coeffs, verbose=False):
    """Inverse of the Radix-3 decomposition."""
    if verbose:
        print(f" -> IDFT-6 Input: {coeffs}")
        
    # 1. Radix-3 inverse butterfly (use conjugate omegas)
    B = np.zeros(6, dtype=np.complex128)
    for k in range(2):
        B[k]   = omega3_1    * (coeffs[k+2] - coeffs[k]) 
        B[k+2] = A2[1-k]     * (coeffs[k+4] - coeffs[k]   + B[k])
        B[k+4] = A2_sq[1-k]  * (coeffs[k+4] - coeffs[k+2] - B[k])

        B[k] = coeffs[k] + coeffs[k+2] + coeffs[k+4]

    # 2. Perform inverse DFT-2 on each subgroup
    X0 = idft2(B[0:2]/3, verbose)
    X1 = idft2(B[2:4]/3, verbose)
    X2 = idft2(B[4:6]/3, verbose)

    out = np.empty_like(coeffs)

    out[::3] = X0
    out[1::3] = X1
    out[2::3] = X2

    if verbose:
        print(f" <- IDFT-6 Output: {out}")
    return out
    
def idft12(coeffs, verbose=False):
    """Top-level inverse of the Radix-2 decomposition."""
    if verbose:
        print(f"-> IDFT-12 Input: {coeffs}")
    
    # 1. Split into top and bottom halves
    Y_top, Y_bottom = coeffs[:6], coeffs[6:]
    
    # 2. Perform inverse Radix-2 butterfly (match dft12 structure)
    E = (Y_top + Y_bottom) / 2.0
    T = (Y_bottom - Y_top) / 2.0
    O = T * W6[::-1]

    # 3. Perform IDFT-6 on each result
    even_coeffs = idft6(E, verbose)
    odd_coeffs = idft6(O, verbose)

    # 4. Reconstruct the output array and apply final scaling
    out = np.zeros(12, dtype=np.complex128)
    out[::2] = even_coeffs
    out[1::2] = odd_coeffs

    # For this specific DFT over a cyclotomic ring, the scaling factor
    # is 1/N which is 1/12. Apply scaling here.
    final_out = out # / 12.0

    if verbose:
        print(f"<- IDFT-12 Output: {final_out}")
    return final_out

# --- Main Execution ---
if __name__ == "__main__":
    def test_compare_dft6():
        print("\n=========================================")
        print("         DFT-6: RADIX-3 VS NAIVE         ")
        print("=========================================")
        x6 = (np.random.randint(-5, 6, size=6) + 1j * np.random.randint(-5, 6, size=6)).astype(np.complex128)
        print(f"Input: {x6}")
        y6_radix = dft6(x6, verbose=True)
        y6_naive = naive_dft6(x6, verbose=True)
        print(f"Radix-3 DFT-6: {y6_radix}")
        print(f"Naive   DFT-6: {y6_naive}")
        print(f"DFT-6 max abs diff: {np.max(np.abs(y6_radix - y6_naive))}")
        # Now test inverse
        x6_rec_radix = idft6(y6_radix, verbose=True)
        x6_rec_naive = naive_idft6(y6_naive, verbose=True)
        print(f"Radix-3 IDFT-6(x6): {x6_rec_radix}")
        print(f"Naive   IDFT-6(x6): {x6_rec_naive}")
        print(f"IDFT-6 max abs diff (radix-3 vs naive): {np.max(np.abs(x6_rec_radix - x6_rec_naive))}")
        print(f"Original vs Radix-3 IDFT-6: {np.max(np.abs(x6 - x6_rec_radix))}")
        print(f"Original vs Naive   IDFT-6: {np.max(np.abs(x6 - x6_rec_naive))}")
    def test_dft6_idft6():
        print("\n=========================================")
        print("         DFT-6 <-> IDFT-6 TEST           ")
        print("=========================================")
        x6 = (np.random.randint(-5, 6, size=6) + 1j * np.random.randint(-5, 6, size=6)).astype(np.complex128)
        print(f"Original 6-point input: {x6}")
        y6 = dft6(x6, verbose=True)
        x6_rec = idft6(y6, verbose=True)
        print(f"Recovered by IDFT-6: {x6_rec}")
        print(f"Allclose: {np.allclose(x6, x6_rec)}")
        print(f"Max abs error: {np.max(np.abs(x6 - x6_rec))}")
    def test_dft3_idft3():
        print("\n=========================================")
        print("         DFT-3 <-> IDFT-3 TEST           ")
        print("=========================================")
        x3 = (np.random.randint(-5, 6, size=3) + 1j * np.random.randint(-5, 6, size=3)).astype(np.complex128)
        print(f"Original 3-point input: {x3}")
        y3 = dft3(x3, verbose=True)
        x3_rec = idft3(y3, verbose=True)
        print(f"Recovered by IDFT-3: {x3_rec}")
        print(f"Allclose: {np.allclose(x3, x3_rec)}")
        print(f"Max abs error: {np.max(np.abs(x3 - x3_rec))}")
    def test_dft2_idft2():
        print("\n=========================================")
        print("         DFT-2 <-> IDFT-2 TEST           ")
        print("=========================================")
        x2 = (np.random.randint(-5, 6, size=2) + 1j * np.random.randint(-5, 6, size=2)).astype(np.complex128)
        print(f"Original 2-point input: {x2}")
        y2 = dft2(x2, verbose=True)
        x2_rec = idft2(y2, verbose=True)
        print(f"Recovered by IDFT-2: {x2_rec}")
        print(f"Allclose: {np.allclose(x2, x2_rec)}")
        print(f"Max abs error: {np.max(np.abs(x2 - x2_rec))}")

    def test_dft12_idft12():
        print("\n=========================================")
        print("         END-TO-END 12-POINT TEST        ")
        print("=========================================")
        p_coeffs = (np.random.randint(-5, 6, size=12) + 1j * np.random.randint(-5, 6, size=12)).astype(np.complex128)
        print(f"Original Coefficients:\n{p_coeffs}\n")
        dft_result = dft12(p_coeffs, verbose=True)
        print("\n-----------------------------------------")
        print(f"Final DFT Result (Evaluations):\n{dft_result}")
        print("-----------------------------------------\n")
        print("=========================================")
        print("        BACKWARD FFT (IDFT)              ")
        print("=========================================")
        print(f"DFT Result (Input to IDFT):\n{dft_result}\n")
        recovered_coeffs = idft12(dft_result, verbose=True)
        print("\n-----------------------------------------")
        print(f"Recovered Coefficients:\n{recovered_coeffs}")
        print("-----------------------------------------\n")
        print("=========================================")
        print("            VERIFICATION                 ")
        print("=========================================")
        print(f"Original matches recovered: {np.allclose(p_coeffs, recovered_coeffs)}")
        error = np.max(np.abs(p_coeffs - recovered_coeffs))
        print(f"Maximum absolute error: {error}")

    # Run all tests
    test_dft2_idft2()
    test_dft3_idft3()
    test_dft6_idft6()
    # test_compare_dft6()
    test_dft12_idft12()