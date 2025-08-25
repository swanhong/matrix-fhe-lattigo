import numpy as np

def factorize_n(n):
    """
    Factorizes n into powers of 2 and 3.
    Equivalent to Sage's factor() for the specific case of n = 2^a * 3^b.
    """
    a = 0
    b = 0
    temp = n
    while temp > 0 and temp % 2 == 0:
        temp //= 2
        a += 1
    while temp > 0 and temp % 3 == 0:
        temp //= 3
        b += 1
    if temp != 1:
        raise ValueError(f"n={n} is not of the form 2^a * 3^b")
    return a, b

class CyclotomicDFT:
    """
    Generalized Cyclotomic DFT for polynomial rings C[X]/(X^N - X^{N/2} + 1)
    where N = 2^a * 3^b with a >= 1, b >= 0.
    
    This implementation is a Python/NumPy translation of the logic from the
    provided SageMath notebook (FFT_python/NTT_style_FFT.ipynb), which uses
    a mixed-radix Cooley-Tukey style algorithm.
    """
    
    def generator(self, n):
        return np.exp(-2j * np.pi / n)
        
    def __init__(self, N: int):
        """
        Initialize the Cyclotomic DFT for size N.
        
        Args:
            N: Transform size, must be of the form 2^a * 3^b where a>=1, b>=0.
        """
        self.n = N  # Use 'n' to match sage_fft_test.py
        self.N = N  # Keep N for backward compatibility

        self.Radix2, self.Radix3 = factorize_n(N)
        if self.Radix2 < 1:
            raise ValueError(f"unsupported n={N}, power of 2 must be >= 1")
        # Note: Radix3 can be 0 (pure powers of 2 are supported)
        
        self.level = self.Radix2 + self.Radix3

        # Set up the primitive root and other constants
        self.w = self.generator(3 * N)
        self.omega = self.w ** self.n
        self.omegaroot = np.exp(2j * np.pi / 3)
        self.omegaroot_inv = np.conjugate(self.omegaroot)
        
        self._build_tree_and_zetas()
        
        self.z = self.w**(self.n // 2)
        self.zminusz5inv = 1.0 / (self.z - self.z**5)
        self.level1 = 1.0 / self.n
        self.level2 = 2.0 / self.n

    def _build_tree_and_zetas(self):
        """
        Pre-computes all the twiddle factors (zetas) needed for the FFT and IFFT,
        storing them in self.zetas. This is a direct translation of the logic
        from the Sage notebook.
        """
        num_levels = self.Radix2 + self.Radix3
        self.tree = np.zeros((num_levels + 1, self.n), dtype=np.int64)
        self.tree[0, 0] = 3 * self.n

        self.zetas = [self.w**0]  # Start with w^0 = 1

        # Level 1: Radix-2 with cyclotomic adjustment
        self.tree[1, 0] = self.tree[0, 0] // 6
        self.tree[1, 1] = 5 * self.tree[0, 0] // 6
        self.zetas.append(self.w ** self.tree[1, 0])

        # Radix-3 part
        for ll in range(1, self.Radix3 + 1):
            for ii in range(2 * 3**(ll - 1)):
                self.tree[ll + 1, 3 * ii] = self.tree[ll, ii] // 3
                self.tree[ll + 1, 3 * ii + 1] = self.tree[ll + 1, 3 * ii] + self.tree[0, 0] // 3
                self.tree[ll + 1, 3 * ii + 2] = self.tree[ll + 1, 3 * ii] + 2 * self.tree[0, 0] // 3

                self.zetas.append(self.w ** self.tree[ll + 1, 3 * ii])
                self.zetas.append(self.w ** (2 * self.tree[ll + 1, 3 * ii]))

        # Radix-2 part
        for ll in range(self.Radix3 + 1, self.level):
            num_loops = 2 * 3**(self.Radix3) * 2**(ll - (self.Radix3 + 1))
            for ii in range(num_loops):
                self.tree[ll + 1, 2 * ii] = self.tree[ll, ii] // 2
                self.tree[ll + 1, 2 * ii + 1] = self.tree[ll, ii] // 2 + self.tree[0, 0] // 2
                self.zetas.append(self.w ** self.tree[ll + 1, 2 * ii])

        self.zetas = np.array(self.zetas)
        self.inv_zetas = np.conjugate(self.zetas)

    def dft(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Computes the forward Cyclotomic DFT.
        """
        b = np.array(coeffs, dtype=complex)
        zeta_idx = 1  # Skip first zeta (w^0 = 1)

        # 1. Special initial layer for cyclotomic requirement
        zeta = self.zetas[zeta_idx]
        zeta_idx += 1
        for i in range(self.n // 2):
            t = zeta * b[i + self.n // 2]
            b[i + self.n // 2] = b[i] + b[i + self.n // 2] - t
            b[i] = b[i] + t

        # 2. Radix-3 layers
        step = self.n // 6
        while step >= 2 ** (self.Radix2 - 1):
            for start in range(0, self.n, 3 * step):
                zeta1 = self.zetas[zeta_idx]
                zeta2 = self.zetas[zeta_idx + 1]
                zeta_idx += 2
                for i in range(start, start + step):
                    t1 = zeta1 * b[i + step]
                    t2 = zeta2 * b[i + 2 * step]
                    t3 = self.omega * (t1 - t2)
                    
                    b[i + 2 * step] = b[i] - t1 - t3
                    b[i + step] = b[i] - t2 + t3
                    b[i] = b[i] + t1 + t2
            step //= 3

        # 3. Radix-2 layers
        step = 2**(self.Radix2 - 2)
        while step >= 1:
            for start in range(0, self.n, step << 1):
                zeta = self.zetas[zeta_idx]
                zeta_idx += 1
                for i in range(start, start + step):
                    t = zeta * b[i + step]
                    b[i + step] = b[i] - t
                    b[i] = b[i] + t
            step //= 2
            
        return b

    def idft(self, dft_coeffs: np.ndarray) -> np.ndarray:
        """
        Computes the inverse Cyclotomic DFT.
        """
        b = np.array(dft_coeffs, dtype=complex)
        zeta_idx = len(self.zetas) - 1

        # 1. Radix-2 layers (inverse)
        step = 1
        while step <= 2**(self.Radix2 - 2):
            for start in range(0, self.n, 2 * step):
                zeta = self.zetas[zeta_idx]
                zeta_idx -= 1
                for i in range(start, start + step):
                    t = b[i + step]
                    b[i + step] = (t - b[i]) * zeta
                    b[i] = t + b[i]
            step *= 2

        # 2. Radix-3 layers (inverse)
        while step <= self.n // 6:
            for start in range(0, self.n, 3 * step):
                zeta2 = self.zetas[zeta_idx]
                zeta_idx -= 1
                zeta1 = self.zetas[zeta_idx]
                zeta_idx -= 1
                for i in range(start, start + step):
                    t1 = self.omega * (b[i + step] - b[i])
                    t2 = zeta1 * (b[i + 2 * step] - b[i] + t1)
                    t3 = zeta2 * (b[i + 2 * step] - b[i + step] - t1)
                    
                    b[i] = b[i] + b[i + step] + b[i + 2 * step]
                    b[i + step] = t2
                    b[i + 2 * step] = t3
            step *= 3

        # 3. Final cyclotomic layer (inverse)
        for i in range(self.n // 2):
            t1 = b[i] + b[i + self.n // 2]
            t2 = self.zminusz5inv * (b[i] - b[i + self.n // 2])
            
            b[i] = self.level1 * (t1 - t2)
            b[i + self.n // 2] = self.level2 * t2
            
        return b
