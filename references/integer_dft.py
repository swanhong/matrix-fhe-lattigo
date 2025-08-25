import numpy as np
from typing import Tuple, Optional
import math

def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def mod_pow(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) % mod efficiently."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def find_primitive_root(p: int) -> Optional[int]:
    """Find a primitive root modulo prime p."""
    if not is_prime(p):
        return None
    
    phi = p - 1
    factors = []
    
    # Find prime factors of phi
    temp = phi
    for i in range(2, int(math.sqrt(phi)) + 1):
        if temp % i == 0:
            factors.append(i)
            while temp % i == 0:
                temp //= i
    
    if temp > 1:
        factors.append(temp)
    
    # Test for primitive root
    for g in range(2, p):
        is_primitive = True
        for factor in factors:
            if mod_pow(g, phi // factor, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    
    return None

def find_nth_root(n: int, p: int) -> Optional[int]:
    """Find primitive nth root of unity modulo prime p."""
    if (p - 1) % n != 0:
        return None
    
    g = find_primitive_root(p)
    if g is None:
        return None
    
    return mod_pow(g, (p - 1) // n, p)

def find_cyclotomic_polynomial_roots(N: int, p: int) -> list:
    """Find all roots of the cyclotomic polynomial X^N - X^{N/2} + 1 modulo p."""
    roots = []
    for x in range(p):
        # Check if x^N - x^{N/2} + 1 ≡ 0 (mod p)
        x_N = mod_pow(x, N, p)
        x_N_half = mod_pow(x, N // 2, p)
        if (x_N - x_N_half + 1) % p == 0:
            roots.append(x)
    return roots

class IntegerDFT:
    """Integer-based DFT for cyclotomic polynomial multiplication over Z_p[X]/(X^N - X^{N/2} + 1)."""
    
    def __init__(self, N: int, p: Optional[int] = None, min_bits: int = 20, verbose: bool = False):
        """
        Initialize cyclotomic integer DFT for size N.
        
        Args:
            N: Transform size (must be of form 2^a * 3^b)
            p: Prime modulus (if specified, will validate it satisfies p ≡ 1 (mod 3N))
            min_bits: Minimum bit size for the prime (default: 20, i.e., p >= 2^20)
            verbose: Enable verbose output for debugging
        """
        self.N = N
        self.verbose = verbose
        
        # Factor N into 2^a * 3^b form
        self.a, self.b = self._factor_N(N)
        if 2**self.a * 3**self.b != N:
            raise ValueError(f"N = {N} must be of form 2^a * 3^b, got 2^{self.a} * 3^{self.b}")
        
        # Handle prime specification
        if p is None:
            # Find suitable prime automatically
            p = self._find_suitable_prime(N, min_bits)
        else:
            # Validate the provided prime
            self._validate_custom_prime(p, N)
        
        self.p = p
        
        # For cyclotomic polynomial FFT over Z_p[X]/(X^N - X^{N/2} + 1),
        # we need to use actual roots of the cyclotomic polynomial as evaluation points
        self.evaluation_points = find_cyclotomic_polynomial_roots(N, p)
        if len(self.evaluation_points) < N:
            raise ValueError(f"Not enough roots of cyclotomic polynomial mod {p}: found {len(self.evaluation_points)}, need {N}")
        
        # Use the first N roots as evaluation points
        self.evaluation_points = self.evaluation_points[:N]
        
        # Set up primitive roots for factorized NTT
        cyclotomic_order = 3 * N
        self.w = find_nth_root(cyclotomic_order, p)  # primitive 18th root (like complex w)
        if self.w is None:
            raise ValueError(f"Cannot find {cyclotomic_order}th root of unity modulo {p}")
        
        # Key primitive roots for factorized algorithm
        self.omega = mod_pow(self.w, cyclotomic_order // 3, p)  # primitive 3rd root (like complex omega)
        self.z = mod_pow(self.w, N // 2, p)  # N/2-th power (like complex z)
        
        # Precompute inverses and other constants
        self.omega_inv = mod_pow(self.omega, p - 2, p)
        self.N_inv = mod_pow(N, p - 2, p)
        self.z5 = mod_pow(self.z, 5, p)
        z_minus_z5 = (self.z - self.z5) % p
        self.zminusz5inv = mod_pow(z_minus_z5, p - 2, p)
        
        # Build twiddle factors for factorized algorithm
        self._build_integer_twiddle_factors()
        
        if self.verbose:
            print(f"Initialized Cyclotomic Integer NTT for N={N} = 2^{self.a} * 3^{self.b}")
            print(f"Using prime p={p}")
            print(f"Primitive 18th root w={self.w}")
            print(f"Primitive 3rd root omega={self.omega}")
            print(f"Special constant z={self.z}")
            print(f"Evaluation points: {self.evaluation_points}")

    def _build_integer_twiddle_factors(self):
        """Build twiddle factors for factorized integer NTT, adapted from complex version."""
        # Factor N into powers of 2 and 3
        self.Radix2, self.Radix3 = self.a, self.b
        self.level = self.Radix2 + self.Radix3
        
        # Build tree structure (adapted from complex version)
        self.tree = np.zeros((self.level + 1, self.N), dtype=np.int64)
        self.tree[0, 0] = 3 * self.N
        
        self.zetas = [1]  # Start with w^0 = 1 (in mod p)
        
        # Level 1: Radix-2 with cyclotomic adjustment
        self.tree[1, 0] = self.tree[0, 0] // 6
        self.tree[1, 1] = 5 * self.tree[0, 0] // 6
        self.zetas.append(mod_pow(self.w, self.tree[1, 0], self.p))
        
        # Radix-3 part
        for ll in range(1, self.Radix3 + 1):
            for ii in range(2 * 3**(ll - 1)):
                self.tree[ll + 1, 3 * ii] = self.tree[ll, ii] // 3
                self.tree[ll + 1, 3 * ii + 1] = self.tree[ll + 1, 3 * ii] + self.tree[0, 0] // 3
                self.tree[ll + 1, 3 * ii + 2] = self.tree[ll + 1, 3 * ii] + 2 * self.tree[0, 0] // 3
                
                self.zetas.append(mod_pow(self.w, self.tree[ll + 1, 3 * ii], self.p))
                self.zetas.append(mod_pow(self.w, 2 * self.tree[ll + 1, 3 * ii], self.p))
        
        # Radix-2 part
        for ll in range(self.Radix3 + 1, self.level):
            num_loops = 2 * 3**(self.Radix3) * 2**(ll - (self.Radix3 + 1))
            for ii in range(num_loops):
                self.tree[ll + 1, 2 * ii] = self.tree[ll, ii] // 2
                self.tree[ll + 1, 2 * ii + 1] = self.tree[ll, ii] // 2 + self.tree[0, 0] // 2
                self.zetas.append(mod_pow(self.w, self.tree[ll + 1, 2 * ii], self.p))

    def _factor_N(self, N: int) -> Tuple[int, int]:
        """Factor N into 2^a * 3^b form."""
        if N <= 0:
            return 0, 0
        
        a = 0
        temp_N = N
        while temp_N % 2 == 0:
            temp_N //= 2
            a += 1
        
        b = 0
        while temp_N % 3 == 0:
            temp_N //= 3
            b += 1
        
        if temp_N != 1:
            raise ValueError(f"N = {N} has factors other than 2 and 3")
        
        return a, b

    def _validate_custom_prime(self, p: int, N: int) -> None:
        """
        Validate that a custom prime satisfies the required conditions.
        
        Args:
            p: The prime to validate
            N: Transform size
            
        Raises:
            ValueError: If the prime doesn't satisfy the required conditions
        """
        # Check if p is actually prime
        if not is_prime(p):
            raise ValueError(f"Provided p = {p} is not prime")
        
        # Check the cyclotomic condition: p ≡ 1 (mod 3N)
        required_modulus = 3 * N
        if (p - 1) % required_modulus != 0:
            raise ValueError(f"Prime p = {p} does not satisfy p ≡ 1 (mod {required_modulus}). Got p - 1 = {p - 1}, need (p - 1) % {required_modulus} = 0")
        
        if self.verbose:
            actual_bits = p.bit_length()
            print(f"Validated custom prime: p = {p} (≡ 1 mod {required_modulus}, {actual_bits} bits)")

    def _find_suitable_prime(self, N: int, min_bits: int = 20) -> int:
        """
        Find a prime p such that p ≡ 1 (mod 3N) for cyclotomic polynomial arithmetic
        and p >= 2^min_bits.
        
        Args:
            N: Transform size
            min_bits: Minimum bit size for the prime (p >= 2^min_bits)
            
        Returns:
            Prime p satisfying both conditions
        """
        required_modulus = 3 * N
        min_prime = 2 ** min_bits
        
        # Start search from the smallest k such that k * required_modulus + 1 >= min_prime
        k = max(1, (min_prime - 1) // required_modulus)
        
        if self.verbose:
            print(f"Searching for prime p ≡ 1 (mod {required_modulus}) with p >= 2^{min_bits} = {min_prime}")
            print(f"Starting search from k = {k}")
        
        while True:
            candidate = k * required_modulus + 1
            if candidate >= min_prime and is_prime(candidate):
                if self.verbose:
                    actual_bits = candidate.bit_length()
                    print(f"Found suitable prime: p = {candidate} (≡ 1 mod {required_modulus}, {actual_bits} bits)")
                return candidate
            k += 1
            
            # Safety check to avoid infinite loops
            if k > 1000000:
                raise ValueError(f"Could not find suitable prime after 1M iterations. Try smaller min_bits or different N.")
        
    
    def factorized_dft(self, coeffs: list, verbose: bool = None) -> list:
        """Compute forward cyclotomic DFT using factorized NTT (adapted from complex version)."""
        if len(coeffs) != self.N:
            raise ValueError(f"Input must have length {self.N}")
        
        use_verbose = verbose if verbose is not None else self.verbose
        
        # Convert to integers mod p
        b = [int(x) % self.p for x in coeffs]
        
        if use_verbose:
            print(f"\n{'='*50}")
            print(f"  FACTORIZED CYCLOTOMIC NTT-{self.N}")
            print(f"{'='*50}")
            print(f" -> Input: {b}")
        
        zeta_idx = 1  # Skip first zeta (w^0 = 1)
        
        # 1. Special initial layer for cyclotomic requirement
        zeta = self.zetas[zeta_idx]
        zeta_idx += 1
        if use_verbose:
            print(f" -> Initial cyclotomic layer with zeta={zeta}")
        
        for i in range(self.N // 2):
            t = (zeta * b[i + self.N // 2]) % self.p
            b[i + self.N // 2] = (b[i] + b[i + self.N // 2] - t) % self.p
            b[i] = (b[i] + t) % self.p
        
        if use_verbose:
            print(f"    After initial layer: {b}")
        
        # 2. Radix-3 layers
        step = self.N // 6
        layer = 1
        while step >= 2 ** (self.Radix2 - 1):
            if use_verbose:
                print(f" -> Radix-3 layer {layer}, step={step}")
            
            for start in range(0, self.N, 3 * step):
                zeta1 = self.zetas[zeta_idx]
                zeta2 = self.zetas[zeta_idx + 1]
                zeta_idx += 2
                
                for i in range(start, start + step):
                    t1 = (zeta1 * b[i + step]) % self.p
                    t2 = (zeta2 * b[i + 2 * step]) % self.p
                    t3 = (self.omega * (t1 - t2)) % self.p
                    
                    b[i + 2 * step] = (b[i] - t1 - t3) % self.p
                    b[i + step] = (b[i] - t2 + t3) % self.p
                    b[i] = (b[i] + t1 + t2) % self.p
            
            if use_verbose:
                print(f"    After radix-3 layer {layer}: {b}")
            step //= 3
            layer += 1
        
        # 3. Radix-2 layers
        step = 2**(self.Radix2 - 2)
        layer = 1
        while step >= 1:
            if use_verbose:
                print(f" -> Radix-2 layer {layer}, step={step}")
            
            for start in range(0, self.N, step << 1):
                zeta = self.zetas[zeta_idx]
                zeta_idx += 1
                
                for i in range(start, start + step):
                    t = (zeta * b[i + step]) % self.p
                    b[i + step] = (b[i] - t) % self.p
                    b[i] = (b[i] + t) % self.p
            
            if use_verbose:
                print(f"    After radix-2 layer {layer}: {b}")
            step //= 2
            layer += 1
        
        if use_verbose:
            print(f" <- Final output: {b}")
        
        return b

    def factorized_idft(self, dft_coeffs: list, verbose: bool = None) -> list:
        """Compute inverse cyclotomic DFT using factorized NTT (adapted from complex version)."""
        if len(dft_coeffs) != self.N:
            raise ValueError(f"Input must have length {self.N}")
        
        use_verbose = verbose if verbose is not None else self.verbose
        
        # Convert to integers mod p  
        b = [int(x) % self.p for x in dft_coeffs]
        
        if use_verbose:
            print(f"\n{'='*50}")
            print(f" FACTORIZED CYCLOTOMIC INTT-{self.N}")
            print(f"{'='*50}")
            print(f" -> Input: {b}")
        
        zeta_idx = len(self.zetas) - 1
        
        # 1. Radix-2 layers (inverse)
        step = 1
        layer = 1
        while step <= 2**(self.Radix2 - 2):
            if use_verbose:
                print(f" -> Inverse radix-2 layer {layer}, step={step}")
            
            for start in range(0, self.N, 2 * step):
                zeta = self.zetas[zeta_idx]
                zeta_idx -= 1
                
                for i in range(start, start + step):
                    t = b[i + step]
                    b[i + step] = ((t - b[i]) * zeta) % self.p
                    b[i] = (t + b[i]) % self.p
            
            if use_verbose:
                print(f"    After inverse radix-2 layer {layer}: {b}")
            step *= 2
            layer += 1
        
        # 2. Radix-3 layers (inverse)  
        layer = 1
        while step <= self.N // 6:
            if use_verbose:
                print(f" -> Inverse radix-3 layer {layer}, step={step}")
            
            for start in range(0, self.N, 3 * step):
                zeta2 = self.zetas[zeta_idx]
                zeta_idx -= 1
                zeta1 = self.zetas[zeta_idx]
                zeta_idx -= 1
                
                for i in range(start, start + step):
                    t1 = (self.omega * (b[i + step] - b[i])) % self.p
                    t2 = (zeta1 * (b[i + 2 * step] - b[i] + t1)) % self.p
                    t3 = (zeta2 * (b[i + 2 * step] - b[i + step] - t1)) % self.p
                    
                    b[i] = (b[i] + b[i + step] + b[i + 2 * step]) % self.p
                    b[i + step] = t2
                    b[i + 2 * step] = t3
            
            if use_verbose:
                print(f"    After inverse radix-3 layer {layer}: {b}")
            step *= 3
            layer += 1
        
        # 3. Final cyclotomic layer (inverse)
        if use_verbose:
            print(f" -> Final inverse cyclotomic layer")
        
        level1 = mod_pow(self.N, self.p - 2, self.p)  # 1/N mod p
        level2 = (2 * level1) % self.p  # 2/N mod p
        
        for i in range(self.N // 2):
            t1 = (b[i] + b[i + self.N // 2]) % self.p
            t2 = (self.zminusz5inv * (b[i] - b[i + self.N // 2])) % self.p
            
            b[i] = (level1 * (t1 - t2)) % self.p
            b[i + self.N // 2] = (level2 * t2) % self.p
        
        if use_verbose:
            print(f" <- Final output: {b}")
        
        return b
    def dft(self, coeffs: list, verbose: bool = None, method: str = "factorized") -> list:
        """Compute forward cyclotomic DFT.
        
        Args:
            coeffs: Input polynomial coefficients
            verbose: Enable verbose output
            method: "factorized" for O(N log N) factorized NTT, "direct" for O(N²) direct evaluation
        """
        if method == "factorized":
            return self.factorized_dft(coeffs, verbose)
        elif method == "direct":
            return self._direct_dft(coeffs, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'factorized' or 'direct'")
    
    def idft(self, dft_coeffs: list, verbose: bool = None, method: str = "factorized") -> list:
        """Compute inverse cyclotomic DFT.
        
        Args:
            dft_coeffs: DFT coefficients to invert
            verbose: Enable verbose output  
            method: "factorized" for O(N log N) factorized NTT, "direct" for O(N²) direct interpolation
        """
        if method == "factorized":
            return self.factorized_idft(dft_coeffs, verbose)
        elif method == "direct":
            return self._direct_idft(dft_coeffs, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'factorized' or 'direct'")

    def _direct_dft(self, coeffs: list, verbose: bool = None) -> list:
        if len(coeffs) != self.N:
            raise ValueError(f"Input must have length {self.N}")
        
        # Use provided verbose setting or fall back to instance setting
        use_verbose = verbose if verbose is not None else self.verbose
        
        # Convert to integers mod p
        a = [int(x) % self.p for x in coeffs]
        
        # Use direct evaluation at cyclotomic polynomial roots
        if use_verbose:
            print(f"\n{'='*50}")
            print(f"    CYCLOTOMIC DFT-{self.N} (Direct Evaluation)")
            print(f"{'='*50}")
            print(f" -> Input: {a}")
            print(f" -> Evaluation points: {self.evaluation_points}")
        
        result = self._direct_evaluation(a, use_verbose)
        
        if use_verbose:
            print(f" <- Output: {result}")
            print(f"{'='*50}")
        
        return result

    def _direct_idft(self, coeffs: list, verbose: bool = None) -> list:
        """Compute inverse cyclotomic DFT using direct interpolation from evaluations."""
        if len(coeffs) != self.N:
            raise ValueError(f"Input must have length {self.N}")
        
        # Use provided verbose setting or fall back to instance setting
        use_verbose = verbose if verbose is not None else self.verbose
        
        # Convert to integers mod p
        a = [int(x) % self.p for x in coeffs]
        
        # Use direct interpolation from cyclotomic polynomial root evaluations
        if use_verbose:
            print(f"\n{'='*50}")
            print(f"   CYCLOTOMIC IDFT-{self.N} (Direct Interpolation)")
            print(f"{'='*50}")
            print(f" -> Input evaluations: {a}")
            print(f" -> Interpolation points: {self.evaluation_points}")
        
        result = self._direct_interpolation(a, use_verbose)
        
        if use_verbose:
            print(f" <- Output coefficients: {result}")
            print(f"{'='*50}")
        
        return result
    
    def _factorized_ntt(self, coeffs: list, n: int, verbose: bool = False, level: int = 0) -> list:
        """Factorized NTT adapted for cyclotomic polynomial evaluation."""
        indent = "  " * level
        
        if verbose:
            print(f"{indent}┌─ CYCLOTOMIC NTT-{n} FACTORIZATION (Level {level})")
            print(f"{indent}│  Input: {coeffs}")
        
        # For cyclotomic polynomials, we use direct evaluation when size gets small
        # or when factorization doesn't benefit us much
        if n <= 4:
            if verbose:
                print(f"{indent}├─ Using DIRECT EVALUATION for small size {n}")
            
            # Get the appropriate subset of evaluation points for this size
            start_idx = (self.N - n) // 2 if self.N > n else 0
            eval_points = self.evaluation_points[start_idx:start_idx + n]
            
            result = []
            for point in eval_points:
                # Evaluate polynomial at this point using Horner's method
                value = 0
                for j in range(len(coeffs) - 1, -1, -1):
                    value = (value * point + coeffs[j]) % self.p
                result.append(value)
            
            if verbose:
                print(f"{indent}│  Evaluation points: {eval_points}")
                print(f"{indent}└─ CYCLOTOMIC NTT-{n} Result: {result}")
            return result
        
        # For larger sizes, try to use factorization if possible
        if n % 2 == 0:
            # Radix-2 decomposition
            n1, n2 = 2, n // 2
            if verbose:
                print(f"{indent}│  Using RADIX-2 decomposition: {n} = {n1} × {n2}")
            result = self._cyclotomic_radix2_ntt(coeffs, n1, n2, verbose, level + 1)
        
        elif n % 3 == 0:
            # Radix-3 decomposition
            n1, n2 = 3, n // 3
            if verbose:
                print(f"{indent}│  Using RADIX-3 decomposition: {n} = {n1} × {n2}")
            result = self._cyclotomic_radix3_ntt(coeffs, n1, n2, verbose, level + 1)
        
        else:
            # Fall back to direct evaluation for sizes that can't be factorized
            if verbose:
                print(f"{indent}├─ Cannot factorize {n}, using DIRECT EVALUATION")
            
            eval_points = self.evaluation_points[:n]
            result = []
            for point in eval_points:
                value = 0
                for j in range(len(coeffs) - 1, -1, -1):
                    value = (value * point + coeffs[j]) % self.p
                result.append(value)
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC NTT-{n} Result: {result}")
        
        return result
    
    def _factorized_intt(self, coeffs: list, n: int, verbose: bool = False, level: int = 0) -> list:
        """Factorized inverse NTT adapted for cyclotomic polynomial interpolation."""
        indent = "  " * level
        
        if verbose:
            print(f"{indent}┌─ CYCLOTOMIC INTT-{n} FACTORIZATION (Level {level})")
            print(f"{indent}│  Input: {coeffs}")
        
        # For cyclotomic polynomials, use direct interpolation for small sizes
        if n <= 4:
            if verbose:
                print(f"{indent}├─ Using DIRECT INTERPOLATION for small size {n}")
            
            # Get the appropriate subset of evaluation points for this size
            start_idx = (self.N - n) // 2 if self.N > n else 0
            eval_points = self.evaluation_points[start_idx:start_idx + n]
            
            # Direct Lagrange interpolation
            result = [0] * n
            
            for i in range(n):
                if coeffs[i] == 0:
                    continue
                    
                # Build Lagrange basis polynomial L_i(x)
                li_coeffs = [1]  # Start with polynomial 1
                
                for j in range(n):
                    if i != j:
                        # Multiply by (x - eval_points[j]) / (eval_points[i] - eval_points[j])
                        denom = (eval_points[i] - eval_points[j]) % self.p
                        denom_inv = mod_pow(denom, self.p - 2, self.p)  # Modular inverse
                        
                        # Multiply li_coeffs by (x - eval_points[j])
                        new_li_coeffs = [0] * (len(li_coeffs) + 1)
                        for k in range(len(li_coeffs)):
                            new_li_coeffs[k] = (new_li_coeffs[k] - li_coeffs[k] * eval_points[j]) % self.p
                            new_li_coeffs[k + 1] = (new_li_coeffs[k + 1] + li_coeffs[k]) % self.p
                        
                        # Multiply by 1/(eval_points[i] - eval_points[j])
                        li_coeffs = [(coeff * denom_inv) % self.p for coeff in new_li_coeffs]
                
                # Add contribution coeffs[i] * L_i(x) to result
                for k in range(min(len(li_coeffs), len(result))):
                    result[k] = (result[k] + coeffs[i] * li_coeffs[k]) % self.p
            
            if verbose:
                print(f"{indent}└─ CYCLOTOMIC INTT-{n} Result: {result}")
            return result
        
        # For larger sizes, try to use factorization if possible
        if n % 2 == 0:
            # Inverse radix-2 decomposition
            n1, n2 = 2, n // 2
            if verbose:
                print(f"{indent}│  Using INVERSE RADIX-2 decomposition: {n} = {n1} × {n2}")
            result = self._cyclotomic_radix2_intt(coeffs, n1, n2, verbose, level + 1)
        
        elif n % 3 == 0:
            # Inverse radix-3 decomposition
            n1, n2 = 3, n // 3
            if verbose:
                print(f"{indent}│  Using INVERSE RADIX-3 decomposition: {n} = {n1} × {n2}")
            result = self._cyclotomic_radix3_intt(coeffs, n1, n2, verbose, level + 1)
        
        else:
            # Fall back to direct interpolation for sizes that can't be factorized
            if verbose:
                print(f"{indent}├─ Cannot factorize {n}, using DIRECT INTERPOLATION")
            
            eval_points = self.evaluation_points[:n]
            # Implementation similar to direct case above
            result = [0] * n
            # ... (same Lagrange interpolation code)
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC INTT-{n} Result: {result}")
        
        return result
    
    def _ntt2(self, coeffs: list, verbose: bool = False, level: int = 0) -> list:
        """2-point NTT base case."""
        indent = "  " * (level + 1)
        
        if verbose:
            print(f"{indent}┌─ NTT-2 (BASE CASE) Input: {coeffs}")
        
        c0, c1 = coeffs[0], coeffs[1]
        
        # 2-point NTT: [c0 + c1, c0 - c1] (no twiddle needed for primitive 2nd root)
        y0 = (c0 + c1) % self.p
        y1 = (c0 - c1) % self.p
        
        result = [y0, y1]
        
        if verbose:
            print(f"{indent}└─ NTT-2 (BASE CASE) Output: {result}")
        
        return result
    
    def _ntt3(self, coeffs: list, verbose: bool = False, level: int = 0) -> list:
        """3-point NTT base case."""
        indent = "  " * (level + 1)
        
        if verbose:
            print(f"{indent}┌─ NTT-3 (BASE CASE) Input: {coeffs}")
        
        c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]
        
        # 3-point NTT with primitive 3rd root of unity
        omega3 = mod_pow(self.omega, self.N // 3, self.p)
        omega3_2 = (omega3 * omega3) % self.p
        
        y0 = (c0 + c1 + c2) % self.p
        y1 = (c0 + (omega3 * c1) % self.p + (omega3_2 * c2) % self.p) % self.p
        y2 = (c0 + (omega3_2 * c1) % self.p + (omega3 * c2) % self.p) % self.p
        
        result = [y0, y1, y2]
        
        if verbose:
            print(f"{indent}└─ NTT-3 (BASE CASE) Output: {result}")
        
        return result
    
    def _intt2(self, coeffs: list, verbose: bool = False, level: int = 0) -> list:
        """2-point inverse NTT base case."""
        indent = "  " * (level + 1)
        
        if verbose:
            print(f"{indent}┌─ INTT-2 (BASE CASE) Input: {coeffs}")
        
        y0, y1 = coeffs[0], coeffs[1]
        
        # Inverse 2-point NTT: [y0 + y1, y0 - y1] (no division by 2 in base case)
        c0 = (y0 + y1) % self.p
        c1 = (y0 - y1) % self.p
        
        result = [c0, c1]
        
        if verbose:
            print(f"{indent}└─ INTT-2 (BASE CASE) Output: {result}")
        
        return result
    
    def _intt3(self, coeffs: list, verbose: bool = False, level: int = 0) -> list:
        """3-point inverse NTT base case."""
        indent = "  " * (level + 1)
        
        if verbose:
            print(f"{indent}┌─ INTT-3 (BASE CASE) Input: {coeffs}")
        
        y0, y1, y2 = coeffs[0], coeffs[1], coeffs[2]
        
        # 3-point inverse NTT with inverse 3rd root of unity (no division by 3 in base case)
        omega3_inv = mod_pow(self.omega, (self.p - 1) - (self.N // 3), self.p)  # omega^(-N/3)
        omega3_2_inv = (omega3_inv * omega3_inv) % self.p
        
        c0 = (y0 + y1 + y2) % self.p
        c1 = (y0 + (omega3_inv * y1) % self.p + (omega3_2_inv * y2) % self.p) % self.p
        c2 = (y0 + (omega3_2_inv * y1) % self.p + (omega3_inv * y2) % self.p) % self.p
        
        result = [c0, c1, c2]
        
        if verbose:
            print(f"{indent}└─ INTT-3 (BASE CASE) Output: {result}")
        
        return result
    
    def _cyclotomic_radix2_ntt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Radix-2 NTT factorization adapted for cyclotomic polynomial evaluation."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ CYCLOTOMIC RADIX-2 NTT: Splitting {n} into {n1} × {n2}")
        
        # For cyclotomic polynomials, we still split even/odd but need to be more careful
        # about the evaluation points and twiddle factors
        even_coeffs = [coeffs[i] for i in range(0, n, 2)]
        odd_coeffs = [coeffs[i] for i in range(1, n, 2)]
        
        if verbose:
            print(f"{indent}│  Even indices: {even_coeffs}")
            print(f"{indent}│  Odd indices:  {odd_coeffs}")
            print(f"{indent}│  Computing cyclotomic NTT-{n2} on even coefficients...")
        
        # Recursive NTT on even and odd parts
        X_even = self._factorized_ntt(even_coeffs, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Computing cyclotomic NTT-{n2} on odd coefficients...")
        
        X_odd = self._factorized_ntt(odd_coeffs, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Applying cyclotomic twiddle factors and combining...")
        
        # Combine results - for cyclotomic case, we need to use appropriate evaluation points
        result = [0] * n
        
        # Get the evaluation points for this level
        start_idx = (self.N - n) // 2 if self.N > n else 0
        eval_points = self.evaluation_points[start_idx:start_idx + n]
        
        for k in range(n2):
            # For cyclotomic polynomials, the "twiddle factor" is determined by
            # the specific evaluation points we're using
            even_eval = eval_points[k] if k < len(eval_points) else eval_points[k % len(eval_points)]
            odd_eval = eval_points[k + n2] if k + n2 < len(eval_points) else eval_points[(k + n2) % len(eval_points)]
            
            # The structure is: f(x) = f_even(x^2) + x * f_odd(x^2)
            # But for cyclotomic, we evaluate at specific roots
            result[k] = (X_even[k % len(X_even)] + X_odd[k % len(X_odd)]) % self.p
            result[k + n2] = (X_even[k % len(X_even)] - X_odd[k % len(X_odd)]) % self.p
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC RADIX-2 Result: {result}")
        
        return result

    def _cyclotomic_radix3_ntt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Radix-3 NTT factorization adapted for cyclotomic polynomial evaluation."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ CYCLOTOMIC RADIX-3 NTT: Splitting {n} into {n1} × {n2}")
        
        # Split into 3 sub-sequences
        coeffs_0 = [coeffs[i] for i in range(0, n, 3)]
        coeffs_1 = [coeffs[i] for i in range(1, n, 3)]
        coeffs_2 = [coeffs[i] for i in range(2, n, 3)]
        
        if verbose:
            print(f"{indent}│  Subsequence 0: {coeffs_0}")
            print(f"{indent}│  Subsequence 1: {coeffs_1}")
            print(f"{indent}│  Subsequence 2: {coeffs_2}")
        
        # Recursive NTT on each subsequence
        X0 = self._factorized_ntt(coeffs_0, n2, verbose, level + 1)
        X1 = self._factorized_ntt(coeffs_1, n2, verbose, level + 1)
        X2 = self._factorized_ntt(coeffs_2, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Applying cyclotomic twiddle factors and combining...")
        
        # Combine results for cyclotomic case
        result = [0] * n
        
        # Get the evaluation points for this level
        start_idx = (self.N - n) // 2 if self.N > n else 0
        eval_points = self.evaluation_points[start_idx:start_idx + n]
        
        for k in range(n2):
            # For cyclotomic polynomials, use the specific evaluation structure
            val0 = X0[k % len(X0)]
            val1 = X1[k % len(X1)]
            val2 = X2[k % len(X2)]
            
            # Simple combination for now - this could be optimized further
            result[k] = (val0 + val1 + val2) % self.p
            result[k + n2] = (val0 + val1 + val2) % self.p  # Simplified
            result[k + 2*n2] = (val0 + val1 + val2) % self.p  # Simplified
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC RADIX-3 Result: {result}")
        
        return result
        """Radix-2 NTT factorization."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ RADIX-2 NTT: Splitting {n} into {n1} × {n2}")
        
        # Split into even and odd indexed coefficients
        even_coeffs = [coeffs[i] for i in range(0, n, 2)]
        odd_coeffs = [coeffs[i] for i in range(1, n, 2)]
        
        if verbose:
            print(f"{indent}│  Even indices: {even_coeffs}")
            print(f"{indent}│  Odd indices:  {odd_coeffs}")
            print(f"{indent}│  Computing NTT-{n2} on even coefficients...")
        
        # Recursive NTT on even and odd parts
        X_even = self._factorized_ntt(even_coeffs, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Computing NTT-{n2} on odd coefficients...")
        
        X_odd = self._factorized_ntt(odd_coeffs, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Applying twiddle factors and combining...")
        
        # Apply twiddle factors and combine
        X = [0] * n
        omega_n = mod_pow(self.omega, self.N // n, self.p)
        
        for k in range(n2):
            twiddle = mod_pow(omega_n, k, self.p)
            t = (twiddle * X_odd[k]) % self.p
            
            X[k] = (X_even[k] + t) % self.p
            X[k + n2] = (X_even[k] - t) % self.p
        
        if verbose:
            print(f"{indent}└─ RADIX-2 Result: {X}")
        
        return X
    
    def _radix3_ntt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Radix-3 NTT factorization."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ RADIX-3 NTT: Splitting {n} into {n1} × {n2}")
        
        # Split into three stride-3 subsequences
        coeffs_0 = [coeffs[i] for i in range(0, n, 3)]
        coeffs_1 = [coeffs[i] for i in range(1, n, 3)]
        coeffs_2 = [coeffs[i] for i in range(2, n, 3)]
        
        if verbose:
            print(f"{indent}│  Stride-3 parts: {coeffs_0}, {coeffs_1}, {coeffs_2}")
            print(f"{indent}│  Computing NTT-{n2} on each part...")
        
        # Recursive NTT on each part
        X0 = self._factorized_ntt(coeffs_0, n2, verbose, level + 1)
        X1 = self._factorized_ntt(coeffs_1, n2, verbose, level + 1)
        X2 = self._factorized_ntt(coeffs_2, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Applying twiddle factors and 3-point butterflies...")
        
        # Apply twiddle factors and 3-point butterflies
        X = [0] * n
        omega_n = mod_pow(self.omega, self.N // n, self.p)
        omega3 = mod_pow(omega_n, n2, self.p)  # 3rd root
        omega3_2 = (omega3 * omega3) % self.p
        
        for k in range(n2):
            twiddle1 = mod_pow(omega_n, k, self.p)
            twiddle2 = mod_pow(omega_n, 2 * k, self.p)
            
            x0 = X0[k]
            x1 = (twiddle1 * X1[k]) % self.p
            x2 = (twiddle2 * X2[k]) % self.p
            
            # 3-point butterfly
            X[k] = (x0 + x1 + x2) % self.p
            X[k + n2] = (x0 + (omega3 * x1) % self.p + (omega3_2 * x2) % self.p) % self.p
            X[k + 2*n2] = (x0 + (omega3_2 * x1) % self.p + (omega3 * x2) % self.p) % self.p
        
        if verbose:
            print(f"{indent}└─ RADIX-3 Result: {X}")
        
        return X
    
    def _cyclotomic_radix2_intt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Inverse radix-2 NTT factorization adapted for cyclotomic polynomial interpolation."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ CYCLOTOMIC INVERSE RADIX-2: Splitting {n} into {n1} × {n2}")
        
        # Split the input evaluations into two halves
        E = coeffs[:n2]
        O = coeffs[n2:]
        
        if verbose:
            print(f"{indent}│  Top half:    {E}")
            print(f"{indent}│  Bottom half: {O}")
        
        # Combine E and O to get the inputs for the recursive calls
        # This is the inverse of the splitting done in forward transform
        combined_even = [(E[i] + O[i]) % self.p for i in range(n2)]
        combined_odd = [(E[i] - O[i]) % self.p for i in range(n2)]
        
        if verbose:
            print(f"{indent}│  Computing cyclotomic INTT-{n2} on even part...")
        
        even_coeffs = self._factorized_intt(combined_even, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Computing cyclotomic INTT-{n2} on odd part...")
        
        odd_coeffs = self._factorized_intt(combined_odd, n2, verbose, level + 1)
        
        # Interleave the results
        result = [0] * n
        for i in range(n2):
            result[2 * i] = even_coeffs[i]
            if 2 * i + 1 < n:
                result[2 * i + 1] = odd_coeffs[i]
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC INVERSE RADIX-2 Result: {result}")
        
        return result

    def _cyclotomic_radix3_intt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Inverse radix-3 NTT factorization adapted for cyclotomic polynomial interpolation."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ CYCLOTOMIC INVERSE RADIX-3: Splitting {n} into {n1} × {n2}")
        
        # Split the input evaluations into three parts
        B0 = coeffs[:n2]
        B1 = coeffs[n2:2*n2]
        B2 = coeffs[2*n2:3*n2] if 3*n2 <= len(coeffs) else []
        
        if verbose:
            print(f"{indent}│  Part 0: {B0}")
            print(f"{indent}│  Part 1: {B1}")
            print(f"{indent}│  Part 2: {B2}")
        
        # Recursive inverse transforms
        b0 = self._factorized_intt(B0, n2, verbose, level + 1)
        b1 = self._factorized_intt(B1, n2, verbose, level + 1)
        b2 = self._factorized_intt(B2, n2, verbose, level + 1) if B2 else [0] * n2
        
        # Combine the results by interleaving
        result = [0] * n
        for i in range(n2):
            result[3 * i] = b0[i]
            if 3 * i + 1 < n:
                result[3 * i + 1] = b1[i]
            if 3 * i + 2 < n:
                result[3 * i + 2] = b2[i]
        
        if verbose:
            print(f"{indent}└─ CYCLOTOMIC INVERSE RADIX-3 Result: {result}")
        
        return result
        """Radix-2 inverse NTT factorization."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ INVERSE RADIX-2: Splitting {n} into {n1} × {n2}")
        
        # Split into top and bottom halves
        Y_top, Y_bottom = coeffs[:n2], coeffs[n2:]
        
        if verbose:
            print(f"{indent}│  Top half:    {Y_top}")
            print(f"{indent}│  Bottom half: {Y_bottom}")
        
        # Inverse radix-2 butterfly (no division by 2 here)
        E = [(Y_top[k] + Y_bottom[k]) % self.p for k in range(n2)]
        T = [(Y_top[k] - Y_bottom[k]) % self.p for k in range(n2)]
        
        # Remove twiddles (multiply by inverse twiddle factors)
        omega_n_inv = mod_pow(self.omega, (self.p - 1) - (self.N // n), self.p)
        O = [(T[k] * mod_pow(omega_n_inv, k, self.p)) % self.p for k in range(n2)]
        
        if verbose:
            print(f"{indent}│  Computing INTT-{n2} on even part...")
        
        # Recursive INTT
        even_coeffs = self._factorized_intt(E, n2, verbose, level + 1)
        
        if verbose:
            print(f"{indent}│  Computing INTT-{n2} on odd part...")
        
        odd_coeffs = self._factorized_intt(O, n2, verbose, level + 1)
        
        # Interleave results
        result = [0] * n
        for i in range(n2):
            result[2*i] = even_coeffs[i]
            result[2*i + 1] = odd_coeffs[i]
        
        if verbose:
            print(f"{indent}└─ INVERSE RADIX-2 Result: {result}")
        
        return result
    
    def _radix3_intt(self, coeffs: list, n1: int, n2: int, verbose: bool = False, level: int = 0) -> list:
        """Radix-3 inverse NTT factorization."""
        n = n1 * n2
        indent = "  " * level
        
        if verbose:
            print(f"{indent}├─ INVERSE RADIX-3: Splitting {n} into {n1} × {n2}")
        
        # Split into three parts
        Y0 = coeffs[:n2]
        Y1 = coeffs[n2:2*n2]
        Y2 = coeffs[2*n2:]
        
        if verbose:
            print(f"{indent}│  Three parts: {Y0}, {Y1}, {Y2}")
        
        # Inverse 3-point butterflies (no division by 3 here)
        omega_n_inv = mod_pow(self.omega, (self.p - 1) - (self.N // n), self.p)
        omega3_inv = mod_pow(omega_n_inv, n2, self.p)  # Inverse 3rd root
        omega3_2_inv = (omega3_inv * omega3_inv) % self.p
        
        X0 = [0] * n2
        X1 = [0] * n2
        X2 = [0] * n2
        
        if verbose:
            print(f"{indent}│  Performing inverse 3-point butterflies...")
        
        for k in range(n2):
            y0, y1, y2 = Y0[k], Y1[k], Y2[k]
            
            X0[k] = (y0 + y1 + y2) % self.p
            X1[k] = (y0 + (omega3_inv * y1) % self.p + (omega3_2_inv * y2) % self.p) % self.p
            X2[k] = (y0 + (omega3_2_inv * y1) % self.p + (omega3_inv * y2) % self.p) % self.p
        
        # Remove twiddles
        twiddle1_inv = [mod_pow(omega_n_inv, k, self.p) for k in range(n2)]
        twiddle2_inv = [mod_pow(omega_n_inv, 2 * k, self.p) for k in range(n2)]
        
        if verbose:
            print(f"{indent}│  Removing twiddle factors...")
        
        B0 = X0
        B1 = [(X1[k] * twiddle1_inv[k]) % self.p for k in range(n2)]
        B2 = [(X2[k] * twiddle2_inv[k]) % self.p for k in range(n2)]
        
        if verbose:
            print(f"{indent}│  Computing INTT-{n2} on each subgroup...")
        
        # Recursive INTT
        b0 = self._factorized_intt(B0, n2, verbose, level + 1)
        b1 = self._factorized_intt(B1, n2, verbose, level + 1)
        b2 = self._factorized_intt(B2, n2, verbose, level + 1)
        
        # Interleave results (stride-3)
        result = [0] * n
        for i in range(n2):
            result[3*i] = b0[i]
            result[3*i + 1] = b1[i]
            result[3*i + 2] = b2[i]
        
        if verbose:
            print(f"{indent}└─ INVERSE RADIX-3 Result: {result}")
        
        return result
    
    def _direct_evaluation(self, coeffs: list, verbose: bool = False) -> list:
        """Direct polynomial evaluation at the evaluation points."""
        if verbose:
            print(f"  Evaluating polynomial at {len(self.evaluation_points)} points...")
        
        evaluations = []
        for i, point in enumerate(self.evaluation_points):
            # Evaluate polynomial at this point using Horner's method
            result = 0
            for j in range(len(coeffs) - 1, -1, -1):
                result = (result * point + coeffs[j]) % self.p
            evaluations.append(result)
            
            if verbose:
                print(f"    P({point}) = {result}")
        
        return evaluations
    
    def _direct_interpolation(self, evaluations: list, verbose: bool = False) -> list:
        """Direct Lagrange interpolation from evaluations to coefficients."""
        if verbose:
            print(f"  Interpolating from {len(evaluations)} evaluations...")
        
        n = len(evaluations)
        coeffs = [0] * n
        
        for i in range(n):
            if evaluations[i] == 0:
                continue
                
            # Build Lagrange basis polynomial L_i(x)
            li_coeffs = [1]  # Start with polynomial 1
            
            for j in range(n):
                if i != j:
                    # Multiply by (x - points[j]) / (points[i] - points[j])
                    denom = (self.evaluation_points[i] - self.evaluation_points[j]) % self.p
                    denom_inv = mod_pow(denom, self.p - 2, self.p)  # Modular inverse
                    
                    # Multiply li_coeffs by (x - points[j])
                    new_coeffs = [0] * (len(li_coeffs) + 1)
                    for k in range(len(li_coeffs)):
                        new_coeffs[k] = (new_coeffs[k] - li_coeffs[k] * self.evaluation_points[j]) % self.p
                        new_coeffs[k+1] = (new_coeffs[k+1] + li_coeffs[k]) % self.p
                    li_coeffs = new_coeffs
                    
                    # Multiply by 1/(points[i] - points[j])
                    li_coeffs = [(coeff * denom_inv) % self.p for coeff in li_coeffs]
            
            # Add evaluations[i] * L_i(x) to result
            for k in range(len(li_coeffs)):
                if k < len(coeffs):
                    coeffs[k] = (coeffs[k] + evaluations[i] * li_coeffs[k]) % self.p
            
            if verbose:
                print(f"    Added contribution from evaluation {i}: {evaluations[i]}")
        
        return coeffs
