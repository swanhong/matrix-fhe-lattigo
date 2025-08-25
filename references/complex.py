from cyclotomic_dft import CyclotomicDFT
import numpy as np

def complex_poly_mult(p1: np.ndarray, p2: np.ndarray, N: int, verbose: bool = False) -> np.ndarray:
    """
    Computes the product of two polynomials in the cyclotomic ring 
    C[X]/(X^N - X^{N/2} + 1) using the generalized Cyclotomic DFT.

    Args:
        p1: Coefficients of the first polynomial.
        p2: Coefficients of the second polynomial.
        N: The degree of the cyclotomic polynomial.
        verbose: If True, prints detailed error information.

    Returns:
        The coefficients of the resulting polynomial product.
    """
    if len(p1) != N or len(p2) != N:
        raise ValueError(f"Input polynomials must have length {N}")

    try:
        dft_handler = CyclotomicDFT(N)
    except ValueError as e:
        if verbose:
            print(f"Failed to initialize CyclotomicDFT for N={N}: {e}")
        # Fallback or re-raise, depending on desired behavior
        raise e

    # Forward transform
    dft1 = dft_handler.dft(p1)
    dft2 = dft_handler.dft(p2)

    # Point-wise product
    dft_prod = dft1 * dft2

    # Inverse transform
    product = dft_handler.idft(dft_prod)

    return product

def test_cyclotomic_dft(N: int, num_tests: int = 1, verbose: bool = False) -> bool:
    """Test the basic DFT implementation."""
    try:
        dft_handler = CyclotomicDFT(N)
        
        for i in range(num_tests):
            # Generate random polynomial
            coeffs = np.random.random(N) + 1j * np.random.random(N)
            
            # Forward and inverse transform
            dft_coeffs = dft_handler.dft(coeffs)
            recovered = dft_handler.idft(dft_coeffs)
            
            # Check error
            error = np.max(np.abs(coeffs - recovered))
            if error > 1e-9: # Relaxed tolerance for FFT
                if verbose:
                    print(f"Test {i+1} failed with error {error}")
                return False
                
        if verbose:
            print(f"All {num_tests} tests passed")
        return True
        
    except Exception as e:
        if verbose:
            print(f"Test failed with exception: {e}")
        return False


def benchmark_cyclotomic_dft(N: int, num_runs: int = 100, verbose: bool = False) -> float:
    """Benchmark the DFT implementation."""
    import time
    
    try:
        dft_handler = CyclotomicDFT(N)
        coeffs = np.random.random(N) + 1j * np.random.random(N)
        
        # Warmup
        for _ in range(5):
            dft_coeffs = dft_handler.dft(coeffs)
            dft_handler.idft(dft_coeffs)
            
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            dft_coeffs = dft_handler.dft(coeffs)
            dft_handler.idft(dft_coeffs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        if verbose:
            print(f"Average time for N={N}: {avg_time*1000:.3f} ms")
            
        return avg_time
        
    except Exception as e:
        if verbose:
            print(f"Benchmark failed with exception: {e}")
        return float('inf')

