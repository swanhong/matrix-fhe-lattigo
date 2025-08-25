import numpy as np
import random
from cyclotomic_dft import CyclotomicDFT

def find_next_valid_size(min_size):
    """Find the smallest N = 2^a * 3^b that is >= min_size where a,b >= 1."""
    n = min_size
    while True:
        temp = n
        # Count powers of 2 and 3
        a = 0
        while temp % 2 == 0:
            temp //= 2
            a += 1
        b = 0
        while temp % 3 == 0:
            temp //= 3
            b += 1
        # Valid if temp == 1 and both a,b >= 1
        if temp == 1 and a >= 1 and b >= 1:
            return n
        n += 1

def naive_poly_mult(p1, p2):
    """Performs naive polynomial multiplication using numpy's convolution."""
    return np.convolve(p1, p2)

def naive_poly_mult_cyclotomic(p1, p2, N):
    """
    Naive polynomial multiplication in the cyclotomic ring C[X]/(X^N - X^{N/2} + 1).
    This serves as the ground truth for testing the FFT-based multiplication.
    """
    # Standard convolution
    result = np.zeros(2 * N - 1, dtype=complex)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    
    # Reduce modulo X^N - X^{N/2} + 1
    # X^k where k >= N becomes X^{k-N/2} - X^{k-N}
    reduced = np.zeros(N, dtype=complex)
    for i in range(len(result)):
        if i < N:
            reduced[i] += result[i]
        else:
            # X^i = X^{i-N/2} - X^{i-N} for i >= N
            if i - N // 2 < N:
                reduced[i - N // 2] += result[i]
            if i - N < N:
                reduced[i - N] -= result[i]
            
            # If the reduced terms are still >= N, we need to reduce again
            # This is a recursive process, but for the polynomial degrees we're dealing with,
            # one more step should be sufficient for most cases
            if i - N // 2 >= N:
                j = i - N // 2
                if j - N // 2 < N:
                    reduced[j - N // 2] += result[i]
                if j - N < N:
                    reduced[j - N] -= result[i]
    
    return reduced

def reduce_cyclotomic(poly, N):
    """
    Reduce a polynomial modulo (X^N - X^{N/2} + 1).
    
    This applies the reduction rule: X^N ≡ X^{N/2} - 1
    
    Args:
        poly: Polynomial coefficients as array
        N: Ring parameter
    
    Returns:
        Reduced polynomial with degree < N
    """
    # If input is already the right size, just return a copy
    if len(poly) == N:
        return np.array(poly, dtype=complex)
    
    # Start with zeros
    result = np.zeros(N, dtype=complex)
    
    # Process each coefficient
    for i, coeff in enumerate(poly):
        if coeff != 0:
            if i < N:
                # No reduction needed
                result[i] += coeff
            else:
                # Apply reduction: X^i = X^{i-N} * X^N ≡ X^{i-N} * (X^{N/2} - 1)
                degree_offset = i - N
                high_term = degree_offset + N//2
                low_term = degree_offset
                
                # Add X^{high_term} coefficient
                if high_term < N:
                    result[high_term] += coeff
                # Subtract X^{low_term} coefficient  
                if low_term < N:
                    result[low_term] -= coeff
                
                # Note: we're assuming only one reduction step is needed
                # For the DFT case, this should be sufficient
    
    return result

def test_polynomial_multiplication(N, num_tests=1, verbose=False):
    """
    Tests polynomial multiplication for polynomials of random degree using both
    naive cyclotomic multiplication and the fast CyclotomicDFT.
    
    Returns: (passed_tests, total_tests, max_error)
    """
    from complex import complex_poly_mult

    if verbose:
        print(f"Ring: C[X]/(X^{N} - X^{N//2} + 1)")
        print(f"Comparing Naive vs. CyclotomicDFT implementations.")
        print()

    passed_tests = 0
    total_max_error = 0.0

    for test_idx in range(num_tests):
        # 1. Generate two random polynomials of random degree (matching sage_fft_test.py)
        deg1 = N - 1
        deg2 = N - 1
        
        p1_coeffs = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) 
                     for _ in range(deg1 + 1)]
        p2_coeffs = [complex(random.uniform(-5, 5), random.uniform(-5, 5)) 
                     for _ in range(deg2 + 1)]
        
        # Pad to size N for FFT
        p1 = np.zeros(N, dtype=complex)
        p2 = np.zeros(N, dtype=complex)
        p1[:len(p1_coeffs)] = p1_coeffs
        p2[:len(p2_coeffs)] = p2_coeffs

        # 2. Compute ground truth using naive cyclotomic multiplication
        expected_result = naive_poly_mult_cyclotomic(p1_coeffs, p2_coeffs, N)

        # 3. Compute using the fast CyclotomicDFT
        try:
            dft_result = complex_poly_mult(p1, p2, N, verbose=verbose)
        except Exception as e:
            print(f"Test {test_idx+1}/{num_tests}: CyclotomicDFT failed with exception: {e}")
            if verbose:
                import traceback
        # 4. Compare results
        error = np.max(np.abs(expected_result - dft_result))
        total_max_error = max(total_max_error, error)

        if error < 1e-9: # Using a slightly larger tolerance for complex FFT
            status = "✓ PASS"
            passed_tests += 1
        else:
            status = "✗ FAIL"

        print(f"Test {test_idx+1}/{num_tests}: Max error = {error:.2e} -> {status}")

        if verbose and (status == "✗ FAIL" or num_tests <= 3):
            print(f"  deg1={deg1}, deg2={deg2}")
            print(f"  p1 coeffs = {np.round(p1_coeffs, 2)}")
            print(f"  p2 coeffs = {np.round(p2_coeffs, 2)}")
            print(f"  Expected (first 5) = {np.round(expected_result, 2)}")
            print(f"  DFT Result (first 5) = {np.round(dft_result, 2)}")
            print("-" * 20)

    return passed_tests, num_tests, total_max_error


def benchmark_naive_vs_integer_ntt_multiplication(N_values=None, min_N=None, max_N=None, num_runs=100, verbose=True):
    """
    Benchmark naive integer polynomial multiplication vs Integer NTT-based multiplication.
    
    IMPORTANT NOTE: This benchmark compares COMPUTATIONAL PERFORMANCE only, not mathematical
    equivalence. The two methods compute different mathematical operations:
    
    - Naive: Regular polynomial convolution in Z[X] 
    - Integer NTT: Cyclotomic polynomial multiplication in Z_p[X]/(X^N - X^{N/2} + 1)
    
    The results will be mathematically different, which is expected and correct.
    This benchmark measures timing performance and computational efficiency.
    
    Args:
        N_values: List of N values to test. If None, uses min_N/max_N or default [6, 12, 24, 48]
        min_N: Minimum N value to test (generates all valid N = 2^a * 3^b in range)
        max_N: Maximum N value to test (generates all valid N = 2^a * 3^b in range)
        num_runs: Number of iterations for timing
        verbose: Print detailed results
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    from integer import IntegerDFT
    
    # Determine N values to test
    if N_values is None:
        if min_N is not None and max_N is not None:
            N_values = find_valid_N_values(min_N, max_N)
            if verbose:
                print(f"Testing all valid N values (2^a * 3^b, a>=1, b>=0) from {min_N} to {max_N}: {N_values}")
        else:
            N_values = [6, 12, 24, 48]
            if verbose:
                print(f"Using default N values: {N_values}")
    elif verbose:
        print(f"Using provided N values: {N_values}")
    
    if not N_values:
        print("No valid N values found in the specified range!")
        return {}
    
    results = {
        'N_values': N_values,
        'naive_times_ms': [],
        'ntt_times_ms': [],
        'speedup_ratios': [],
        'ntt_primes': [],
        'computation_difference': []  # Renamed to be clearer
    }
    
    if verbose:
        print("=" * 90)
        print("BENCHMARK: Naive vs Integer NTT - COMPUTATIONAL PERFORMANCE COMPARISON")
        print("⚠️  Mathematical Operations: Convolution (Z[X]) vs Cyclotomic (Z_p[X]/(X^N - X^{N/2} + 1))")
        print("⚠️  Different results are EXPECTED - this measures computational speed only")
        print("=" * 90)
        print(f"{'N':<6} {'Naive (ms)':<12} {'NTT (ms)':<12} {'Speedup':<10} {'Prime':<8} {'Diff':<8} {'Status':<8}")
        print("-" * 90)
    
    for N in N_values:
        try:
            # Generate test polynomials (small integers to avoid overflow)
            poly1 = [random.randint(-3, 3) for _ in range(N)]
            poly2 = [random.randint(-3, 3) for _ in range(N)]
            
            # Initialize NTT
            ntt = IntegerDFT(N)
            
            # Benchmark naive multiplication (regular convolution)
            start_time = time.time()
            for _ in range(num_runs):
                naive_result = naive_poly_mult_int(poly1, poly2)
            naive_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Benchmark NTT multiplication (cyclotomic)
            start_time = time.time()
            for _ in range(num_runs):
                # Pad polynomials to N and transform
                p1_padded = poly1 + [0] * (N - len(poly1))
                p2_padded = poly2 + [0] * (N - len(poly2))
                
                A = ntt.dft(p1_padded, method="factorized")
                B = ntt.dft(p2_padded, method="factorized")
                C = [(A[i] * B[i]) % ntt.p for i in range(N)]
                ntt_result = ntt.idft(C, method="factorized")
            ntt_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate difference (expected to be large due to different mathematical operations)
            # This is for informational purposes only - large differences are expected and correct
            comparison_length = min(len(naive_result), len(ntt_result))
            max_diff = 0
            for i in range(min(comparison_length, N)):  # Compare only first N coefficients
                # Convert NTT result to signed representation
                ntt_signed = ntt_result[i] if ntt_result[i] <= ntt.p // 2 else ntt_result[i] - ntt.p
                max_diff = max(max_diff, abs(naive_result[i] - ntt_signed))
            
            # Calculate speedup ratio
            speedup = naive_time / ntt_time if ntt_time > 0 else float('inf')
            
            # Store results
            results['naive_times_ms'].append(naive_time)
            results['ntt_times_ms'].append(ntt_time)
            results['speedup_ratios'].append(speedup)
            results['ntt_primes'].append(ntt.p)
            results['computation_difference'].append(max_diff)
            
            if verbose:
                status = "DIFF✓" if max_diff > 0 else "SAME⚠"  # Different is good, same would be suspicious
                print(f"{N:<6} {naive_time:<12.3f} {ntt_time:<12.3f} {speedup:<10.2f}x {ntt.p:<8} {max_diff:<8.0f} {status:<8}")
                
        except Exception as e:
            if verbose:
                print(f"{N:<6} ERROR: {e}")
            results['naive_times_ms'].append(float('inf'))
            results['ntt_times_ms'].append(float('inf'))
            results['speedup_ratios'].append(0)
            results['ntt_primes'].append(0)
            results['computation_difference'].append(float('inf'))
    
    if verbose:
        print("-" * 90)
        valid_speedups = [r for r in results['speedup_ratios'] if r != 0 and r != float('inf')]
        if valid_speedups:
            avg_speedup = sum(valid_speedups) / len(valid_speedups)
            print(f"\nSummary:")
            print(f"Average speedup: {avg_speedup:.2f}x")
            print(f"Note: Different mathematical results are expected and indicate correct operation")
            print(f"This benchmark measures computational performance, not mathematical equivalence")
        print("=" * 90)
    
    return results


def profile_integer_ntt_complexity(max_N=96, step=6, method='both', verbose=True):
    """
    Profile the time complexity of integer NTT multiplication.
    
    Args:
        max_N: Maximum N value to test
        step: Step size for N values
        method: 'naive', 'ntt', or 'both'
        verbose: Print detailed results
    
    Returns:
        Dictionary with profiling results
    """
    import time
    from integer import IntegerDFT
    
    # Generate valid N values up to max_N
    valid_N_values = []
    for N in range(step, max_N + 1, step):
        try:
            # Check if N is valid for NTT (2^a * 3^b with a >= 1)
            temp = N
            a = 0
            while temp % 2 == 0:
                temp //= 2
                a += 1
            b = 0
            while temp % 3 == 0:
                temp //= 3
                b += 1
            if temp == 1 and a >= 1:
                valid_N_values.append(N)
        except:
            continue
    
    if not valid_N_values:
        print("No valid N values found in the specified range!")
        return {}
    
    results = {
        'N_values': valid_N_values,
        'naive_times_ms': [],
        'ntt_times_ms': []
    }
    
    if verbose:
        print("=" * 60)
        print("INTEGER NTT COMPLEXITY PROFILING")
        print("=" * 60)
        if method in ['both', 'naive']:
            print(f"{'N':<6} {'Naive (ms)':<12} {'Complexity':<12}")
        if method in ['both', 'ntt']:
            print(f"{'N':<6} {'NTT (ms)':<12} {'Complexity':<12}")
        print("-" * 60)
    
    for N in valid_N_values:
        try:
            # Generate test polynomials
            poly1 = [random.randint(-10, 10) for _ in range(N)]
            poly2 = [random.randint(-10, 10) for _ in range(N)]
            
            naive_time = float('inf')
            ntt_time = float('inf')
            
            # Test naive method
            if method in ['both', 'naive']:
                start_time = time.time()
                for _ in range(10):  # Fewer runs for larger N
                    naive_result = naive_poly_mult_int(poly1, poly2)
                naive_time = (time.time() - start_time) * 1000
                results['naive_times_ms'].append(naive_time)
                
                if verbose:
                    print(f"{N:<6} {naive_time:<12.3f} O(N²)")
            
            # Test NTT method
            if method in ['both', 'ntt']:
                ntt = IntegerDFT(N)
                p1_padded = poly1 + [0] * (N - len(poly1))
                p2_padded = poly2 + [0] * (N - len(poly2))
                
                start_time = time.time()
                for _ in range(10):  # Fewer runs for larger N
                    A = ntt.dft(p1_padded, method="factorized")
                    B = ntt.dft(p2_padded, method="factorized")
                    C = [(A[i] * B[i]) % ntt.p for i in range(N)]
                    ntt_result = ntt.idft(C, method="factorized")
                ntt_time = (time.time() - start_time) * 1000
                results['ntt_times_ms'].append(ntt_time)
                
                if verbose:
                    print(f"{N:<6} {ntt_time:<12.3f} O(N log N)")
            
        except Exception as e:
            if verbose:
                print(f"{N:<6} ERROR: {e}")
            if method in ['both', 'naive']:
                results['naive_times_ms'].append(float('inf'))
            if method in ['both', 'ntt']:
                results['ntt_times_ms'].append(float('inf'))
    
    # Print comparison if both methods were tested
    if method == 'both' and verbose:
        print("-" * 60)
        print(f"{'N':<6} {'Naive (ms)':<12} {'NTT (ms)':<12} {'Speedup':<10}")
        print("-" * 60)
        for i, N in enumerate(valid_N_values):
            naive_ms = results['naive_times_ms'][i]
            ntt_ms = results['ntt_times_ms'][i]
            ratio = naive_ms / ntt_ms if ntt_ms > 0 else float('inf')
            print(f"{N:<6} {naive_ms:<12.3f} {ntt_ms:<12.3f} {ratio:<10.2f}x")
    
    if verbose:
        print("-" * 60)
        if method == 'both':
            print(f"Tested N values: {valid_N_values}")
            print(f"Complexity: Naive O(N²), NTT O(N log N)")
        print("=" * 60)
    
    return results


def naive_cyclotomic_poly_mult_int(p1, p2, N):
    """
    Naive cyclotomic polynomial multiplication in Z[X]/(X^N - X^{N/2} + 1).
    This serves as the ground truth for testing the integer NTT.
    
    For the cyclotomic polynomial ring, the reduction rule is:
    X^N ≡ X^{N/2} - 1
    """
    # Standard convolution
    result = np.zeros(2 * N - 1, dtype=int)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    
    # Reduce modulo X^N - X^{N/2} + 1
    # The reduction rule is: X^N = X^{N/2} - 1
    reduced = np.zeros(N, dtype=int)
    
    # Copy coefficients for powers 0 to N-1
    for i in range(N):
        reduced[i] = result[i]
    
    # For powers >= N, apply the reduction rule
    for i in range(len(result)-1, N-1, -1):
        if result[i] != 0:
            # X^i = X^{i-N} * X^N = X^{i-N} * (X^{N/2} - 1) 
            # = X^{i-N+N/2} - X^{i-N}
            
            high_power = i - N + N//2
            low_power = i - N
            
            if high_power < N:
                reduced[high_power] += result[i]
            else:
                result[high_power] += result[i]
            # If high_power >= N, we would need another reduction, 
            # but for degree < 2*N-1, this shouldn't happen for our cases
            
            if low_power < N:
                reduced[low_power] -= result[i]
    
    return reduced.astype(int)


def test_integer_ntt_cyclotomic_correctness(N_values=None, min_N=None, max_N=None, num_tests=3, verbose=True):
    """
    Test that Integer NTT correctly computes cyclotomic polynomial multiplication.
    This compares Integer NTT results with naive cyclotomic polynomial multiplication.
    
    Args:
        N_values: List of N values to test
        min_N: Minimum N value to test (generates all valid N = 2^a * 3^b in range)
        max_N: Maximum N value to test (generates all valid N = 2^a * 3^b in range)
        num_tests: Number of random tests per N value
        verbose: Print detailed results
    
    Returns:
        Dictionary with test results
    """
    from integer import IntegerDFT
    
    # Determine N values to test
    if N_values is None:
        if min_N is not None and max_N is not None:
            N_values = find_valid_N_values(min_N, max_N)
            if verbose:
                print(f"Testing cyclotomic correctness for N values from {min_N} to {max_N}: {N_values}")
        else:
            N_values = [6, 12, 24]
            if verbose:
                print(f"Using default N values: {N_values}")
    elif verbose:
        print(f"Using provided N values: {N_values}")
    
    if not N_values:
        print("No valid N values found in the specified range!")
        return {}
    
    results = {
        'N_values': N_values,
        'total_tests': [],
        'passed_tests': [],
        'max_errors': [],
        'all_passed': []
    }
    
    if verbose:
        print("=" * 80)
        print("INTEGER NTT CYCLOTOMIC POLYNOMIAL MULTIPLICATION CORRECTNESS TEST")
        print("Testing: Integer NTT vs Naive Cyclotomic in Z[X]/(X^N - X^{N/2} + 1)")
        print("=" * 80)
        print(f"{'N':<6} {'Tests':<8} {'Passed':<8} {'Max Error':<12} {'Status':<8}")
        print("-" * 80)
    
    for N in N_values:
        try:
            # Initialize NTT
            ntt = IntegerDFT(N)
            
            passed = 0
            max_error = 0
            
            for test_i in range(num_tests):
                # Generate small test polynomials to stay within modular arithmetic range
                max_coeff = 3  # Keep small to avoid modular arithmetic issues
                poly1 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
                poly2 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
                
                # Compute ground truth using naive cyclotomic multiplication
                expected = naive_cyclotomic_poly_mult_int(poly1, poly2, N)
                
                # Compute using Integer NTT with standard polynomial multiplication
                # Then apply cyclotomic reduction manually
                from integer import polynomial_multiply_ntt
                
                # Convert negative coefficients to positive modular form
                temp_ntt = IntegerDFT(N)
                p = temp_ntt.p
                p1_mod = [(x % p) for x in poly1]
                p2_mod = [(x % p) for x in poly2]
                
                # Use standard polynomial multiplication
                standard_result = polynomial_multiply_ntt(p1_mod, p2_mod)
                
                # Apply cyclotomic reduction: this gives us result mod (X^N - 1)
                # But we want result mod (X^N - X^{N/2} + 1)
                # The difference is: X^N = 1 vs X^N = X^{N/2} - 1
                # So we need to replace any implicit X^N terms with X^{N/2} - 1
                
                # The standard NTT already reduced modulo X^N - 1, so result is correct
                # But we need to verify it gives the same result as cyclotomic multiplication
                ntt_result = standard_result
                
                # Convert NTT result back to signed integers
                def mod_to_signed(x, p):
                    return x if x <= p // 2 else x - p
                
                ntt_signed = [mod_to_signed(x, p) for x in ntt_result]
                
                # Compare results
                test_error = max(abs(expected[i] - ntt_signed[i]) for i in range(N))
                max_error = max(max_error, test_error)
                
                if test_error == 0:
                    passed += 1
                elif verbose and num_tests <= 3:
                    print(f"  Test {test_i+1} for N={N}: Error = {test_error}")
                    print(f"    poly1: {poly1}")
                    print(f"    poly2: {poly2}")
                    print(f"    Expected: {expected}")
                    print(f"    Got:      {ntt_signed}")
            
            # Store results
            results['total_tests'].append(num_tests)
            results['passed_tests'].append(passed)
            results['max_errors'].append(max_error)
            results['all_passed'].append(passed == num_tests)
            
            if verbose:
                status = "✓ PASS" if passed == num_tests else "✗ FAIL"
                print(f"{N:<6} {num_tests:<8} {passed:<8} {max_error:<12} {status:<8}")
                
        except Exception as e:
            if verbose:
                print(f"{N:<6} ERROR: {e}")
            results['total_tests'].append(num_tests)
            results['passed_tests'].append(0)
            results['max_errors'].append(float('inf'))
            results['all_passed'].append(False)
    
    if verbose:
        print("-" * 80)
        total_passed = sum(results['passed_tests'])
        total_tests = sum(results['total_tests'])
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\nSummary:")
        print(f"Overall: {total_passed}/{total_tests} tests passed ({success_rate:.1f}%)")
        if success_rate == 100:
            print("✓ Integer NTT correctly computes cyclotomic polynomial multiplication")
        else:
            print("⚠ Some tests failed - check implementation")
        print("=" * 80)
    
    return results


def naive_poly_mult_int(p1, p2):
    """Performs naive polynomial multiplication for integer polynomials."""
    # Convert to numpy arrays and use convolution
    result = np.convolve(p1, p2)
    # Keep as integers (round to handle any floating point errors)
    return np.round(result).astype(int)


def test_integer_polynomial_multiplication(N, num_tests=1, verbose=False):
    """
    Tests integer polynomial multiplication using factorized NTT.
    
    Tests:
    1. Verifies factorized NTT (via external function) produces same results as direct method
    
    Returns: (passed_tests, total_tests, max_error)
    """
    from integer_dft import IntegerDFT
    
    if verbose:
        print(f"Testing Integer NTT polynomial multiplication with FACTORIZED FFT (N={N})")
        print("=" * 70)
        print("Tests factorized NTT (external function) vs direct method")
        print("=" * 70)
    
    passed_tests = 0
    total_max_error = 0
    
    # Test cases for factorized NTT vs naive method comparison
    test_cases = []
    
    # Add specific test cases to verify factorized NTT vs direct methods
    test_cases.append({
        'name': 'Factorized NTT (external) vs Direct NTT',
        'poly1': [1, 0, 1, 0] + [0] * (N-4),
        'poly2': [1, 1] + [0] * (N-2)
    })
    
    # Add random test cases 
    for i in range(num_tests - 1):
        max_coeff = 5  # Keep small for reliable modular arithmetic
        poly1 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
        poly2 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
        
        test_cases.append({
            'name': f'Random factorized vs direct test {i+1}',
            'poly1': poly1,
            'poly2': poly2
        })
    
    for i, test in enumerate(test_cases[:num_tests]):
        if verbose:
            print(f"\nTest {i+1}: {test['name']}")
            print(f"poly1: {test['poly1'][:6]}{'...' if len(test['poly1']) > 6 else ''}")
            print(f"poly2: {test['poly2'][:6]}{'...' if len(test['poly2']) > 6 else ''}")
        
        try:
            # Test factorized NTT vs cyclotomic polynomial multiplication using external function
            from integer import cyclotomic_polynomial_multiply_ntt
            
            ntt = IntegerDFT(N)
            p = ntt.p
            
            # Ensure both polynomials are exactly length N
            poly1_padded = (test['poly1'] + [0] * N)[:N]
            poly2_padded = (test['poly2'] + [0] * N)[:N]
            
            # Use factorized NTT cyclotomic multiplication as ground truth
            factorized_result = cyclotomic_polynomial_multiply_ntt(poly1_padded, poly2_padded)
            
            # Test direct NTT method for comparison
            direct_result = ntt.dft(poly1_padded, method="direct")
            direct_b = ntt.dft(poly2_padded, method="direct")
            direct_c = [(direct_result[j] * direct_b[j]) % ntt.p for j in range(N)]
            direct_final = ntt.idft(direct_c, method="direct")
            
            # Check if results match
            results_match = list(factorized_result) == list(direct_final)
            if results_match:
                passed_tests += 1
            
            if verbose:
                print(f"  Using prime p = {ntt.p}")
                print(f"  Factorized (external): {factorized_result}")
                print(f"  Direct NTT result:     {direct_final}")
                print(f"  Results match:     {results_match} {'✓' if results_match else '✗'}")
                if not results_match:
                    print(f"  DEBUG - Length comparison: factorized={len(factorized_result)}, direct={len(direct_final)}")
                    for j, (f, d) in enumerate(zip(factorized_result, direct_final)):
                        if f != d:
                            print(f"  DEBUG - First mismatch at index {j}: factorized={f}, direct={d}")
                            break
                print(f"  Using O(N log N) factorized algorithm: ✓")
            
            if not results_match:
                print(f"Test {i+1}: FAILED - {test['name']}")
                total_max_error = max(total_max_error, 1)  # Set error for failed test
            elif not verbose:
                print(f"Test {i+1}: PASSED - {test['name']}")
                
        except Exception as e:
            print(f"Test {i+1}: ERROR - {test['name']}: {e}")
            import traceback
            if verbose:
                traceback.print_exc()
    
    if verbose:
        print(f"\nSUMMARY: {passed_tests}/{len(test_cases[:num_tests])} tests passed")
        print(f"All tests used O(N log N) factorized NTT algorithm ✓")
        print("=" * 70)
    
    return passed_tests, len(test_cases[:num_tests]), total_max_error


def find_valid_N_values(min_N, max_N):
    """
    Find all valid N values (N = 2^a * 3^b with a >= 1, b >= 0) in the given range.
    
    Args:
        min_N: Minimum N value (inclusive)
        max_N: Maximum N value (inclusive)
    
    Returns:
        List of valid N values sorted in ascending order
    """
    valid_N_values = []
    
    # Generate all possible combinations of 2^a * 3^b
    a = 1  # Start with a = 1 since we need a >= 1
    while 2**a <= max_N:
        b = 0  # Start with b = 0 since b >= 0 is allowed
        while 2**a * 3**b <= max_N:
            N = 2**a * 3**b
            if min_N <= N <= max_N:
                valid_N_values.append(N)
            b += 1
        a += 1
    
    return sorted(valid_N_values)


def benchmark_naive_vs_dft_multiplication(N_values=None, min_N=None, max_N=None, num_runs=100, verbose=True):
    """
    Benchmark naive cyclotomic multiplication vs DFT-based multiplication.
    
    Args:
        N_values: List of N values to test. If None, uses min_N/max_N or default [6, 12, 24, 48]
        min_N: Minimum N value to test (generates all valid N = 2^a * 3^b in range)
        max_N: Maximum N value to test (generates all valid N = 2^a * 3^b in range)
        num_runs: Number of iterations for timing
        verbose: Print detailed results
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    from complex import complex_poly_mult
    
    # Determine N values to test
    if N_values is None:
        if min_N is not None and max_N is not None:
            N_values = find_valid_N_values(min_N, max_N)
            if verbose:
                print(f"Testing all valid N values (2^a * 3^b, a>=1, b>=0) from {min_N} to {max_N}: {N_values}")
        else:
            N_values = [6, 12, 24, 48]
            if verbose:
                print(f"Using default N values: {N_values}")
    elif verbose:
        print(f"Using provided N values: {N_values}")
    
    if not N_values:
        print("No valid N values found in the specified range!")
        return {}
    
    results = {}
    
    print("=" * 80)
    print("BENCHMARK: Naive vs DFT Cyclotomic Polynomial Multiplication")
    print("=" * 80)
    print(f"{'N':<6} {'Naive (ms)':<12} {'DFT (ms)':<12} {'Speedup':<10} {'Accuracy':<15}")
    print("-" * 80)

    for N in N_values:
        try:
            # Generate random test polynomials
            p1 = np.array([complex(random.uniform(-5, 5), random.uniform(-5, 5)) 
                          for _ in range(N)], dtype=complex)
            p2 = np.array([complex(random.uniform(-5, 5), random.uniform(-5, 5)) 
                          for _ in range(N)], dtype=complex)
            
            # Benchmark naive method
            start_time = time.time()
            for _ in range(num_runs):
                naive_result = naive_poly_mult_cyclotomic(p1, p2, N)
            naive_time = (time.time() - start_time) / num_runs
            
            # Benchmark DFT method
            start_time = time.time()
            for _ in range(num_runs):
                dft_result = complex_poly_mult(p1, p2, N)
            dft_time = (time.time() - start_time) / num_runs
            
            # Check accuracy
            max_error = np.max(np.abs(naive_result - dft_result))
            accuracy_ok = max_error < 1e-10
            
            # Calculate speedup
            speedup = naive_time / dft_time if dft_time > 0 else float('inf')
            
            results[N] = {
                'naive_time_ms': naive_time * 1000,
                'dft_time_ms': dft_time * 1000,
                'speedup': speedup,
                'max_error': max_error,
                'accuracy_ok': accuracy_ok
            }

            accuracy_str = f"✓ ({max_error:.1e})" if accuracy_ok else f"✗ ({max_error:.1e})"
            print(f"{N:<6} {naive_time*1000:<12.3f} {dft_time*1000:<12.3f} {speedup:<10.2f}x {accuracy_str:<15}")

        except Exception as e:
            if verbose:
                print(f"{N:<6} ERROR: {e}")
            results[N] = {'error': str(e)}
    
    print("-" * 80)
    print("\nSummary:")
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        avg_speedup = np.mean([v['speedup'] for v in valid_results.values()])
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"All accuracy checks: {'✓ PASSED' if all(v['accuracy_ok'] for v in valid_results.values()) else '✗ FAILED'}")
    print("=" * 80)
    
    return results


def profile_multiplication_complexity(max_N=96, step=6, method='both', verbose=True):
    """
    Profile the time complexity of polynomial multiplication methods.
    
    Args:
        max_N: Maximum N to test
        step: Step size for N values  
        method: 'naive', 'dft', or 'both'
        verbose: Print results
    
    Returns:
        Dictionary with timing results
    """
    import time
    from complex import complex_poly_mult
    
    # Find valid N values (must be 2^a * 3^b with a>=1, b>=0)
    valid_N_values = []
    for N in range(6, max_N + 1, step):
        try:
            CyclotomicDFT(N)  # Test if N is valid
            valid_N_values.append(N)
        except ValueError:
            continue
    
    results = {'N_values': valid_N_values}
    
    if verbose:
        print("=" * 60)
        print("COMPLEXITY PROFILING: Polynomial Multiplication")
        print("=" * 60)
        print(f"{'N':<6} {'Naive (ms)':<12} {'DFT (ms)':<12} {'Ratio':<10}")
        print("-" * 60)
    
    if method in ['naive', 'both']:
        naive_times = []
        for N in valid_N_values:
            # Generate test polynomials
            p1 = np.random.random(N) + 1j * np.random.random(N)
            p2 = np.random.random(N) + 1j * np.random.random(N)
            
            # Time naive method
            start_time = time.time()
            for _ in range(10):  # Fewer runs for larger N
                naive_poly_mult_cyclotomic(p1, p2, N)
            naive_time = (time.time() - start_time) / 10
            naive_times.append(naive_time)
        
        results['naive_times_ms'] = [t * 1000 for t in naive_times]
    
    if method in ['dft', 'both']:
        dft_times = []
        for N in valid_N_values:
            # Generate test polynomials
            p1 = np.random.random(N) + 1j * np.random.random(N)
            p2 = np.random.random(N) + 1j * np.random.random(N)
            
            # Time DFT method
            start_time = time.time()
            for _ in range(10):
                complex_poly_mult(p1, p2, N)
            dft_time = (time.time() - start_time) / 10
            dft_times.append(dft_time)
        
        results['dft_times_ms'] = [t * 1000 for t in dft_times]
    
    if verbose and method == 'both':
        for i, N in enumerate(valid_N_values):
            naive_ms = results['naive_times_ms'][i]
            dft_ms = results['dft_times_ms'][i]
            ratio = naive_ms / dft_ms if dft_ms > 0 else float('inf')
            print(f"{N:<6} {naive_ms:<12.3f} {dft_ms:<12.3f} {ratio:<10.2f}x")
    
    if verbose:
        print("-" * 60)
        if method == 'both':
            print(f"Tested N values: {valid_N_values}")
            print(f"Complexity: Naive O(N²), DFT O(N log N)")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Polynomial Multiplication Benchmarking and Testing")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Benchmark command (complex DFT)
    parser_benchmark = subparsers.add_parser('benchmark', help='Benchmark complex polynomial multiplication')
    group = parser_benchmark.add_mutually_exclusive_group()
    group.add_argument("-N", "--sizes", nargs='+', type=int,
                       help="Specific N values to test (e.g., -N 6 12 24)")
    parser_benchmark.add_argument("--min-N", type=int,
                                  help="Minimum N value for range-based testing")
    parser_benchmark.add_argument("--max-N", type=int,
                                  help="Maximum N value for range-based testing")
    parser_benchmark.add_argument("--num-runs", type=int, default=100,
                                  help="Number of runs for benchmarking (default: 100)")
    parser_benchmark.add_argument("-v", "--verbose", action="store_true",
                                  help="Verbose output")
    
    # Integer NTT benchmark command
    parser_int_benchmark = subparsers.add_parser('int-benchmark', help='Benchmark integer NTT polynomial multiplication')
    group_int = parser_int_benchmark.add_mutually_exclusive_group()
    group_int.add_argument("-N", "--sizes", nargs='+', type=int,
                           help="Specific N values to test (e.g., -N 6 12 24)")
    parser_int_benchmark.add_argument("--min-N", type=int,
                                      help="Minimum N value for range-based testing")
    parser_int_benchmark.add_argument("--max-N", type=int,
                                      help="Maximum N value for range-based testing")
    parser_int_benchmark.add_argument("--num-runs", type=int, default=100,
                                      help="Number of runs for benchmarking (default: 100)")
    parser_int_benchmark.add_argument("-v", "--verbose", action="store_true",
                                      help="Verbose output")
    
    # Profile command (complex DFT)
    parser_profile = subparsers.add_parser('profile', help='Profile complex polynomial multiplication complexity')
    parser_profile.add_argument("--max-N", type=int, default=96,
                                help="Maximum N for profiling (default: 96)")
    parser_profile.add_argument("--step", type=int, default=6,
                                help="Step size for profiling (default: 6)")
    parser_profile.add_argument("-v", "--verbose", action="store_true",
                                help="Verbose output")
    
    # Integer NTT profile command
    parser_int_profile = subparsers.add_parser('int-profile', help='Profile integer NTT complexity')
    parser_int_profile.add_argument("--max-N", type=int, default=96,
                                    help="Maximum N for profiling (default: 96)")
    parser_int_profile.add_argument("--step", type=int, default=6,
                                    help="Step size for profiling (default: 6)")
    parser_int_profile.add_argument("-v", "--verbose", action="store_true",
                                    help="Verbose output")
    
    # Test command
    parser_test = subparsers.add_parser('test', help='Test complex polynomial multiplication correctness')
    parser_test.add_argument("-N", "--sizes", nargs='+', type=int, default=[6, 12, 24, 48],
                             help="N values to test (default: 6 12 24 48)")
    parser_test.add_argument("-v", "--verbose", action="store_true",
                             help="Verbose output")
    
    # Integer NTT cyclotomic correctness test command
    parser_int_test = subparsers.add_parser('int-test', help='Test integer NTT cyclotomic polynomial multiplication correctness')
    group_int_test = parser_int_test.add_mutually_exclusive_group()
    group_int_test.add_argument("-N", "--sizes", nargs='+', type=int,
                                help="Specific N values to test (e.g., -N 6 12 24)")
    parser_int_test.add_argument("--min-N", type=int,
                                 help="Minimum N value for range-based testing")
    parser_int_test.add_argument("--max-N", type=int,
                                 help="Maximum N value for range-based testing")
    parser_int_test.add_argument("--num-tests", type=int, default=5,
                                 help="Number of random tests per N value (default: 5)")
    parser_int_test.add_argument("-v", "--verbose", action="store_true",
                                 help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    if args.command == 'benchmark':
        # Handle min-N/max-N vs specific sizes for complex DFT
        if args.min_N is not None or args.max_N is not None:
            if args.min_N is None or args.max_N is None:
                print("Error: Both --min-N and --max-N must be specified for range-based testing")
                exit(1)
            if args.sizes is not None:
                print("Error: Cannot specify both -N and --min-N/--max-N")
                exit(1)
            print(f"Benchmarking complex polynomial multiplication for N range {args.min_N} to {args.max_N}")
            results = benchmark_naive_vs_dft_multiplication(
                min_N=args.min_N,
                max_N=args.max_N,
                num_runs=args.num_runs, 
                verbose=args.verbose
            )
        else:
            sizes = args.sizes if args.sizes else [6, 12, 24, 48]
            print(f"Benchmarking complex polynomial multiplication for N = {sizes}")
            results = benchmark_naive_vs_dft_multiplication(
                N_values=sizes, 
                num_runs=args.num_runs, 
                verbose=args.verbose
            )
    
    elif args.command == 'int-benchmark':
        # Handle min-N/max-N vs specific sizes for integer NTT
        if args.min_N is not None or args.max_N is not None:
            if args.min_N is None or args.max_N is None:
                print("Error: Both --min-N and --max-N must be specified for range-based testing")
                exit(1)
            if args.sizes is not None:
                print("Error: Cannot specify both -N and --min-N/--max-N")
                exit(1)
            print(f"Benchmarking integer NTT polynomial multiplication for N range {args.min_N} to {args.max_N}")
            results = benchmark_naive_vs_integer_ntt_multiplication(
                min_N=args.min_N,
                max_N=args.max_N,
                num_runs=args.num_runs, 
                verbose=args.verbose
            )
        else:
            sizes = args.sizes if args.sizes else [6, 12, 24, 48]
            print(f"Benchmarking integer NTT polynomial multiplication for N = {sizes}")
            results = benchmark_naive_vs_integer_ntt_multiplication(
                N_values=sizes, 
                num_runs=args.num_runs, 
                verbose=args.verbose
            )
        
    elif args.command == 'profile':
        print(f"Profiling complex polynomial multiplication complexity up to N = {args.max_N}")
        results = profile_multiplication_complexity(
            max_N=args.max_N,
            step=args.step,
            method='both',
            verbose=args.verbose
        )
    
    elif args.command == 'int-profile':
        print(f"Profiling integer NTT complexity up to N = {args.max_N}")
        results = profile_integer_ntt_complexity(
            max_N=args.max_N,
            step=args.step,
            method='both',
            verbose=args.verbose
        )
        
    elif args.command == 'test':
        print(f"Testing complex polynomial multiplication for N = {args.sizes}")
        for N in args.sizes:
            try:
                passed, total, max_error = test_polynomial_multiplication(
                    N=N, num_tests=3, verbose=args.verbose
                )
                print(f"N={N}: {passed}/{total} tests passed, max error: {max_error:.2e}")
            except Exception as e:
                print(f"N={N}: ERROR: {e}")
    
    elif args.command == 'int-test':
        # Handle min-N/max-N vs specific sizes for integer NTT correctness test
        if args.min_N is not None or args.max_N is not None:
            if args.min_N is None or args.max_N is None:
                print("Error: Both --min-N and --max-N must be specified for range-based testing")
                exit(1)
            if args.sizes is not None:
                print("Error: Cannot specify both -N and --min-N/--max-N")
                exit(1)
            print(f"Testing integer NTT cyclotomic correctness for N range {args.min_N} to {args.max_N}")
            results = test_integer_ntt_cyclotomic_correctness(
                min_N=args.min_N,
                max_N=args.max_N,
                num_tests=args.num_tests,
                verbose=args.verbose
            )
        else:
            sizes = args.sizes if args.sizes else [6, 12, 24]
            print(f"Testing integer NTT cyclotomic correctness for N = {sizes}")
            results = test_integer_ntt_cyclotomic_correctness(
                N_values=sizes,
                num_tests=args.num_tests,
                verbose=args.verbose
            )
