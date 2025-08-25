"""
Clean polynomial multiplication functions using Number Theoretic Transform (NTT).
This module provides both standard and cyclotomic polynomial multiplication.
"""

import argparse
import time
from integer_dft import IntegerDFT

def _find_next_valid_size(min_size: int) -> int:
    """Find the smallest size >= min_size that is of form 2^a * 3^b."""
    if min_size <= 1:
        return 1
    
    # Generate candidates of form 2^a * 3^b up to a reasonable limit
    candidates = []
    max_power = 20  # Reasonable limit to avoid infinite search
    
    for a in range(max_power):
        for b in range(max_power):
            candidate = (2**a) * (3**b)
            if candidate >= min_size and candidate <= 2**max_power:
                candidates.append(candidate)
    
    if not candidates:
        # Fallback to next power of 2 if no suitable 2^a * 3^b found
        N = 1
        while N < min_size:
            N *= 2
        return N
    
    return min(candidates)

def polynomial_multiply_ntt(poly1: list, poly2: list, min_bits: int = 20, custom_prime: int = None) -> list:
    """Multiply two polynomials using cyclotomic NTT with automatic size selection."""
    n1, n2 = len(poly1), len(poly2)
    
    # Determine transform size  
    min_size = n1 + n2 - 1
    N = _find_next_valid_size(min_size)
        
    # Pad polynomials to size N
    a = poly1 + [0] * (N - n1)
    b = poly2 + [0] * (N - n2)
    
    # Create cyclotomic NTT instance
    if custom_prime is not None:
        ntt = IntegerDFT(N, p=custom_prime)
    else:
        ntt = IntegerDFT(N, min_bits=min_bits)
    
    # Transform, multiply, and inverse transform using factorized NTT
    A = ntt.dft(a, method="factorized")
    B = ntt.dft(b, method="factorized")
    C = [(A[i] * B[i]) % ntt.p for i in range(N)]
    result = ntt.idft(C, method="factorized")
    
    return result[:min_size]

def cyclotomic_polynomial_multiply_ntt(poly1: list, poly2: list, min_bits: int = 20, custom_prime: int = None) -> list:
    """Multiply two polynomials in the cyclotomic ring Z_p[X]/(X^N - X^{N/2} + 1)."""
    N = len(poly1)
    assert len(poly2) == N, "Polynomials must be of the same degree"
    
    # Create cyclotomic NTT instance
    if custom_prime is not None:
        ntt = IntegerDFT(N, p=custom_prime)
    else:
        ntt = IntegerDFT(N, min_bits=min_bits)
    
    # Transform, multiply, and inverse transform using factorized NTT
    A = ntt.dft(poly1, method="factorized")
    B = ntt.dft(poly2, method="factorized")
    C = [(A[i] * B[i]) % ntt.p for i in range(N)]
    result = ntt.idft(C, method="factorized")
    
    return result

def compare_all_methods(poly1: list, poly2: list, verbose: bool = False) -> dict:
    """
    Compare factorized NTT, direct NTT, and naive cyclotomic multiplication.
    
    Returns:
        Dictionary with results from all three methods and comparison info
    """
    N = len(poly1)
    assert len(poly2) == N, "Polynomials must be of the same degree"
    
    # Import the naive function from poly_mult
    from poly_mult import naive_cyclotomic_poly_mult_int
    
    # Create NTT instance
    ntt = IntegerDFT(N)
    p = ntt.p
    
    results = {}
    
    if verbose:
        print(f"Comparing all methods for N={N}, prime p={p}")
        print(f"poly1: {poly1}")
        print(f"poly2: {poly2}")
        print("-" * 50)
    
    # Method 1: Factorized NTT
    try:
        import time
        start_time = time.time()
        A_fact = ntt.dft(poly1, method="factorized")
        B_fact = ntt.dft(poly2, method="factorized")
        C_fact = [(A_fact[i] * B_fact[i]) % p for i in range(N)]
        factorized_result = ntt.idft(C_fact, method="factorized")
        factorized_time = time.time() - start_time
        
        results['factorized'] = {
            'result': factorized_result,
            'time': factorized_time,
            'success': True
        }
        
        if verbose:
            print(f"Factorized NTT: {factorized_result}")
            print(f"Time: {factorized_time:.6f}s")
    except Exception as e:
        results['factorized'] = {'result': None, 'time': 0, 'success': False, 'error': str(e)}
        if verbose:
            print(f"Factorized NTT: ERROR - {e}")
    
    # Method 3: Naive cyclotomic multiplication
    try:
        start_time = time.time()
        naive_result_raw = naive_cyclotomic_poly_mult_int(poly1, poly2, N)
        naive_result = [(x % p + p) % p for x in naive_result_raw]
        naive_time = time.time() - start_time
        
        results['naive'] = {
            'result': naive_result,
            'result_raw': naive_result_raw,
            'time': naive_time,
            'success': True
        }
        
        if verbose:
            print(f"Naive (raw):    {naive_result_raw}")
            print(f"Naive (mod {p}): {naive_result}")
            print(f"Time: {naive_time:.6f}s")
    except Exception as e:
        results['naive'] = {'result': None, 'time': 0, 'success': False, 'error': str(e)}
        if verbose:
            print(f"Naive: ERROR - {e}")
    
    # Compare results
    methods = ['factorized', 'naive']
    successful_methods = [m for m in methods if results[m]['success']]
    
    if verbose and len(successful_methods) > 1:
        print("-" * 50)
        print("COMPARISONS:")
        
        for i, method1 in enumerate(successful_methods):
            for method2 in successful_methods[i+1:]:
                result1 = results[method1]['result']
                result2 = results[method2]['result']
                
                if result1 is not None and result2 is not None:
                    match = list(result1) == list(result2)
                    print(f"{method1.capitalize()} vs {method2.capitalize()}: {match} {'✓' if match else '✗'}")
                    
                    if not match:
                        print(f"  {method1}: {result1}")
                        print(f"  {method2}: {result2}")
                        # Show first difference
                        for j, (a, b) in enumerate(zip(result1, result2)):
                            if a != b:
                                print(f"  First difference at index {j}: {method1}={a}, {method2}={b}")
                                break
    
    # Add comparison summary
    results['comparisons'] = {}
    for i, method1 in enumerate(successful_methods):
        for method2 in successful_methods[i+1:]:
            result1 = results[method1]['result']
            result2 = results[method2]['result']
            if result1 is not None and result2 is not None:
                results['comparisons'][f"{method1}_vs_{method2}"] = list(result1) == list(result2)
    
    return results

def test_all_methods_comparison(N: int = 12, num_tests: int = 5, verbose: bool = False):
    """
    Test and compare factorized NTT, direct NTT, and naive cyclotomic multiplication.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE METHOD COMPARISON (N={N})")
    print(f"Testing Factorized NTT vs Direct NTT vs Naive Cyclotomic")
    print(f"{'='*80}")
    
    import random
    random.seed(42)  # For reproducible results
    
    test_cases = [
        {
            'name': 'Simple test: [1,0,1,0,...] * [1,1,0,0,...]',
            'poly1': [1, 0, 1] + [0] * (N-3),
            'poly2': [1, 1] + [0] * (N-2)
        },
        {
            'name': 'Delta functions: X^2 * X^3',
            'poly1': [0, 0, 1] + [0] * (N-3),
            'poly2': [0, 0, 0, 1] + [0] * (N-4)
        }
    ]
    
    # Add random test cases
    for i in range(num_tests - 2):
        max_coeff = 5
        poly1 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
        poly2 = [random.randint(-max_coeff, max_coeff) for _ in range(N)]
        test_cases.append({
            'name': f'Random test {i+1}',
            'poly1': poly1,
            'poly2': poly2
        })
    
    overall_results = {
        'total_tests': len(test_cases),
        'factorized_vs_direct': {'passed': 0, 'failed': 0},
        'factorized_vs_naive': {'passed': 0, 'failed': 0},
        'direct_vs_naive': {'passed': 0, 'failed': 0},
        'all_methods_agree': {'passed': 0, 'failed': 0}
    }
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        if verbose:
            print(f"poly1: {test_case['poly1'][:6]}{'...' if len(test_case['poly1']) > 6 else ''}")
            print(f"poly2: {test_case['poly2'][:6]}{'...' if len(test_case['poly2']) > 6 else ''}")
        
        results = compare_all_methods(test_case['poly1'], test_case['poly2'], verbose=verbose)
        
        # Check comparisons
        comparisons = results.get('comparisons', {})
        
        # Factorized vs Direct
        if 'factorized_vs_direct' in comparisons:
            if comparisons['factorized_vs_direct']:
                overall_results['factorized_vs_direct']['passed'] += 1
                if not verbose:
                    print("  Factorized vs Direct: ✓ PASS")
            else:
                overall_results['factorized_vs_direct']['failed'] += 1
                print("  Factorized vs Direct: ✗ FAIL")
        
        # Factorized vs Naive
        if 'factorized_vs_naive' in comparisons:
            if comparisons['factorized_vs_naive']:
                overall_results['factorized_vs_naive']['passed'] += 1
                if not verbose:
                    print("  Factorized vs Naive:  ✓ PASS")
            else:
                overall_results['factorized_vs_naive']['failed'] += 1
                print("  Factorized vs Naive:  ✗ FAIL")
        
        # Direct vs Naive
        if 'direct_vs_naive' in comparisons:
            if comparisons['direct_vs_naive']:
                overall_results['direct_vs_naive']['passed'] += 1
                if not verbose:
                    print("  Direct vs Naive:      ✓ PASS")
            else:
                overall_results['direct_vs_naive']['failed'] += 1
                print("  Direct vs Naive:      ✗ FAIL")
        
        # All methods agree
        all_agree = all(comparisons.values()) if comparisons else False
        if all_agree:
            overall_results['all_methods_agree']['passed'] += 1
            if not verbose:
                print("  All methods agree:    ✓ PASS")
        else:
            overall_results['all_methods_agree']['failed'] += 1
            if not verbose:
                print("  All methods agree:    ✗ FAIL")
        
        if verbose:
            print("-" * 50)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"Total tests: {overall_results['total_tests']}")
    print()
    
    for comparison, results in overall_results.items():
        if comparison != 'total_tests':
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            if total > 0:
                percentage = (passed / total) * 100
                status = "✓ PASS" if failed == 0 else "✗ FAIL"
                print(f"{comparison.replace('_', ' ').title():25}: {passed}/{total} ({percentage:5.1f}%) {status}")
    
    print()
    
    # Overall status
    all_passed = all(results['failed'] == 0 for key, results in overall_results.items() 
                    if key != 'total_tests')
    
    if all_passed:
        print("🎉 SUCCESS: All methods produce identical results!")
        print("✅ Factorized NTT, Direct NTT, and Naive methods all agree")
    else:
        print("❌ FAILURE: There are discrepancies between methods")
        print("   This indicates bugs in one or more implementations")
    
    return overall_results

def test_integer_dft(N: int, verbose: bool = False):
    """Test the integer DFT implementation."""
    print(f"\nTesting Factorized Integer NTT for N={N}")
    print("=" * 50)
    
    # Create NTT instance
    ntt = IntegerDFT(N, verbose=verbose)
    print(f"Using prime p = {ntt.p}, omega = {ntt.omega}")
    
    # Test cases
    test_cases = [
        [1] + [0] * (N-1),  # Delta function
        [1] * N,            # Constant
    ]
    
    if N >= 4:
        # Alternating sequence
        alt = [1 if i % 2 == 0 else -1 for i in range(N)]
        test_cases.append(alt)
    
    for i, coeffs in enumerate(test_cases):
        print(f"\nTest case {i+1}: {coeffs}")
        
        # Forward transform using factorized NTT
        transformed = ntt.dft(coeffs, verbose=verbose, method="factorized")
        if not verbose:
            print(f"NTT:  {transformed}")
        
        # Inverse transform using factorized NTT
        recovered = ntt.idft(transformed, verbose=verbose, method="factorized")
        if not verbose:
            print(f"INTT: {recovered}")
        
        # Check roundtrip accuracy
        original_mod = [x % ntt.p for x in coeffs]
        success = all(a == b for a, b in zip(original_mod, recovered))
        
        if success:
            print("✓ Roundtrip successful")
        else:
            print("✗ Roundtrip failed")
            print(f"Expected: {original_mod}")
            print(f"Got:      {recovered}")

def benchmark_integer_ntt(N: int):
    """Run polynomial multiplication benchmarks using Integer NTT."""
    print(f"Benchmarking Integer NTT polynomial multiplication (N={N})")
    print("=" * 50)
    
    # Test different polynomial degrees that fit within N
    degrees = []
    for deg in [4, 8, 16, 32, 64, 128, 256]:
        if deg <= N // 2:  # Ensure result fits in transform size
            degrees.append(deg)
    
    if not degrees:
        degrees = [N // 4] if N >= 4 else [1]
    
    for degree in degrees:
        # Generate test polynomials
        poly1 = list(range(1, degree + 1))
        poly2 = list(range(degree, 0, -1))
        
        # Benchmark the multiplication
        start_time = time.time()
        try:
            result = polynomial_multiply_ntt(poly1, poly2)
            ntt_time = time.time() - start_time
            result_degree = len([x for x in reversed(result) if x != 0])
            
            print(f"Degree {degree:3d}: {ntt_time:.6f}s (result degree: {result_degree})")
        except Exception as e:
            print(f"Degree {degree:3d}: Failed - {e}")
            
    print()

def naive_cyclotomic_multiply(poly1: list, poly2: list, p: int) -> list:
    """Naive cyclotomic multiplication with iterative reduction."""
    N = len(poly1)
    
    # Standard polynomial multiplication
    product = [0] * (2*N - 1)
    for i in range(N):
        for j in range(N):
            if poly1[i] != 0 and poly2[j] != 0:
                product[i + j] = (product[i + j] + poly1[i] * poly2[j]) % p
    
    # Iteratively reduce modulo X^N - X^{N/2} + 1
    # Rule: X^k with k >= N becomes X^{k-N+N/2} - X^{k-N}
    while any(i >= N for i in range(len(product)) if product[i] != 0):
        new_product = [0] * max(N, len(product))
        
        for i in range(len(product)):
            coeff = product[i]
            if coeff == 0:
                continue
                
            if i < N:
                new_product[i] = (new_product[i] + coeff) % p
            else:
                # X^i = X^{i-N} * X^N ≡ X^{i-N} * (X^{N/2} - 1)
                deg1 = i - N + N // 2
                deg2 = i - N
                
                if deg1 < len(new_product):
                    new_product[deg1] = (new_product[deg1] + coeff) % p
                else:
                    new_product.extend([0] * (deg1 + 1 - len(new_product)))
                    new_product[deg1] = coeff % p
                
                if deg2 < len(new_product):
                    new_product[deg2] = (new_product[deg2] - coeff) % p
                else:
                    new_product.extend([0] * (deg2 + 1 - len(new_product)))
                    new_product[deg2] = (-coeff) % p
        
        product = new_product
    
    return [product[i] % p if i < len(product) else 0 for i in range(N)]

def test_cyclotomic_correctness():
    """Test cyclotomic polynomial multiplication: naive vs DFT."""
    print("\n" + "="*60)
    print("CYCLOTOMIC POLYNOMIAL MULTIPLICATION TEST")
    print("Comparing Naive vs DFT Implementation")
    print("="*60)
    
    # Get prime from a sample IntegerDFT instance
    sample_ntt = IntegerDFT(6)
    p = sample_ntt.p
    
    test_cases = [
        {
            'name': 'X^3 * X^3 = X^6 ≡ X^3 - 1', 
            'poly1': [0, 0, 0, 1, 0, 0], 
            'poly2': [0, 0, 0, 1, 0, 0],
            'expected_desc': 'X^3 - 1'
        },
        {
            'name': '(1+X) * (1+X) = 1 + 2X + X^2', 
            'poly1': [1, 1, 0, 0, 0, 0], 
            'poly2': [1, 1, 0, 0, 0, 0],
            'expected_desc': '1 + 2X + X^2'
        },
        {
            'name': 'X^5 * X^5 = X^10 ≡ -X', 
            'poly1': [0, 0, 0, 0, 0, 1], 
            'poly2': [0, 0, 0, 0, 0, 1],
            'expected_desc': '-X'
        },
        {
            'name': 'X^4 * X^4 = X^8 ≡ X^5 - X^2',
            'poly1': [0, 0, 0, 0, 1, 0],
            'poly2': [0, 0, 0, 0, 1, 0], 
            'expected_desc': 'X^5 - X^2'
        },
        {
            'name': 'X^3 * X^4 = X^7 ≡ X^4 - X',
            'poly1': [0, 0, 0, 1, 0, 0],
            'poly2': [0, 0, 0, 0, 1, 0],
            'expected_desc': 'X^4 - X'
        },
        {
            'name': '(1+X^2) * (1+X^3)', 
            'poly1': [1, 0, 1, 0, 0, 0], 
            'poly2': [1, 0, 0, 1, 0, 0],
            'expected_desc': '1 + X^2 + X^3 + X^5'
        }
    ]
    
    print(f"Using prime p = {p}")
    print(f"Working in ring Z_{p}[X]/(X^6 - X^3 + 1)")
    
    all_tests_passed = True
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"Expected: {test['expected_desc']}")
        print(f"poly1: {test['poly1']}")
        print(f"poly2: {test['poly2']}")
        
        try:
            # Method 1: Naive implementation
            start_time = time.time()
            naive_result = naive_cyclotomic_multiply(test['poly1'], test['poly2'], p)
            naive_time = time.time() - start_time
            
            # Method 2: DFT implementation
            start_time = time.time()
            dft_result = cyclotomic_polynomial_multiply_ntt(test['poly1'], test['poly2'])
            dft_time = time.time() - start_time
            
            print(f"Naive result:     {naive_result}")
            print(f"DFT result:       {dft_result}")
            
            # Check if results match
            results_match = naive_result == dft_result
            print(f"Results match:    {results_match} {'✓' if results_match else '✗'}")
            
            if not results_match:
                all_tests_passed = False
                print(f"ERROR: Naive and DFT results differ!")
            
            # Performance comparison
            if naive_time > 0 and dft_time > 0:
                speedup = naive_time / dft_time
                print(f"Timing: Naive={naive_time:.6f}s, DFT={dft_time:.6f}s, Speedup={speedup:.2f}x")
            
        except Exception as e:
            print(f"ERROR: {e}")
            all_tests_passed = False
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print(f"\nSUMMARY:")
    print(f"All tests passed: {all_tests_passed}")
    if all_tests_passed:
        print("🎉 SUCCESS: Naive and DFT implementations produce identical results!")
        print("✅ Cyclotomic polynomial multiplication is working correctly")
    else:
        print("❌ FAILURE: There are discrepancies between naive and DFT implementations")
    
    return all_tests_passed

def test_standard_correctness():
    """Test standard polynomial multiplication: naive vs DFT."""
    print("\n" + "="*60)
    print("STANDARD POLYNOMIAL MULTIPLICATION TEST")
    print("Comparing Naive vs DFT Implementation")
    print("="*60)
    
    def naive_multiply(poly1, poly2):
        """Naive polynomial multiplication."""
        result = [0] * (len(poly1) + len(poly2) - 1)
        for i in range(len(poly1)):
            for j in range(len(poly2)):
                result[i + j] += poly1[i] * poly2[j]
        return result
    
    test_cases = [
        {
            'name': '(1+X) * (1+X)',
            'poly1': [1, 1],
            'poly2': [1, 1]
        },
        {
            'name': '(1+X+X^2) * (1-X)',
            'poly1': [1, 1, 1],
            'poly2': [1, -1]
        },
        {
            'name': 'X^3 * X^2',
            'poly1': [0, 0, 0, 1],
            'poly2': [0, 0, 1]
        },
        {
            'name': '(1+2X+3X^2) * (4+5X)',
            'poly1': [1, 2, 3],
            'poly2': [4, 5]
        }
    ]
    
    all_tests_passed = True
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['name']}")
        print(f"poly1: {test['poly1']}")
        print(f"poly2: {test['poly2']}")
        
        try:
            # Method 1: Naive implementation
            start_time = time.time()
            naive_result = naive_multiply(test['poly1'], test['poly2'])
            naive_time = time.time() - start_time
            
            # Method 2: DFT implementation
            start_time = time.time()
            dft_result = polynomial_multiply_ntt(test['poly1'], test['poly2'])
            dft_time = time.time() - start_time
            
            print(f"Naive result:     {naive_result}")
            print(f"DFT result:       {dft_result}")
            
            # Check if results match (modulo the prime used in NTT)
            # We need to use the same transform size and get the actual prime used
            min_size = len(test['poly1']) + len(test['poly2']) - 1
            N = _find_next_valid_size(min_size)
            actual_ntt = IntegerDFT(N)
            p = actual_ntt.p
            
            # Convert naive result to modulo p, handling negative numbers correctly
            naive_mod = [(x % p + p) % p for x in naive_result]
            
            results_match = naive_mod == dft_result
            print(f"Results match:    {results_match} {'✓' if results_match else '✗'}")
            print(f"(mod {p}):        {naive_mod}")
            print(f"Transform size:   {N}, Prime: {p}")
            
            if not results_match:
                all_tests_passed = False
                print(f"ERROR: Naive and DFT results differ!")
            
            # Performance comparison
            if naive_time > 0 and dft_time > 0:
                speedup = naive_time / dft_time
                print(f"Timing: Naive={naive_time:.6f}s, DFT={dft_time:.6f}s, Speedup={speedup:.2f}x")
            
        except Exception as e:
            print(f"ERROR: {e}")
            all_tests_passed = False
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print(f"\nSUMMARY:")
    print(f"All tests passed: {all_tests_passed}")
    if all_tests_passed:
        print("✅ Standard polynomial multiplication is working correctly")
    else:
        print("❌ There are discrepancies in standard polynomial multiplication")
    
    return all_tests_passed

if __name__ == "__main__":
    # Simple interface for backward compatibility
    import sys
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
            verbose = '-v' in sys.argv or '--verbose' in sys.argv
            test_integer_dft(N, verbose)
        except ValueError:
            print("Usage: python integer.py <N> [-v|--verbose]")
            print("For comprehensive testing, use: python main.py poly-mult-integer <N>")
    else:
        print("Usage: python integer.py <N> [-v|--verbose]")
        print("For comprehensive testing, use: python main.py poly-mult-integer <N>")
        print("\nExamples:")
        print("  python integer.py 12 -v                     # Test NTT-12 with verbose output")
        print("  python main.py poly-mult-integer 12 -v      # Full test suite with verbose output")
