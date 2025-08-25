#!/usr/bin/env python3
"""
Unified interface for testing both Complex Cyclotomic DFT and Integer NTT implementations.

This script provides a single entry point to test:
1. Complex Cyclot  %(prog)s complex 6 -v           # Test complex DFT with verbose output
  %(prog)s integer 12             # Test integer NTT with 3 random vectors  
  %(prog)s both 8 --benchmark    # Test both implementations with benchmark (3 tests)
  %(prog)s complex 12 --num-tests 10  # Test with 10 random vectors
  %(prog)s both 16 --num-tests 50 -v  # Test both with 50 random vectors, verbose
  %(prog)s poly-mult-complex 6 -v    # Test complex polynomial multiplication with verbose output
  %(prog)s poly-mult-integer 12 --num-tests 10  # Test integer polynomial multiplication with 10 cases
  %(prog)s poly-mult-both 6 -v       # Test both polynomial multiplication methods
  %(prog)s complex --all --benchmark  # Test all sizes with benchmarks
  %(prog)s benchmark --min-N 6 --max-N 48 --num-runs 100 -v  # Comprehensive benchmark with custom range
  %(prog)s benchmark --min-N 6 --max-N 24 --min-bits 32 -v  # Use primes >= 2^32 for integer NTT
  %(prog)s benchmark --min-N 6 --max-N 12 --custom-prime 1009 -v  # Use specific prime p=1009(factorized with DFT-2/DFT-3 primitives)
2. Integer NTT (factorized with NTT-2/NTT-3 primitives for cyclotomic polynomial FFT)

Both implementations use the same factorization strategy for N = 2^a * 3^b.
"""

import sys
import argparse
from pathlib import Path

# Import the implementations
try:
    from complex import test_cyclotomic_dft, benchmark_cyclotomic_dft
    COMPLEX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Complex DFT not available: {e}")
    COMPLEX_AVAILABLE = False

try:
    from integer import test_integer_dft, benchmark_integer_ntt, test_all_methods_comparison
    INTEGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integer NTT not available: {e}")
    INTEGER_AVAILABLE = False

try:
    from poly_mult import test_polynomial_multiplication, test_integer_polynomial_multiplication
    POLY_MULT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Polynomial multiplication not available: {e}")
    POLY_MULT_AVAILABLE = False

try:
    from poly_mult import test_polynomial_multiplication
    POLY_MULT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Polynomial multiplication testing not available: {e}")
    POLY_MULT_AVAILABLE = False


def generate_random_vector(N, mode='complex'):
    """Generate a random test vector of size N."""
    import random
    
    if mode == 'complex':
        # Generate complex coefficients
        real_parts = [random.randint(-10, 10) for _ in range(N)]
        imag_parts = [random.randint(-5, 5) for _ in range(N)]
        return [complex(r, i) for r, i in zip(real_parts, imag_parts)]
    else:
        # Generate integer coefficients
        return [random.randint(-20, 20) for _ in range(N)]


def test_complex_random(N, num_tests, verbose=False):
    """Test complex DFT with random inputs."""
    import random
    import numpy as np
    from complex import CyclotomicDFT
    
    print(f"Testing Complex Cyclotomic DFT with {num_tests} random vectors (N={N})")
    print("=" * 60)
    
    try:
        dft = CyclotomicDFT(N)
        if verbose:
            print(f"Initialized Fast CyclotomicDFT for N={N} = 2^{dft.a} * 3^{dft.b}")
            print(f"Cyclotomic polynomial: X^{N} - X^{N//2} + 1")
            print()
        
        all_passed = True
        max_error = 0.0
        
        for i in range(num_tests):
            # Generate random test vector as numpy array
            test_vec = np.array(generate_random_vector(N, 'complex'), dtype=complex)
            
            
            if verbose:
                vector_info = f"Test {i+1}: Random vector {test_vec}"
            else:
                vector_info = f"Test {i+1}: Random complex vector"

            # Forward transform
            result = dft.dft(test_vec, verbose=verbose and num_tests <= 3)
            
            # Inverse transform
            recovered = dft.idft(result, verbose=verbose and num_tests <= 3)
            
            # Check accuracy
            error = max(abs(orig - rec) for orig, rec in zip(test_vec, recovered))
            max_error = max(max_error, error)
            
            if error < 1e-10:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
                all_passed = False
                
            print(f"{vector_info}, {status}")  # Append the random complex vector info
            
            if verbose or not all_passed:
                print(f"  Roundtrip error: {error:.2e} {status}")
            # elif num_tests > 3:
            #     print(f"  Error: {error:.2e} {status}")
            
            if verbose and num_tests <= 3:
                print()
        
        print(f"\nSummary: {num_tests} tests, max error: {max_error:.2e}")
        return all_passed
        
    except Exception as e:
        print(f"Error in complex DFT test: {e}")
        return False


def test_integer_random(N, num_tests, verbose=False, min_bits=None, custom_prime=None):
    """Test integer NTT with random inputs.""" 
    import random
    from integer import IntegerDFT
    
    print(f"Testing Integer NTT with {num_tests} random vectors (N={N})")
    print("=" * 60)
    
    try:
        ntt = IntegerDFT(N, verbose=verbose and num_tests <= 3, min_bits=min_bits, p=custom_prime)
        
        all_passed = True
        
        for i in range(num_tests):
            # Generate random test vector
            test_vec = generate_random_vector(N, 'integer')
            
            if verbose:
                vector_info = f"Test {i+1}: Random vector {test_vec}"
            else:
                vector_info = f"Test {i+1}: Random integer vector"
            
            # Forward transform
            result = ntt.dft(test_vec, verbose=verbose and num_tests <= 3)
            
            # Inverse transform  
            recovered = ntt.idft(result, verbose=verbose and num_tests <= 3)
            
            # Check accuracy (mod p)
            original_mod = [x % ntt.p for x in test_vec]
            success = all(a == b for a, b in zip(original_mod, recovered))
            
            if success:
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
                all_passed = False
                
            print(f"{vector_info}, {status}")
            
            if verbose or not success:
                if not success:
                    print(f"  Expected: {original_mod}")
                    print(f"  Got:      {recovered}")
                print(f"  Roundtrip: {status}")
            # elif num_tests > 3:
            #     print(f"  {status}")
            
            if verbose and num_tests <= 3:
                print()
        
        print(f"\nSummary: {num_tests} tests completed")
        return all_passed
        
    except Exception as e:
        print(f"Error in integer NTT test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Unified test interface for Factorized DFT/NTT implementations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s complex 4 -v          # Test complex cyclotomic DFT with verbose output (3 tests)
  %(prog)s integer 6 -v          # Test integer NTT with verbose output (3 tests)
  %(prog)s both 8 --benchmark    # Test both implementations with benchmark (3 tests)
  %(prog)s complex 12 --num-tests 10  # Test with 10 random vectors
  %(prog)s both 16 --num-tests 50 -v  # Test both with 50 random vectors, verbose
  %(prog)s poly-mult-complex 6 -v    # Test complex polynomial multiplication with verbose output
  %(prog)s poly-mult-integer 12 --num-tests 10  # Test integer polynomial multiplication with 10 cases
  %(prog)s poly-mult-both 6 -v       # Test both polynomial multiplication methods
  %(prog)s complex --all --benchmark  # Test all sizes with benchmarks
        """)
    
    parser.add_argument('mode', choices=['complex', 'integer', 'both', 'poly-mult-complex', 'poly-mult-integer', 'poly-mult-both', 'benchmark'],
                       help='Which test to run: complex DFT, integer NTT, both, complex poly mult, integer poly mult, both poly mult, or comprehensive benchmark')
    parser.add_argument('N', type=int, nargs='?',
                       help='Transform size N (must be of form 2^a * 3^b). Optional if --all is used or for benchmark mode.')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output to show factorization details')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--all', action='store_true',
                       help='Test all supported sizes (4, 6, 8, 9, 12, 16, 18, 24, 27, 32)')
    parser.add_argument('--num-tests', type=int, default=3,
                       help='Number of random test cases to run (default: 3)')
    parser.add_argument('--min-N', type=int, default=6,
                       help='Minimum N for benchmark mode (default: 6)')
    parser.add_argument('--max-N', type=int, default=72,
                       help='Maximum N for benchmark mode (default: 72)')
    parser.add_argument('--num-runs', type=int, default=50,
                       help='Number of timing runs per method in benchmark mode (default: 50)')
    parser.add_argument('--min-bits', type=int, default=20,
                       help='Minimum bit size for integer NTT prime in benchmark mode (default: 20, i.e., p >= 2^20)')
    parser.add_argument('--custom-prime', type=int,
                       help='Use a specific custom prime for integer NTT (must satisfy p ≡ 1 (mod 3N))')
    
    args = parser.parse_args()
    
    # Validate that either N is provided or --all is used (except for benchmark mode)
    if args.mode != 'benchmark' and not args.all and args.N is None:
        parser.error("Either specify N, use --all flag, or use benchmark mode")
    
    if args.all and args.N is not None:
        parser.error("Cannot specify both N and --all flag")
    
    # Validate arguments
    if not COMPLEX_AVAILABLE and args.mode in ['complex', 'both', 'poly-mult-complex', 'poly-mult-both', 'benchmark']:
        print("Error: Complex DFT implementation not available")
        return 1
        
    if not INTEGER_AVAILABLE and args.mode in ['integer', 'both', 'poly-mult-integer', 'poly-mult-both', 'benchmark']:
        print("Error: Integer NTT implementation not available")
        return 1
    
    if not POLY_MULT_AVAILABLE and args.mode in ['poly-mult-complex', 'poly-mult-integer', 'poly-mult-both', 'benchmark']:
        print("Error: Polynomial multiplication testing not available")
        return 1
    
    # Validate N format if specified
    if args.N is not None and not validate_transform_size(args.N):
        print(f"Error: N={args.N} must be of form 2^a * 3^b where a,b >= 1")
        return 1
    
    success = True
    results = {}
    
    # Handle benchmark mode
    if args.mode == 'benchmark':
        print("🚀 COMPREHENSIVE POLYNOMIAL MULTIPLICATION BENCHMARK")
        print("=" * 60)
        benchmark_results = benchmark_polynomial_multiplication(
            min_N=args.min_N, 
            max_N=args.max_N, 
            num_runs=args.num_runs, 
            min_bits=args.min_bits,
            custom_prime=args.custom_prime,
            verbose=args.verbose
        )
        return 0 if benchmark_results else 1
    
    print(f"Testing Factorized DFT/NTT implementations for N={args.N}")
    print("=" * 60)
    
    # Test complex implementation
    if args.mode in ['complex', 'both']:
        print(f"\n🔢 COMPLEX CYCLOTOMIC DFT (N={args.N})")
        print("-" * 40)
        try:
            # Use random vector tests
            complex_success = test_complex_random(args.N, args.num_tests, verbose=args.verbose)
                
            results['complex'] = complex_success
            if not complex_success:
                success = False
                print("❌ Complex DFT tests failed!")
            else:
                print("✅ Complex DFT tests passed!")
                
            if args.benchmark:
                print("\n📊 Complex DFT Benchmark:")
                benchmark_cyclotomic_dft(args.N)
                
        except Exception as e:
            print(f"❌ Complex DFT error: {e}")
            success = False
            results['complex'] = False
    
    # Test integer implementation  
    if args.mode in ['integer', 'both']:
        print(f"\n🔢 INTEGER NTT (N={args.N})")
        print("-" * 40)
        try:
            # Use random vector tests
            integer_success = test_integer_random(args.N, args.num_tests, verbose=args.verbose, 
                                                  min_bits=args.min_bits, custom_prime=args.custom_prime)
            
            results['integer'] = integer_success
            if not integer_success:
                success = False
                print("❌ Integer NTT tests failed!")
            else:
                print("✅ Integer NTT tests passed!")
                
            if args.benchmark:
                print("\n📊 Integer NTT Benchmark:")
                benchmark_integer_ntt(args.N)
                
        except Exception as e:
            print(f"❌ Integer NTT error: {e}")
            success = False
            results['integer'] = False
    
    # Test complex polynomial multiplication
    if args.mode in ['poly-mult-complex', 'poly-mult-both']:
        print(f"\n🔢 COMPLEX POLYNOMIAL MULTIPLICATION (N={args.N})")
        print("-" * 40)
        try:
            passed, total, max_error = test_polynomial_multiplication(args.N, args.num_tests, verbose=args.verbose)
            
            results['poly-mult-complex'] = (passed == total)
            if passed != total:
                success = False
                print(f"❌ Complex polynomial multiplication tests failed! ({passed}/{total} passed)")
            else:
                print(f"✅ Complex polynomial multiplication tests passed! (Max error: {max_error:.2e})")
                
        except Exception as e:
            print(f"❌ Complex polynomial multiplication error: {e}")
            success = False
            results['poly-mult-complex'] = False
    
    # Test integer polynomial multiplication  
    if args.mode in ['poly-mult-integer', 'poly-mult-both']:
        print(f"\n🔢 INTEGER POLYNOMIAL MULTIPLICATION (N={args.N})")
        print("-" * 40)
        
        # Default: factorized NTT vs naive cyclotomic comparison
        try:
            print(f"Running factorized NTT vs naive cyclotomic comparison")
            from integer import compare_all_methods
            import random
            random.seed(42)
            
            passed_tests = 0
            total_tests = args.num_tests
            
            for i in range(total_tests):
                if i == 0:
                    # Simple test case
                    poly1 = [1, 0, 1] + [0] * (args.N - 3)
                    poly2 = [1, 1] + [0] * (args.N - 2)
                    test_name = "Simple test"
                else:
                    # Random test case
                    max_coeff = 5
                    poly1 = [random.randint(-max_coeff, max_coeff) for _ in range(args.N)]
                    poly2 = [random.randint(-max_coeff, max_coeff) for _ in range(args.N)]
                    test_name = f"Random test {i}"
                
                if args.verbose:
                    print(f"\nTest {i+1}: {test_name}")
                
                comparison_result = compare_all_methods(poly1, poly2, verbose=args.verbose)
                
                # Check factorized vs naive comparison
                comparisons = comparison_result.get('comparisons', {})
                if 'factorized_vs_naive' in comparisons:
                    if comparisons['factorized_vs_naive']:
                        passed_tests += 1
                        if not args.verbose:
                            print(f"Test {i+1}: ✓ PASS - {test_name}")
                    else:
                        if not args.verbose:
                            print(f"Test {i+1}: ✗ FAIL - {test_name}")
                
            results['poly-mult-integer'] = (passed_tests == total_tests)
            if passed_tests != total_tests:
                success = False
                print(f"❌ Factorized vs Naive comparison failed! ({passed_tests}/{total_tests} passed)")
            else:
                print(f"✅ Factorized vs Naive comparison passed! ({passed_tests}/{total_tests} tests)")
                
        except Exception as e:
            print(f"❌ Factorized vs Naive comparison error: {e}")
            success = False
            results['poly-mult-integer'] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    
    if args.mode == 'both':
        if 'complex' in results:
            status = "✅ PASS" if results['complex'] else "❌ FAIL"
            print(f"Complex DFT:  {status}")
        if 'integer' in results:
            status = "✅ PASS" if results['integer'] else "❌ FAIL"
            print(f"Integer NTT:  {status}")
    elif args.mode == 'poly-mult-both':
        if 'poly-mult-complex' in results:
            status = "✅ PASS" if results['poly-mult-complex'] else "❌ FAIL"
            print(f"Complex Polynomial Multiplication: {status}")
        if 'poly-mult-integer' in results:
            status = "✅ PASS" if results['poly-mult-integer'] else "❌ FAIL"
            print(f"Integer Polynomial Multiplication: {status}")
    elif args.mode == 'poly-mult-complex':
        if 'poly-mult-complex' in results:
            status = "✅ PASS" if results['poly-mult-complex'] else "❌ FAIL"
            print(f"Complex Polynomial Multiplication: {status}")
    elif args.mode == 'poly-mult-integer':
        if 'poly-mult-integer' in results:
            status = "✅ PASS" if results['poly-mult-integer'] else "❌ FAIL"
            print(f"Integer Polynomial Multiplication: {status}")
    else:
        impl_name = "Complex DFT" if args.mode == 'complex' else "Integer NTT"
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{impl_name}: {status}")
    
    return 0 if success else 1


def benchmark_polynomial_multiplication(min_N=6, max_N=72, num_runs=50, min_bits=20, custom_prime=None, verbose=True):
    """
    Comprehensive benchmark comparing naive, complex NTT, and integer NTT polynomial multiplication.
    
    Args:
        min_N: Minimum transform size to test
        max_N: Maximum transform size to test  
        num_runs: Number of timing runs per method per size
        min_bits: Minimum bit size for integer NTT prime (default: 20, i.e., p >= 2^20)
        custom_prime: Use a specific custom prime for integer NTT (must satisfy p ≡ 1 (mod 3N))
        verbose: Enable detailed output
        
    Returns:
        Dictionary with complete benchmark results
    """
    import time
    import numpy as np
    from integer import IntegerDFT
    from complex import complex_poly_mult
    from poly_mult import naive_poly_mult_cyclotomic, naive_cyclotomic_poly_mult_int
    
    def find_valid_N_values(min_N, max_N):
        """Find all valid N = 2^a * 3^b values in range where a,b >= 1."""
        valid_N = []
        for N in range(min_N, max_N + 1):
            if validate_transform_size(N):
                valid_N.append(N)
        return valid_N
    
    # Find all valid transform sizes in range
    N_values = find_valid_N_values(min_N, max_N)
    
    if not N_values:
        print(f"No valid N values found in range [{min_N}, {max_N}]")
        return {}
    
    if verbose:
        print("=" * 100)
        print("COMPREHENSIVE POLYNOMIAL MULTIPLICATION BENCHMARK")
        print("Comparing: Naive Cyclotomic, Complex NTT, Integer NTT")
        print("=" * 100)
        print(f"Testing N values: {N_values}")
        print(f"Runs per method: {num_runs}")
        if custom_prime is not None:
            print(f"Using custom prime: p = {custom_prime}")
        else:
            print(f"Prime constraint: p >= 2^{min_bits} = {2**min_bits}")
        print("=" * 100)
        print(f"{'N':<4} {'Naive (ms)':<12} {'Complex (ms)':<13} {'Integer (ms)':<13} {'C/N Ratio':<10} {'I/N Ratio':<10} {'I/C Ratio':<10} {'Prime':<12} {'Status':<10}")
        print("-" * 100)
    
    results = {
        'N_values': N_values,
        'naive_times_ms': [],
        'complex_times_ms': [],
        'integer_times_ms': [],
        'complex_vs_naive_speedup': [],
        'integer_vs_naive_speedup': [],
        'integer_vs_complex_speedup': [],
        'primes_used': [],
        'errors': []
    }
    
    for N in N_values:
        errors = []
        timings = {'naive': 0, 'complex': 0, 'integer': 0}
        
        try:
            # Generate test polynomials - use small coefficients to avoid numerical issues
            np.random.seed(42)  # For reproducibility
            poly1 = [complex(np.random.randint(-3, 4), np.random.randint(-3, 4)) for _ in range(N)]
            poly2 = [complex(np.random.randint(-3, 4), np.random.randint(-3, 4)) for _ in range(N)]
            
            # Convert to appropriate types for each method
            poly1_complex = np.array(poly1, dtype=complex)
            poly2_complex = np.array(poly2, dtype=complex)
            poly1_int = [int(x.real) for x in poly1]  # Integer NTT uses real integers only
            poly2_int = [int(x.real) for x in poly2]
            
            # Initialize integer NTT with specified parameters
            if custom_prime is not None:
                ntt = IntegerDFT(N, p=custom_prime, verbose=verbose)
            else:
                ntt = IntegerDFT(N, min_bits=min_bits, verbose=verbose)
            prime_used = ntt.p
            
            # 1. Naive cyclotomic multiplication (complex version)
            try:
                start_time = time.time()
                for _ in range(num_runs):
                    naive_result = naive_poly_mult_cyclotomic(poly1_complex, poly2_complex, N)
                timings['naive'] = (time.time() - start_time) * 1000 / num_runs
            except Exception as e:
                errors.append(f"Naive: {str(e)}")
                timings['naive'] = float('inf')
            
            # 2. Complex NTT (cyclotomic DFT)
            try:
                start_time = time.time()
                for _ in range(num_runs):
                    complex_result = complex_poly_mult(poly1_complex, poly2_complex, N)
                timings['complex'] = (time.time() - start_time) * 1000 / num_runs
            except Exception as e:
                errors.append(f"Complex: {str(e)}")
                timings['complex'] = float('inf')
            
            # 3. Integer NTT (cyclotomic over prime field)
            try:
                start_time = time.time()
                for _ in range(num_runs):
                    A = ntt.dft(poly1_int, method="factorized")
                    B = ntt.dft(poly2_int, method="factorized") 
                    C = [(A[i] * B[i]) % ntt.p for i in range(N)]
                    integer_result = ntt.idft(C, method="factorized")
                timings['integer'] = (time.time() - start_time) * 1000 / num_runs
            except Exception as e:
                errors.append(f"Integer: {str(e)}")
                timings['integer'] = float('inf')
            
            # Calculate speedup ratios
            complex_speedup = timings['naive'] / timings['complex'] if timings['complex'] > 0 else 0
            integer_speedup = timings['naive'] / timings['integer'] if timings['integer'] > 0 else 0
            int_vs_complex = timings['complex'] / timings['integer'] if timings['integer'] > 0 else 0
            
            # Store results
            results['naive_times_ms'].append(timings['naive'])
            results['complex_times_ms'].append(timings['complex'])
            results['integer_times_ms'].append(timings['integer'])
            results['complex_vs_naive_speedup'].append(complex_speedup)
            results['integer_vs_naive_speedup'].append(integer_speedup)
            results['integer_vs_complex_speedup'].append(int_vs_complex)
            results['primes_used'].append(prime_used)
            results['errors'].append(errors)
            
            # Display results
            if verbose:
                status = "✓ OK" if not errors else "✗ ERR"
                if timings['naive'] == float('inf'):
                    naive_str = "FAIL"
                else:
                    naive_str = f"{timings['naive']:.3f}"
                
                if timings['complex'] == float('inf'):
                    complex_str = "FAIL"
                else:
                    complex_str = f"{timings['complex']:.3f}"
                
                if timings['integer'] == float('inf'):
                    integer_str = "FAIL"
                else:
                    integer_str = f"{timings['integer']:.3f}"
                
                print(f"{N:<4} {naive_str:<12} {complex_str:<13} {integer_str:<13} "
                      f"{complex_speedup:<10.2f} {integer_speedup:<10.2f} {int_vs_complex:<10.2f} {prime_used:<12} {status:<10}")
                
                if errors:
                    for error in errors:
                        print(f"     Error: {error}")
                        
        except Exception as e:
            if verbose:
                print(f"{N:<4} GLOBAL ERROR: {str(e)}")
            results['naive_times_ms'].append(float('inf'))
            results['complex_times_ms'].append(float('inf'))
            results['integer_times_ms'].append(float('inf'))
            results['complex_vs_naive_speedup'].append(0)
            results['integer_vs_naive_speedup'].append(0)
            results['integer_vs_complex_speedup'].append(0)
            results['primes_used'].append(0)
            results['errors'].append([f"Global: {str(e)}"])
    
    if verbose:
        print("-" * 100)
        
        # Calculate summary statistics
        valid_complex_speedups = [s for s in results['complex_vs_naive_speedup'] if s > 0 and s != float('inf')]
        valid_integer_speedups = [s for s in results['integer_vs_naive_speedup'] if s > 0 and s != float('inf')]
        valid_int_vs_complex = [s for s in results['integer_vs_complex_speedup'] if s > 0 and s != float('inf')]
        
        print("\nSUMMARY STATISTICS:")
        if valid_complex_speedups:
            avg_complex = sum(valid_complex_speedups) / len(valid_complex_speedups)
            print(f"Complex NTT vs Naive - Average speedup: {avg_complex:.2f}x")
        else:
            print("Complex NTT vs Naive - No valid measurements")
            
        if valid_integer_speedups:
            avg_integer = sum(valid_integer_speedups) / len(valid_integer_speedups)
            print(f"Integer NTT vs Naive - Average speedup: {avg_integer:.2f}x")
        else:
            print("Integer NTT vs Naive - No valid measurements")
            
        if valid_int_vs_complex:
            avg_int_vs_complex = sum(valid_int_vs_complex) / len(valid_int_vs_complex)
            print(f"Integer NTT vs Complex NTT - Average ratio: {avg_int_vs_complex:.2f}x")
        else:
            print("Integer NTT vs Complex NTT - No valid measurements")
            
        # Error summary
        total_errors = sum(len(error_list) for error_list in results['errors'])
        print(f"\nTotal errors encountered: {total_errors}")
        
        print("=" * 100)
    
    return results


def validate_transform_size(N):
    """Validate that N is of the form 2^a * 3^b where a,b >= 1."""
    if N <= 0:
        return False
    
    # Factor out powers of 2
    temp = N
    a = 0
    while temp % 2 == 0:
        temp //= 2
        a += 1
    
    # Factor out powers of 3
    b = 0
    while temp % 3 == 0:
        temp //= 3
        b += 1
    
    # Should be left with 1 if N = 2^a * 3^b, and both a,b >= 1
    return temp == 1 and a >= 1 and b >= 1


if __name__ == "__main__":
    sys.exit(main())
