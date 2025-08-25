# 3N-NTT Testing Guide

This guide shows how to run the 3N-NTT (Number Theoretic Transform) tests in this project.

## Prerequisites

- Go 1.19 or later
- Clone this repository

## Quick Start

Navigate to the project root and run:

```bash
cd /path/to/matrix-fhe-lattigo
```

## Available 3N-NTT Tests

### 1. Run All 3N-NTT Tests
```bash
go test -v ./ring -run "TestNTT3N"
```

### 2. Individual Test Functions

#### Basic Round-Trip Tests
```bash
# Test N=12 round-trip
go test -v ./ring -run "TestNTT3N_RoundTrip_N12"

# Test N=24 round-trip  
go test -v ./ring -run "TestNTT3N_RoundTrip_N24"

# Test stability across multiple rounds
go test -v ./ring -run "TestNTT3N_MultipleRoundsStability"

# Test edge cases (zero and one)
go test -v ./ring -run "TestNTT3N_ZeroAndOne"
```

#### Comprehensive Tests
```bash
# Extensive round-trip testing (multiple N values)
go test -v ./ring -run "TestNTT3NRoundTripExtensive"

# Extensive polynomial multiplication testing  
go test -v ./ring -run "TestNTT3NPolynomialMultiplicationExtensive"
```

#### Utility Tests
```bash
# Test polynomial multiplication mod cyclotomic
go test -v ./ring -run "TestPolyMulModCyclotomic"
```

## Expected Output

When tests pass, you should see output like:
```
=== RUN   TestNTT3NRoundTripExtensive
    ntt_3n_test.go:380: === ROUND-TRIP TEST RESULTS ===
    ntt_3n_test.go:381: N      Prime    Tests      Passed       Rate    
    ntt_3n_test.go:382: ---    -----    -----      ------       ----    
    ntt_3n_test.go:439: 6      4159     50         50           100.0   %
    ntt_3n_test.go:439: 12     4177     50         50           100.0   %
    ntt_3n_test.go:439: 18     4159     30         30           100.0   %
    ntt_3n_test.go:439: 24     4177     30         30           100.0   %
    ntt_3n_test.go:439: 36     4861     20         20           100.0   %
    ntt_3n_test.go:439: 48     4177     20         20           100.0   %
    ntt_3n_test.go:450: TOTAL           200        200          100.0   %
    ntt_3n_test.go:451: === END ROUND-TRIP RESULTS ===
--- PASS: TestNTT3NRoundTripExtensive (0.13s)
PASS
```

## Test What They Do

- **Round-Trip Tests**: Verify that `Forward(x)` followed by `Inverse()` returns the original polynomial
- **Polynomial Multiplication Tests**: Verify that NTT-based polynomial multiplication produces correct results
- **Stability Tests**: Ensure repeated transforms don't accumulate errors
- **Edge Case Tests**: Test special inputs like zero and one polynomials

## File Locations

- **Implementation**: `ring/ntt_3n.go`
- **Tests**: `ring/ntt_3n_test.go`
- **Prime Generation**: `ring/primes_3n.go`

## Troubleshooting

If tests fail:
1. Make sure you're in the project root directory
2. Ensure Go modules are properly initialized: `go mod tidy`
3. Check that all dependencies are available: `go mod download`

## Performance Testing

To run benchmarks:
```bash
go test -bench=. ./ring
```

## Implementation Details

The 3N-NTT uses:
- Cyclotomic polynomial structure X^N + 1
- 3N-friendly primes (p ≡ 1 mod 3N)
- Radix-2 and radix-3 decomposition for efficient computation
- Forward and inverse transforms with proper normalization
