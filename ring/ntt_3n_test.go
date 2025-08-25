package ring

import (
	"crypto/rand"
	"fmt"
	"math/big"
	mathrand "math/rand"
	"testing"
	"time"
)

// make3NSubRing builds a minimal SubRing for the step-1 (permutation-only) scaffold.
// No Montgomery/Barrett constants are needed yet.
func make3NSubRing(t *testing.T, N int, bitSize, searchBudget int) *SubRing {
	if t != nil {
		t.Helper()
	}

	qs, err := Find3NRNSPrimes(N, bitSize, 1, searchBudget)
	if err != nil {
		if t != nil {
			t.Fatalf("failed to find 3N-friendly prime: %v", err)
		} else {
			panic(err)
		}
	}
	q := qs[0]

	nttTab := &NTTTable{
		// Not used in step-1
		NthRoot:       0,
		PrimitiveRoot: 0,
		RootsForward:  nil,
		RootsBackward: nil,
		NInv:          0, // not used by the permutation scaffold
	}

	return &SubRing{
		N:            N,
		Modulus:      q,
		MRedConstant: 0,               // unused in step-1
		BRedConstant: [2]uint64{0, 0}, // unused in step-1
		NTTTable:     nttTab,
	}
}

func randVecMod(t *testing.T, n int, q uint64) []uint64 {
	if t != nil {
		t.Helper()
	}
	out := make([]uint64, n)
	for i := 0; i < n; i++ {
		r, err := rand.Int(rand.Reader, new(big.Int).SetUint64(q))
		if err != nil {
			if t != nil {
				t.Fatalf("rand error: %v", err)
			} else {
				panic(err)
			}
		}
		out[i] = r.Uint64()
	}
	return out
}

func equalMod(a, b []uint64, q uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if (a[i] % q) != (b[i] % q) {
			return false
		}
	}
	return true
}

// Test polynomial multiplication mod X^N - X^{N/2} + 1 (cyclotomic reduction)
func TestPolyMulModCyclotomic(t *testing.T) {
	N := 12
	r := make3NSubRing(t, N, 31, 1<<16)
	q := r.Modulus

	// 두 임의의 다항식 생성
	// a := randVecMod(t, N, q)
	// b := randVecMod(t, N, q)
	// a = 1 + X^5 + X^10, b = 1 + X^3 + X^6로 고정
	a := make([]uint64, N)
	b := make([]uint64, N)
	a[0], a[5], a[10] = 1, 1, 1 // 1 + 1*X^5 + 1*X^10
	b[0], b[3], b[6] = 1, 1, 1  // 1 + 1*X^3 + 1*X^6
	t.Logf("a: %v", a)
	t.Logf("b: %v", b)

	// 1. Naive 곱셈 후 cyclotomic reduction
	naiveFull := make([]uint64, 2*N-1)
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			naiveFull[i+j] = (naiveFull[i+j] + a[i]*b[j]) % q
		}
	}
	t.Logf("naiveFull (before reduction): %v", naiveFull)
	// X^N = X^{N/2} - 1로 치환하여 cyclotomic reduction
	for k := 2*N - 2; k >= N; k-- {
		naiveFull[k-N/2] = (naiveFull[k-N/2] + naiveFull[k]) % q
		naiveFull[k-N] = (naiveFull[k-N] + q - naiveFull[k]) % q
	}
	t.Logf("naiveFull (after reduction): %v", naiveFull)
	naive := make([]uint64, N)
	for i := 0; i < N; i++ {
		naive[i] = naiveFull[i] % q
	}
	t.Logf("naive (final reduced): %v", naive)

	// 2. NTT 기반 곱셈
	ntt := NewNumberTheoreticTransformer3N(r, N)
	A := make([]uint64, N)
	B := make([]uint64, N)
	C := make([]uint64, N)
	ntt.Forward(a, A)
	ntt.Forward(b, B)
	t.Logf("A (NTT(a)): %v", A)
	t.Logf("B (NTT(b)): %v", B)
	for i := 0; i < N; i++ {
		C[i] = (A[i] * B[i]) % q
	}
	t.Logf("C (pointwise mult): %v", C)
	ntt.Backward(C, C)
	t.Logf("C (after INTT): %v", C)

	// 3. 결과 비교
	if !equalMod(naive, C, q) {
		t.Fatalf("PolyMul mod (X^N-X^{N/2}+1) failed: naive=%v, ntt=%v", naive, C)
	}
}

// === tests ==================================================================

func TestNTT3N_RoundTrip_N12(t *testing.T) {
	N := 12
	r := make3NSubRing(t, N, 31, 1<<16)

	ntt := NewNumberTheoreticTransformer3N(r, N)

	in := randVecMod(t, N, r.Modulus)
	tmp := make([]uint64, N)
	out := make([]uint64, N)

	ntt.Forward(in, tmp)
	ntt.Backward(tmp, out)

	if !equalMod(in, out, r.Modulus) {
		t.Fatalf("INTT(NTT(x)) != x (mod q) for N=%d", N)
	}
}

func TestNTT3N_RoundTrip_N24(t *testing.T) {
	N := 24
	r := make3NSubRing(t, N, 31, 1<<16)

	ntt := NewNumberTheoreticTransformer3N(r, N)

	in := randVecMod(t, N, r.Modulus)
	tmp := make([]uint64, N)
	out := make([]uint64, N)

	ntt.Forward(in, tmp)
	ntt.Backward(tmp, out)

	if !equalMod(in, out, r.Modulus) {
		t.Fatalf("INTT(NTT(x)) != x (mod q) for N=%d", N)
	}
}

func TestNTT3N_MultipleRoundsStability(t *testing.T) {
	N := 24
	r := make3NSubRing(t, N, 31, 1<<16)

	ntt := NewNumberTheoreticTransformer3N(r, N)

	ref := randVecMod(t, N, r.Modulus)
	cur := append([]uint64(nil), ref...)
	tmp := make([]uint64, N)

	// Do several Forward+Backward cycles; result must remain stable mod q
	const rounds = 8
	for i := 0; i < rounds; i++ {
		ntt.Forward(cur, tmp)
		ntt.Backward(tmp, cur)
	}

	if !equalMod(ref, cur, r.Modulus) {
		t.Fatalf("multiple NTT/INTT cycles are not stable for N=%d", N)
	}
}

func TestNTT3N_ZeroAndOne(t *testing.T) {
	N := 12
	r := make3NSubRing(t, N, 31, 1<<16)

	ntt := NewNumberTheoreticTransformer3N(r, N)

	zero := make([]uint64, N)
	tmp := make([]uint64, N)
	out := make([]uint64, N)

	ntt.Forward(zero, tmp)
	ntt.Backward(tmp, out)
	if !equalMod(zero, out, r.Modulus) {
		t.Fatalf("NTT/INTT(0) != 0 for N=%d", N)
	}

	one := make([]uint64, N)
	for i := range one {
		one[i] = 1
	}
	ntt.Forward(one, tmp)
	ntt.Backward(tmp, out)
	if !equalMod(one, out, r.Modulus) {
		t.Fatalf("NTT/INTT(1) != 1 for N=%d", N)
	}
}

// Benchmark: NTT-based polynomial multiplication time for various N
func BenchmarkNTT3N_Mul(b *testing.B) {
	// Smaller sizes for working benchmark version - avoid extremely slow large sizes
	sizes := []int{6, 12, 24, 48, 96, 192}
	for _, N := range sizes {
		b.Run("N="+itoa(N), func(b *testing.B) {
			r := make3NSubRing(nil, N, 31, 1<<16)
			q := r.Modulus
			a := randVecMod(nil, N, q)
			bvec := randVecMod(nil, N, q)
			ntt := NewNumberTheoreticTransformer3N(r, N)
			A := make([]uint64, N)
			B := make([]uint64, N)
			C := make([]uint64, N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ntt.Forward(a, A)
				ntt.Forward(bvec, B)
				for j := 0; j < N; j++ {
					C[j] = (A[j] * B[j]) % q
				}
				ntt.Backward(C, C)
			}
			// ms/op, KB/op 커스텀 메트릭 추가
			if b.N > 0 {
				nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
				b.ReportMetric(nsPerOp/1e6, "ms/op")
			}
		})
	}
}

// Helper for int to string (no strconv import)
func itoa(n int) string {
	return fmt.Sprintf("%d", n)
}

// Benchmark: Original NTT-based polynomial multiplication time for various N
func BenchmarkNTT_Original_Mul(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048, 4096, 8192, 16384}
	for _, N := range sizes {
		b.Run("N="+itoa(N), func(b *testing.B) {
			// 표준 NTT용 적당한 bitSize의 prime 찾기
			bitSize := 64
			nthRoot := uint64(2 * N)
			gen := NewNTTFriendlyPrimesGenerator(uint64(bitSize), nthRoot)
			q, err := gen.NextUpstreamPrime()
			if err != nil {
				b.Skipf("NTTFriendlyPrimesGenerator failed for N=%d: %v", N, err)
				return
			}
			r, err := NewRing(N, []uint64{q})
			if err != nil {
				b.Skipf("NewRing failed for N=%d: %v", N, err)
				return
			}
			a := randVecMod(nil, N, q)
			bvec := randVecMod(nil, N, q)
			ntt := NewNumberTheoreticTransformerStandard(r.SubRings[0], N)
			A := make([]uint64, N)
			B := make([]uint64, N)
			C := make([]uint64, N)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ntt.Forward(a, A)
				ntt.Forward(bvec, B)
				for j := 0; j < N; j++ {
					C[j] = (A[j] * B[j]) % q
				}
				ntt.Backward(C, C)
			}
			if b.N > 0 {
				nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
				b.ReportMetric(nsPerOp/1e6, "ms/op")
			}
		})
	}
}

// findTestPrime3N finds a suitable prime for 3N-NTT testing
func findTestPrime3N(N int, bitSize int) (uint64, error) {
	// Find a 3N-friendly prime (p ≡ 1 (mod 3N)) for 3N-NTT
	return Find3NFriendlyPrime(N, bitSize)
}

// naiveCyclotomicMultiply3N performs naive polynomial multiplication in the cyclotomic ring
// Z_q[X]/(X^N - X^{N/2} + 1), serving as ground truth for testing NTT-based multiplication
// The reduction rule is: X^N ≡ X^{N/2} - 1
func naiveCyclotomicMultiply3N(p1, p2 []uint64, N int, q uint64) []uint64 {
	// Standard convolution
	result := make([]int64, 2*N-1)
	for i := 0; i < len(p1); i++ {
		for j := 0; j < len(p2); j++ {
			result[i+j] += int64(p1[i]) * int64(p2[j])
		}
	}

	// Reduce modulo X^N - X^{N/2} + 1
	// The reduction rule is: X^N = X^{N/2} - 1
	reduced := make([]int64, N)

	// Copy coefficients for powers 0 to N-1
	for i := 0; i < N; i++ {
		reduced[i] = result[i]
	}

	// For powers >= N, apply the reduction rule
	for i := len(result) - 1; i >= N; i-- {
		if result[i] != 0 {
			// X^i = X^{i-N} * X^N = X^{i-N} * (X^{N/2} - 1)
			// = X^{i-N+N/2} - X^{i-N}

			highPower := i - N + N/2
			lowPower := i - N

			if highPower < N {
				reduced[highPower] += result[i]
			} else {
				// If high_power >= N, we need another reduction
				result[highPower] += result[i]
			}

			if lowPower < N {
				reduced[lowPower] -= result[i]
			}
		}
	}

	// Convert back to uint64 and apply modular reduction
	output := make([]uint64, N)
	for i := 0; i < N; i++ {
		// Handle negative coefficients
		val := reduced[i] % int64(q)
		if val < 0 {
			val += int64(q)
		}
		output[i] = uint64(val)
	}

	return output
}

// TestNTT3NRoundTripExtensive tests round-trip functionality across multiple N values
func TestNTT3NRoundTripExtensive(t *testing.T) {
	testCases := []struct {
		N        int
		numTests int
	}{
		{6, 50},
		{12, 50},
		{18, 30},
		{24, 30},
		{36, 20},
		{48, 20},
	}

	t.Logf("=== ROUND-TRIP TEST RESULTS ===")
	t.Logf("%-6s %-8s %-10s %-12s %-8s", "N", "Prime", "Tests", "Passed", "Rate")
	t.Logf("%-6s %-8s %-10s %-12s %-8s", "---", "-----", "-----", "------", "----")

	totalPassed := 0
	totalTests := 0

	for _, tc := range testCases {
		// Find a suitable prime for this N
		q, err := findTestPrime3N(tc.N, 12)
		if err != nil {
			t.Logf("%-6d %-8s %-10d %-12s %-8s", tc.N, "ERROR", tc.numTests, "FAILED", "0.0%")
			continue
		}

		// Create a SubRing structure
		subRing := &SubRing{
			Modulus:      q,
			MRedConstant: ((uint64(1) << 63) / q) << 1,
			BRedConstant: [2]uint64{0, 0},
		}

		// Create 3N-NTT (use the working reference implementation)
		ntt := NewNumberTheoreticTransformer3N(subRing, tc.N)
		ntt3N := ntt.(*NumberTheoreticTransformer3N)

		// Run round-trip tests
		passed := 0
		rng := mathrand.New(mathrand.NewSource(time.Now().UnixNano()))

		for i := 0; i < tc.numTests; i++ {
			// Generate random polynomial
			input := make([]uint64, tc.N)
			for j := 0; j < tc.N; j++ {
				input[j] = uint64(rng.Intn(int(q)))
			}

			// Test round-trip
			nttResult := make([]uint64, tc.N)
			ntt3N.Forward(input, nttResult)

			recovered := make([]uint64, tc.N)
			ntt3N.Backward(nttResult, recovered)

			// Check if round-trip worked
			success := true
			for k := 0; k < tc.N; k++ {
				if recovered[k] != input[k] {
					success = false
					break
				}
			}

			if success {
				passed++
			}
		}

		successRate := float64(passed) / float64(tc.numTests) * 100
		t.Logf("%-6d %-8d %-10d %-12d %-8.1f%%", tc.N, q, tc.numTests, passed, successRate)

		totalPassed += passed
		totalTests += tc.numTests

		if passed != tc.numTests {
			t.Errorf("Round-trip tests failed for N=%d: %d/%d passed", tc.N, passed, tc.numTests)
		}
	}

	overallRate := float64(totalPassed) / float64(totalTests) * 100
	t.Logf("%-6s %-8s %-10d %-12d %-8.1f%%", "TOTAL", "", totalTests, totalPassed, overallRate)
	t.Logf("=== END ROUND-TRIP RESULTS ===")
}

// TestNTT3NPolynomialMultiplicationExtensive tests polynomial multiplication across multiple N values
func TestNTT3NPolynomialMultiplicationExtensive(t *testing.T) {
	testCases := []struct {
		N        int
		numTests int
	}{
		{6, 25},
		{12, 25},
		{18, 20},
		{24, 20},
		{36, 15},
		{48, 15},
	}

	t.Logf("=== POLYNOMIAL MULTIPLICATION TEST RESULTS ===")
	t.Logf("%-6s %-8s %-10s %-12s %-8s", "N", "Prime", "Tests", "Passed", "Rate")
	t.Logf("%-6s %-8s %-10s %-12s %-8s", "---", "-----", "-----", "------", "----")

	totalPassed := 0
	totalTests := 0

	for _, tc := range testCases {
		// Find a suitable prime for this N
		q, err := findTestPrime3N(tc.N, 12)
		if err != nil {
			t.Logf("%-6d %-8s %-10d %-12s %-8s", tc.N, "ERROR", tc.numTests, "FAILED", "0.0%")
			continue
		}

		// Create a SubRing structure
		subRing := &SubRing{
			Modulus:      q,
			MRedConstant: ((uint64(1) << 63) / q) << 1,
			BRedConstant: [2]uint64{0, 0},
		}

		// Create 3N-NTT (use the working reference implementation)
		ntt := NewNumberTheoreticTransformer3N(subRing, tc.N)
		ntt3N := ntt.(*NumberTheoreticTransformer3N)

		// Run polynomial multiplication tests
		passed := 0
		rng := mathrand.New(mathrand.NewSource(time.Now().UnixNano()))

		for i := 0; i < tc.numTests; i++ {
			// Generate random polynomials with coefficients in range [0, q/4) to avoid overflow
			maxCoeff := q / 4
			p1 := make([]uint64, tc.N)
			p2 := make([]uint64, tc.N)

			for j := 0; j < tc.N; j++ {
				p1[j] = uint64(rng.Intn(int(maxCoeff)))
				p2[j] = uint64(rng.Intn(int(maxCoeff)))
			}

			// Compute ground truth using naive cyclotomic multiplication
			expected := naiveCyclotomicMultiply3N(p1, p2, tc.N, q)

			// Compute using NTT
			A := make([]uint64, tc.N)
			B := make([]uint64, tc.N)
			ntt3N.Forward(p1, A)
			ntt3N.Forward(p2, B)

			C := make([]uint64, tc.N)
			for k := 0; k < tc.N; k++ {
				C[k] = (A[k] * B[k]) % q
			}

			nttResult := make([]uint64, tc.N)
			ntt3N.Backward(C, nttResult)

			// Compare results
			success := true
			for k := 0; k < tc.N; k++ {
				if expected[k] != nttResult[k] {
					success = false
					break
				}
			}

			if success {
				passed++
			}
		}

		successRate := float64(passed) / float64(tc.numTests) * 100
		t.Logf("%-6d %-8d %-10d %-12d %-8.1f%%", tc.N, q, tc.numTests, passed, successRate)

		totalPassed += passed
		totalTests += tc.numTests

		if passed != tc.numTests {
			t.Errorf("Polynomial multiplication tests failed for N=%d: %d/%d passed", tc.N, passed, tc.numTests)
		}
	}

	overallRate := float64(totalPassed) / float64(totalTests) * 100
	t.Logf("%-6s %-8s %-10d %-12d %-8.1f%%", "TOTAL", "", totalTests, totalPassed, overallRate)
	t.Logf("=== END POLYNOMIAL MULTIPLICATION RESULTS ===")
}
