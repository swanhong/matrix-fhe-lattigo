package ring

import (
	"testing"
)

// helper: small primes 1 mod 36
// 73 % 36 = 1, 109 % 36 = 1
var mod36Primes = []uint64{73, 109}

func TestIsValidRingDegreeFor3N(t *testing.T) {
	if !isValidRingDegreeFor3N(12) {
		t.Fatalf("expected N=12 to be valid for 3N form")
	}
	if isValidRingDegreeFor3N(10) {
		t.Fatalf("expected N=10 to be invalid for 3N form")
	}
	if !isValidRingDegreeFor3N(8) {
		t.Fatalf("expected N=8 to be valid for 3N form")
	}
}

func TestNewRing3NStandard(t *testing.T) {
	N := 12
	r, err := NewRing(N, mod36Primes)
	if err != nil {
		t.Fatalf("NewRing(3N) failed: %v", err)
	}
	if got, want := r.N(), N; got != want {
		t.Fatalf("N mismatch: got %d, want %d", got, want)
	}
	if got, want := r.NthRoot(), uint64(3*N); got != want {
		t.Fatalf("NthRoot mismatch: got %d, want %d", got, want)
	}
	// Moduli congruence check
	for _, qi := range r.ModuliChain() {
		if qi%(uint64(3*N)) != 1 {
			t.Fatalf("modulus %d not congruent to 1 mod %d", qi, 3*N)
		}
	}
}

func Test3NPrimitiveRootAndDecomposition(t *testing.T) {
	// N=12 => 3N=36, a=2 (2^a=4), b=1 (3^{b+1}=9)
	a, b := 2, 1
	N := (1 << a) * 3
	m := uint64(3 * N)

	p := uint64(1117) // 1116 divisible by 36, small & convenient
	if (p-1)%m != 0 {
		t.Fatalf("p-1 not divisible by 3N: p=%d, 3N=%d", p, m)
	}
	omega, factors, err := Find3NPrimitiveRoot(p, m, nil)
	t.Logf("Found primitive 3N-th root omega=%d with factors %v", omega, factors)
	if err != nil {
		t.Fatalf("Find3NPrimitiveRoot failed: %v (factors=%v)", err, factors)
	}	
	
	// Check exact order m
	if got := ModExp(omega, m, p); got != 1 {
		t.Fatalf("omega^m != 1 mod p")
	}
	// Derive 2^a and 3^{b+1} parts
	omega1 := ModExp(omega, Uint64Pow(3, uint64(b+1)), p) // order 2^a
	omega2 := ModExp(omega, uint64(1<<a), p)              // order 3^{b+1}

	if got, want := OrderMod(omega1, p), uint64(1<<a); got != want {
		t.Fatalf("order(omega1)=%d, want %d", got, want)
	}
	if got, want := OrderMod(omega2, p), Uint64Pow(3, uint64(b+1)); got != want {
		t.Fatalf("order(omega2)=%d, want %d", got, want)
	}
}
