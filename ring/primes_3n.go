package ring

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

// Find3NRNSPrimes finds 'count' distinct primes p ~= 2^bitSize with p ≡ 1 (mod 3N).
// Uses your IsPrime and simple stepping; returns an error if not enough are found within searchBudget steps.
func Find3NRNSPrimes(N, bitSize, count, searchBudget int) ([]uint64, error) {
	if N <= 0 || bitSize <= 2 || count <= 0 {
		return nil, fmt.Errorf("invalid args")
	}
	threeN := uint64(3 * N)
	base := uint64(1) << bitSize

	// start from first multiple of 3N >= base, then +1 to be in the residue class 1 mod 3N
	start := ((base-1)/threeN + 1) * threeN
	candidate := start + 1

	var out []uint64
	seen := make(map[uint64]struct{})

	steps := 0
	for steps < searchBudget && len(out) < count {
		if candidate%threeN == 1 && IsPrime(candidate) {
			if _, ok := seen[candidate]; !ok {
				out = append(out, candidate)
				seen[candidate] = struct{}{}
			}
		}
		if candidate > ^uint64(0)-threeN {
			break // overflow guard
		}
		candidate += threeN
		steps++
	}
	if len(out) < count {
		return nil, fmt.Errorf("could not find enough 3N-friendly primes (found %d/%d)", len(out), count)
	}
	return out, nil
}

// mulMod performs (a*b) mod m using big.Int to avoid overflow in uint64 intermediate.
func mulMod(a, b, m uint64) uint64 {
	A := new(big.Int).SetUint64(a)
	B := new(big.Int).SetUint64(b)
	M := new(big.Int).SetUint64(m)
	A.Mul(A, B).Mod(A, M)
	return A.Uint64()
}

func powMod(base, exp, mod uint64) uint64 {
	res := uint64(1)
	b := base % mod
	e := exp
	for e > 0 {
		if e&1 == 1 {
			res = mulMod(res, b, mod)
		}
		b = mulMod(b, b, mod)
		e >>= 1
	}
	return res
}

// gcd64 is Euclid's algorithm.
func gcd64(a, b uint64) uint64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// orderMod returns the multiplicative order of 'g' modulo prime 'p', assuming g != 0 mod p.
// Requires that p is prime and g in (Z/pZ)^*.
func OrderMod(g, p uint64) uint64 {
	phi := p - 1
	// factor phi (p-1). Small factorization is fine for our smooth 3N case.
	factors := primeFactors(phi)

	order := phi
	for _, f := range factors {
		for order%f == 0 && powMod(g, order/uint64(f), p) == 1 {
			order /= uint64(f)
		}
	}
	return order
}

// primeFactors returns the list of prime factors of n with multiplicity removed (set of primes).
func primeFactors(n uint64) []uint64 {
	var fs []uint64
	x := n
	for x%2 == 0 {
		if len(fs) == 0 || fs[len(fs)-1] != 2 {
			fs = append(fs, 2)
		}
		x /= 2
	}
	var f uint64 = 3
	for f*f <= x {
		if x%f == 0 {
			if len(fs) == 0 || fs[len(fs)-1] != f {
				fs = append(fs, f)
			}
			x /= f
		} else {
			f += 2
		}
	}
	if x > 1 {
		fs = append(fs, x)
	}
	return fs
}

// HasPrimitiveMthRoot reports whether modulus q supports a primitive m-th root of unity,
// i.e., (q-1) % m == 0.
func HasPrimitiveMthRoot(q, m uint64) bool {
	return (q-1)%m == 0
}

// FindPrimitiveRootOfUnity finds a primitive m-th root of unity modulo prime q.
// Requires (q-1) % m == 0. It tries random bases until one has exact order m.
func FindPrimitiveRootOfUnity(q, m uint64) (uint64, error) {
	if !HasPrimitiveMthRoot(q, m) {
		return 0, fmt.Errorf("no primitive m-th root exists: (q-1) not divisible by m")
	}
	// We need an element g of order exactly m. Search by random sampling.
	// Draw a random x in [2..q-2], set g = x^((q-1)/m) mod q, then check order(g)=m.
	exp := (q - 1) / m
	maxTries := 256
	for i := 0; i < maxTries; i++ {
		x, err := randUint64InRange(2, q-2)
		if err != nil {
			return 0, err
		}
		g := powMod(x, exp, q)
		if g == 0 || g == 1 {
			continue
		}
		if OrderMod(g, q) == m {
			return g, nil
		}
	}
	return 0, fmt.Errorf("failed to find primitive m-th root after %d tries", 256)
}

func randUint64InRange(lo, hi uint64) (uint64, error) {
	if lo >= hi {
		return 0, fmt.Errorf("invalid range")
	}
	span := new(big.Int).SetUint64(hi - lo + 1)
	n, err := rand.Int(rand.Reader, span)
	if err != nil {
		return 0, err
	}
	return lo + n.Uint64(), nil
}

// Derive3NRoots takes q (prime), N=2^a*3^b, and a primitive 3N-th root ω mod q,
// and returns (ω, ω1, ω2) where ω1 has order 2^a and ω2 has order 3^{b+1}.
func Derive3NRoots(q uint64, a, b int, omega uint64) (omega3N, omega2Pow, omega3Pow uint64, err error) {
	if a <= 0 || b < 0 {
		return 0, 0, 0, fmt.Errorf("invalid a,b")
	}
	n := uint64(1<<a) * Uint64Pow(3, uint64(b))
	m := 3 * n // 3N
	if !HasPrimitiveMthRoot(q, m) {
		return 0, 0, 0, fmt.Errorf("q=%d does not support primitive 3N-th root", q)
	}
	// Check omega has exact order 3N
	if OrderMod(omega, q) != uint64(m) {
		return 0, 0, 0, fmt.Errorf("omega does not have order 3N")
	}
	omega1 := powMod(omega, Uint64Pow(3, uint64(b+1)), q) // order 2^a
	omega2 := powMod(omega, uint64(1<<a), q)              // order 3^{b+1}
	return omega, omega1, omega2, nil
}

func Uint64Pow(base, exp uint64) uint64 {
	r := uint64(1)
	for exp > 0 {
		if exp&1 == 1 {
			r *= base
		}
		base *= base
		exp >>= 1
	}
	return r
}
