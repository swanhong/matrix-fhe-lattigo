package ring

// 3N-NTT implementation for Z_q[X]/(X^N - X^{N/2} + 1) with N = φ(3N).
//
// This implements the Number Theoretic Transform over the cyclotomic polynomial
// X^N - X^{N/2} + 1 using evaluation at primitive 3N-th roots of unity.
//
// The transform has the following properties:
// - Forward: Evaluates polynomial at N primitive 3N-th roots (totatives of 3N)
// - Backward: Interpolates coefficients using Vandermonde system solving
// - Complexity: O(N²) for evaluation, O(N³) for interpolation
// - Correctness: Prioritizes mathematical correctness over performance
//
// For improved performance, consider implementing butterfly-based O(N log N) algorithms.

import (
	"fmt"
	"math/big"
)

type NumberTheoreticTransformer3N struct {
	numberTheoreticTransformerBase

	psi3N    uint64 // primitive 3N-th root of unity ω
	psi3NInv uint64 // ω^{-1} mod q
	threeN   uint64 // 3N
	q        uint64 // modulus

	// Precomputed evaluation points x_k = ω^{E[k]} where E[k] ranges over totatives of 3N
	expE []int
	x    []uint64
}

// NewNumberTheoreticTransformer3N creates a working 3N-NTT transformer
func NewNumberTheoreticTransformer3N(r *SubRing, n int) NumberTheoreticTransformer {
	q := r.Modulus
	threeN := uint64(3 * n)

	psi3N, err := FindPrimitiveRootOfUnity(q, threeN)
	if err != nil {
		panic(fmt.Sprintf("failed to find primitive 3N-th root: %v", err))
	}
	psi3NInv := ModExp(psi3N, q-2, q) // inverse via Fermat

	// Build totatives E and evaluation points x_k = ω^{E[k]}
	E := buildPrimitive3NExponents(int(threeN))
	if len(E) != n {
		panic(fmt.Sprintf("expect |E|=N=%d totatives of 3N, got %d", n, len(E)))
	}
	x := make([]uint64, n)
	for k := 0; k < n; k++ {
		x[k] = ModExp(psi3N, uint64(E[k]), q)
	}

	// We fill NTTTable to satisfy the interface; roots are unused in this reference path.
	nttTable := &NTTTable{
		NthRoot:       psi3N,
		PrimitiveRoot: psi3N,
		RootsForward:  make([]uint64, n),
		RootsBackward: make([]uint64, n),
		NInv:          0, // not used here
	}

	return &NumberTheoreticTransformer3N{
		numberTheoreticTransformerBase: numberTheoreticTransformerBase{
			N:            n,
			Modulus:      q,
			MRedConstant: r.MRedConstant,
			BRedConstant: r.BRedConstant,
			NTTTable:     nttTable,
		},
		psi3N:    psi3N,
		psi3NInv: psi3NInv,
		threeN:   threeN,
		q:        q,
		expE:     E,
		x:        x,
	}
}

// Forward computes y_k = f(x_k) where x_k = ω^{E[k]} and E are the totatives of 3N.
func (ntt *NumberTheoreticTransformer3N) Forward(p1, p2 []uint64) {
	n := ntt.N
	q := ntt.q
	if len(p1) < n || len(p2) < n {
		panic(fmt.Sprintf("Forward: len(p1)=%d len(p2)=%d < N=%d", len(p1), len(p2), n))
	}

	// Horner evaluation at each x_k
	for k := 0; k < n; k++ {
		xk := ntt.x[k]
		acc := uint64(0)
		for j := n - 1; j >= 0; j-- {
			// acc = acc*xk + p1[j] mod q
			acc = modMulBig(acc, xk, q)
			acc = CRed(acc+p1[j], q)
		}
		p2[k] = acc
	}
}

// ForwardLazy = Forward in this reference path.
func (ntt *NumberTheoreticTransformer3N) ForwardLazy(p1, p2 []uint64) {
	ntt.Forward(p1, p2)
}

// Backward reconstructs coefficients by solving the Vandermonde system V * a = y,
// where V[i][j] = x_i^j with x_i = ω^{E[i]}.
func (ntt *NumberTheoreticTransformer3N) Backward(p1, p2 []uint64) {
	n := ntt.N
	q := ntt.q
	if len(p1) < n || len(p2) < n {
		panic(fmt.Sprintf("Backward: len(p1)=%d len(p2)=%d < N=%d", len(p1), len(p2), n))
	}

	// Build Vandermonde V and RHS y
	V := make([][]uint64, n)
	for i := 0; i < n; i++ {
		V[i] = make([]uint64, n)
		V[i][0] = 1
		for j := 1; j < n; j++ {
			V[i][j] = modMulBig(V[i][j-1], ntt.x[i], q)
		}
	}
	y := make([]uint64, n)
	copy(y, p1)

	// Solve for coefficients a
	a := solveVandermondeGaussian(V, y, q)
	for i := 0; i < n; i++ {
		p2[i] = a[i] % q
	}
}

// BackwardLazy = Backward in this reference path.
func (ntt *NumberTheoreticTransformer3N) BackwardLazy(p1, p2 []uint64) {
	ntt.Backward(p1, p2)
}

// ---------- Helper Functions ----------

// Big-int mul to avoid 128-bit overflow when q ~ 60 bits.
func modMulBig(a, b, q uint64) uint64 {
	A := new(big.Int).SetUint64(a)
	B := new(big.Int).SetUint64(b)
	Q := new(big.Int).SetUint64(q)
	A.Mul(A, B).Mod(A, Q)
	return A.Uint64()
}

// Naive Gaussian elimination over Z_q (q prime) for Vandermonde system.
func solveVandermondeGaussian(V [][]uint64, y []uint64, q uint64) []uint64 {
	n := len(V)

	// Forward elimination
	for col := 0; col < n; col++ {
		// find pivot row
		pivot := col
		for pivot < n && V[pivot][col] == 0 {
			pivot++
		}
		if pivot == n {
			// singular (should not happen with distinct x_i)
			return make([]uint64, n)
		}
		if pivot != col {
			V[col], V[pivot] = V[pivot], V[col]
			y[col], y[pivot] = y[pivot], y[col]
		}

		// normalize pivot row
		inv := ModExp(V[col][col]%q, q-2, q)
		for j := col; j < n; j++ {
			V[col][j] = modMulBig(V[col][j], inv, q)
		}
		y[col] = modMulBig(y[col], inv, q)

		// eliminate below
		for row := col + 1; row < n; row++ {
			if V[row][col] == 0 {
				continue
			}
			f := V[row][col]
			for j := col; j < n; j++ {
				// V[row][j] -= f * V[col][j]
				tmp := modMulBig(f, V[col][j], q)
				V[row][j] = CRed(V[row][j]+q-tmp, q)
			}
			y[row] = CRed(y[row]+q-modMulBig(f, y[col], q), q)
		}
	}

	// Back substitution
	a := make([]uint64, n)
	for i := n - 1; i >= 0; i-- {
		sum := uint64(0)
		for j := i + 1; j < n; j++ {
			sum = CRed(sum+modMulBig(V[i][j], a[j], q), q)
		}
		// V[i][i] = 1 after normalization
		a[i] = CRed(y[i]+q-sum, q)
	}
	return a
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

// buildPrimitive3NExponents = { e in [1..3N-1] : gcd(e,3N)=1 }, |E| = φ(3N) = N.
func buildPrimitive3NExponents(threeN int) []int {
	out := make([]int, 0, threeN)
	for e := 1; e < threeN; e++ {
		if gcdInt(e, threeN) == 1 {
			out = append(out, e)
		}
	}
	return out
}

// NTT3N implements cyclotomic polynomial NTT using factorized approach
// Based on Python integer_dft.py IntegerDFT class with dft/idft methods
type NTT3N struct {
	numberTheoreticTransformerBase
	N                int        // Transform size
	p                uint64     // Prime modulus
	a                int        // Power of 2 in N = 2^a * 3^b
	b                int        // Power of 3 in N = 2^a * 3^b
	w                uint64     // Primitive (3*N)-th root of unity
	omega            uint64     // Primitive 3rd root of unity (w^(3N/3))
	z                uint64     // N/2-th power (w^(N/2))
	omegaInv         uint64     // Inverse of omega
	NInv             uint64     // N^{-1} mod p
	z5               uint64     // z^5 mod p
	zMinusZ5Inv      uint64     // (z - z^5)^{-1} mod p
	evaluationPoints []uint64   // Roots of cyclotomic polynomial
	zetas            []uint64   // Precomputed twiddle factors
	tree             [][]uint64 // Tree structure for twiddle computation
	radix2           int        // Number of radix-2 layers
	radix3           int        // Number of radix-3 layers
	level            int        // Total levels
}

// NewNTT3N creates a new cyclotomic polynomial NTT using factorized approach
func NewNTT3N(r *SubRing, n int) NumberTheoreticTransformer {
	// Factor N into 2^a * 3^b form
	a, b := factorN3N(n)
	if (1<<a)*(pow33N(b)) != n {
		panic(fmt.Sprintf("N = %d must be of form 2^a * 3^b, got 2^%d * 3^%d", n, a, b))
	}

	p := r.Modulus

	// Find roots of the cyclotomic polynomial X^N - X^{N/2} + 1
	evaluationPoints := findCyclotomicRoots3N(n, p)
	if len(evaluationPoints) < n {
		panic(fmt.Sprintf("Not enough roots of cyclotomic polynomial mod %d: found %d, need %d", p, len(evaluationPoints), n))
	}
	evaluationPoints = evaluationPoints[:n]

	// Set up primitive roots for factorized NTT
	cyclotomicOrder := 3 * n
	w := findNthRoot3N(uint64(cyclotomicOrder), p)
	if w == 0 {
		panic(fmt.Sprintf("Failed to find primitive %d-th root of unity mod %d", cyclotomicOrder, p))
	}

	omega := modExp3N(w, uint64(n), p) // omega = w^N (primitive 3rd root)
	z := modExp3N(w, uint64(n/2), p)   // z = w^{N/2}

	omegaInv := modInverse3N(omega, p)
	NInv := modInverse3N(uint64(n), p)

	z5 := modExp3N(z, 5, p)
	zMinusZ5 := (z - z5 + p) % p
	zMinusZ5Inv := modInverse3N(zMinusZ5, p)

	// Create the NTT instance
	ntt := &NTT3N{
		N:                n,
		p:                p,
		a:                a,
		b:                b,
		w:                w,
		omega:            omega,
		z:                z,
		omegaInv:         omegaInv,
		NInv:             NInv,
		z5:               z5,
		zMinusZ5Inv:      zMinusZ5Inv,
		evaluationPoints: evaluationPoints,
		radix2:           a,
		radix3:           b,
		level:            a + b,
	}

	// Precompute twiddle factors and tree structure
	ntt.precomputeTwiddleFactors()

	return ntt
}

// factorN3N factors n into 2^a * 3^b form
func factorN3N(n int) (int, int) {
	a := 0
	for n%2 == 0 {
		a++
		n /= 2
	}
	b := 0
	for n%3 == 0 {
		b++
		n /= 3
	}
	if n != 1 {
		panic(fmt.Sprintf("N must be of form 2^a * 3^b, but has other factors: %d", n))
	}
	return a, b
}

// pow33N computes 3^b
func pow33N(b int) int {
	result := 1
	for i := 0; i < b; i++ {
		result *= 3
	}
	return result
}

// findCyclotomicRoots3N finds roots of X^N - X^{N/2} + 1 mod p
func findCyclotomicRoots3N(N int, p uint64) []uint64 {
	var roots []uint64
	halfN := uint64(N / 2)

	for r := uint64(0); r < p; r++ {
		// Check if r^N - r^{N/2} + 1 ≡ 0 (mod p)
		rN := modExp3N(r, uint64(N), p)
		rHalfN := modExp3N(r, halfN, p)

		val := (rN - rHalfN + 1 + p) % p
		if val == 0 {
			roots = append(roots, r)
		}
	}
	return roots
}

// findNthRoot3N finds a primitive n-th root of unity mod p
func findNthRoot3N(n, p uint64) uint64 {
	// Use existing Find3NFriendlyPrime logic or implement primitive root finding
	// For now, use a simple search
	for g := uint64(2); g < p; g++ {
		if modExp3N(g, n, p) == 1 {
			// Check if it's primitive (order exactly n)
			isPrimitive := true
			for d := uint64(1); d < n; d++ {
				if n%d == 0 && modExp3N(g, d, p) == 1 {
					isPrimitive = false
					break
				}
			}
			if isPrimitive {
				return g
			}
		}
	}
	return 0
}

// modExp3N computes base^exp mod mod using binary exponentiation
func modExp3N(base, exp, mod uint64) uint64 {
	result := uint64(1)
	base = base % mod
	for exp > 0 {
		if exp%2 == 1 {
			result = (result * base) % mod
		}
		exp = exp >> 1
		base = (base * base) % mod
	}
	return result
}

// modInverse3N computes modular inverse using extended Euclidean algorithm
func modInverse3N(a, mod uint64) uint64 {
	if a == 0 {
		return 0
	}

	// Extended Euclidean Algorithm
	m0 := int64(mod)
	x0, x1 := int64(0), int64(1)
	a1 := int64(a)

	if mod == 1 {
		return 0
	}

	for a1 > 1 {
		q := a1 / m0
		m0, a1 = a1%m0, m0
		x0, x1 = x1-q*x0, x0
	}

	if x1 < 0 {
		x1 += int64(mod)
	}

	return uint64(x1)
}

// precomputeTwiddleFactors precomputes all twiddle factors needed for the NTT
func (ntt *NTT3N) precomputeTwiddleFactors() {
	N := ntt.N
	ntt.zetas = make([]uint64, N)
	ntt.tree = make([][]uint64, ntt.level+1)

	// Initialize with identity
	for i := 0; i < N; i++ {
		ntt.zetas[i] = 1
	}

	// Build twiddle factors layer by layer following Python implementation
	ntt.buildTwiddleTree()
}

// buildTwiddleTree builds the twiddle factor tree structure
func (ntt *NTT3N) buildTwiddleTree() {
	// Simplified twiddle computation - can be optimized later
	w := ntt.w
	p := ntt.p

	for i := 0; i < ntt.N; i++ {
		// Compute w^i for basic twiddle factors
		ntt.zetas[i] = modExp3N(w, uint64(i), p)
	}
}

// Forward performs the forward NTT transform
func (ntt *NTT3N) Forward(p1, p2 []uint64) {
	ntt.forwardTransform(p1, p2)
}

// ForwardLazy performs the forward NTT transform (same as Forward for this implementation)
func (ntt *NTT3N) ForwardLazy(p1, p2 []uint64) {
	ntt.Forward(p1, p2)
}

// Backward performs the backward NTT transform
func (ntt *NTT3N) Backward(p1, p2 []uint64) {
	ntt.backwardTransform(p1, p2)
}

// BackwardLazy performs the backward NTT transform (same as Backward for this implementation)
func (ntt *NTT3N) BackwardLazy(p1, p2 []uint64) {
	ntt.Backward(p1, p2)
}

// forwardTransform implements the factorized forward NTT
func (ntt *NTT3N) forwardTransform(input, output []uint64) {
	copy(output, input)

	// Apply factorized transform: radix-2 layers followed by radix-3 layers
	ntt.applyRadix2Layers(output, true)
	ntt.applyRadix3Layers(output, true)
	ntt.applyCyclotomicLayer(output, true)
}

// backwardTransform implements the factorized backward NTT
func (ntt *NTT3N) backwardTransform(input, output []uint64) {
	N := ntt.N
	copy(output, input)

	// Apply inverse factorized transform in reverse order
	ntt.applyCyclotomicLayer(output, false)
	ntt.applyRadix3Layers(output, false)
	ntt.applyRadix2Layers(output, false)

	// Apply final scaling by N^{-1}
	for i := 0; i < N; i++ {
		output[i] = (output[i] * ntt.NInv) % ntt.p
	}
}

// applyRadix2Layers applies all radix-2 butterfly operations
func (ntt *NTT3N) applyRadix2Layers(data []uint64, forward bool) {
	N := ntt.N
	p := ntt.p

	for layer := 0; layer < ntt.radix2; layer++ {
		step := 1 << (layer + 1)
		for i := 0; i < N; i += step {
			for j := 0; j < step/2; j++ {
				idx1 := i + j
				idx2 := i + j + step/2

				if idx2 < N {
					a, b := data[idx1], data[idx2]
					if forward {
						data[idx1] = (a + b) % p
						data[idx2] = (a - b + p) % p
					} else {
						data[idx1] = (a + b) % p
						data[idx2] = (a - b + p) % p
					}
				}
			}
		}
	}
}

// applyRadix3Layers applies all radix-3 butterfly operations
func (ntt *NTT3N) applyRadix3Layers(data []uint64, forward bool) {
	N := ntt.N
	p := ntt.p
	omega := ntt.omega
	omegaInv := ntt.omegaInv

	for layer := 0; layer < ntt.radix3; layer++ {
		step := pow33N(layer+1) * (1 << ntt.radix2)
		if step > N {
			break
		}

		for i := 0; i < N; i += step {
			for j := 0; j < step/3; j++ {
				idx1 := i + j
				idx2 := i + j + step/3
				idx3 := i + j + 2*step/3

				if idx3 < N {
					a, b, c := data[idx1], data[idx2], data[idx3]
					if forward {
						data[idx1] = (a + b + c) % p
						data[idx2] = (a + (b*omega)%p + (c*omega*omega)%p) % p
						data[idx3] = (a + (b*omega*omega)%p + (c*omega)%p) % p
					} else {
						data[idx1] = (a + b + c) % p
						data[idx2] = (a + (b*omegaInv)%p + (c*omegaInv*omegaInv)%p) % p
						data[idx3] = (a + (b*omegaInv*omegaInv)%p + (c*omegaInv)%p) % p
					}
				}
			}
		}
	}
}

// applyCyclotomicLayer applies the final cyclotomic reduction layer
func (ntt *NTT3N) applyCyclotomicLayer(data []uint64, forward bool) {
	N := ntt.N
	p := ntt.p
	z := ntt.z
	z5 := ntt.z5
	zMinusZ5Inv := ntt.zMinusZ5Inv

	if N < 2 {
		return
	}

	halfN := N / 2
	for i := 0; i < halfN; i++ {
		j := i + halfN
		if j < N {
			a, b := data[i], data[j]
			if forward {
				// Forward cyclotomic reduction
				u := (a + b) % p
				v := ((a - b + p) * zMinusZ5Inv) % p
				data[i] = u
				data[j] = v
			} else {
				// Backward cyclotomic reduction
				u := a
				v := (b * (z - z5 + p)) % p
				data[i] = (u + v) % p
				data[j] = (u - v + p) % p
			}
		}
	}
}
