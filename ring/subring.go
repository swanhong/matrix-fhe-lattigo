package ring

import (
	"fmt"
	"math/big"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/factorization"
)

// isValidRingDegreeFor3N checks if N satisfies the condition 3N = 2^a × 3^{b+1}
// where a >= 0 and b >= 0, which means 3N must be of the form 2^a × 3^{b+1}
func isValidRingDegreeFor3N(N int) bool {
	if N <= 0 {
		return false
	}

	threeN := 3 * N

	// Remove all factors of 3 from 3N
	temp := threeN
	for temp%3 == 0 {
		temp /= 3
	}

	// After removing all factors of 3, the remaining part should be a power of 2
	// Since we start with 3N, we're guaranteed to have at least one factor of 3
	return temp > 0 && (temp&(temp-1)) == 0
}

// SubRing is a struct storing precomputation
// for fast modular reduction and NTT for
// a given modulus.
type SubRing struct {
	ntt NumberTheoreticTransformer

	// Polynomial nb.Coefficients
	N int

	// Modulus
	Modulus uint64

	// Unique factors of Modulus-1
	Factors []uint64

	// 2^bit_length(Modulus) - 1
	Mask uint64

	// Fast reduction constants
	BRedConstant [2]uint64 // Barrett Reduction
	MRedConstant uint64    // Montgomery Reduction

	*NTTTable // NTT related constants
}

// NewSubRing creates a new SubRing with the standard NTT.
// NTT constants still need to be generated using .GenNTTConstants(NthRoot uint64).
// For power-of-2 rings, uses 2*N as NthRoot. For 3N rings, uses 3*N as NthRoot.
func NewSubRing(N int, Modulus uint64) (s *SubRing, err error) {
	// Determine appropriate NthRoot based on ring type
	var nthRoot int
	if isValidRingDegreeFor3N(N) {
		nthRoot = 3 * N
	} else {
		nthRoot = 2 * N // Traditional power-of-2 case
	}
	return NewSubRingWithCustomNTT(N, Modulus, NewNumberTheoreticTransformerStandard, nthRoot)
}

// NewSubRingWithCustomNTT creates a new SubRing with degree N and modulus Modulus with user-defined NTT transform and primitive Nth root of unity.
// Modulus should be equal to 1 modulo the root of unity.
// N must be either a power of 2 greater than 8, OR satisfy 3N = 2^a × 3^{b+1} condition.
func NewSubRingWithCustomNTT(N int, Modulus uint64, ntt func(*SubRing, int) NumberTheoreticTransformer, NthRoot int) (s *SubRing, err error) {

	// Checks if N is valid (either power of 2 OR 3N condition)
	isorder2 := N >= MinimumRingDegreeForLoopUnrolledOperations && (N&(N-1)) == 0
	is3NValid := isValidRingDegreeFor3N(N) && N >= MinimumRingDegreeForLoopUnrolledOperations

	if !isorder2 && !is3NValid {
		return nil, fmt.Errorf("invalid ring degree: N=%d must be either a power of 2 greater than %d, or satisfy 3N = 2^a × 3^{b+1} condition", N, MinimumRingDegreeForLoopUnrolledOperations)
	}

	if NthRoot <= 0 {
		panic(fmt.Errorf("invalid NthRoot: NthRoot=%d should be greater than 0", NthRoot))
	}

	s = &SubRing{}

	s.N = N

	s.Modulus = Modulus
	/* #nosec G115 -- Modulus is ensured to be greater than 0 */
	s.Mask = (1 << uint64(bits.Len64(Modulus-1))) - 1

	// Computes the fast modular reduction constants for the Ring
	s.BRedConstant = GenBRedConstant(Modulus)

	// If qi is not a power of 2, we can compute the MRed (otherwise, it
	// would return an error as there is no valid Montgomery form mod a power of 2)
	if (Modulus&(Modulus-1)) != 0 && Modulus != 0 {
		s.MRedConstant = GenMRedConstant(Modulus)
	}

	s.NTTTable = new(NTTTable)
	s.NthRoot = uint64(NthRoot)

	s.ntt = ntt(s, N)

	return
}

// Type returns the Type of subring which might be either `Standard` or `ConjugateInvariant`.
func (s *SubRing) Type() Type {
	switch s.ntt.(type) {
	case NumberTheoreticTransformerStandard:
		return Standard
	case NumberTheoreticTransformerConjugateInvariant:
		return ConjugateInvariant
	default:
		return Standard
	}
}

// generateNTTConstants generates the NTT constant for the target SubRing.
// The fields `PrimitiveRoot` and `Factors` can be set manually to
// bypass the search for the primitive root (which requires to
// factor Modulus-1) and speedup the generation of the constants.
func (s *SubRing) generateNTTConstants() (err error) {

	if s.N == 0 || s.Modulus == 0 {
		return fmt.Errorf("invalid t parameters (missing)")
	}

	Modulus := s.Modulus
	NthRoot := s.NthRoot

	// Checks if each qi is prime and equal to 1 mod NthRoot
	if !IsPrime(Modulus) {
		return fmt.Errorf("invalid modulus: %d is not prime)", Modulus)
	}

	if Modulus%NthRoot != 1 {
		return fmt.Errorf("invalid modulus: %d != 1 mod NthRoot)", Modulus)
	}

	// Check if this is a 3N NTT case
	is3NNTT := false
	switch s.ntt.(type) {
	case *NumberTheoreticTransformer3N:
		is3NNTT = true
	}

	// It is possible to manually set the primitive root along with the factors of q-1.
	// This is notably useful when marshalling the SubRing, to avoid re-factoring q-1.
	// If both are set, then checks that that the root is indeed primitive.
	// Else, factorize q-1 and finds a primitive root.
	if s.PrimitiveRoot != 0 && s.Factors != nil {
		if err = CheckPrimitiveRoot(s.PrimitiveRoot, s.Modulus, s.Factors); err != nil {
			return
		}
	} else {
		if is3NNTT {
			// For 3N NTT, we need a 3N-th primitive root
			if s.PrimitiveRoot, s.Factors, err = Find3NPrimitiveRoot(s.Modulus, NthRoot, s.Factors); err != nil {
				return
			}
		} else {
			// Standard case: find primitive root of q-1
			if s.PrimitiveRoot, s.Factors, err = PrimitiveRoot(s.Modulus, s.Factors); err != nil {
				return
			}
		}
	}

	// 1.1 Computes N^(-1) mod Q in Montgomery form
	if is3NNTT {
		// For 3N NTT, we need N^(-1) mod p, where N = NthRoot/3
		actualN := NthRoot / 3
		s.NInv = MForm(ModExp(actualN, Modulus-2, Modulus), Modulus, s.BRedConstant)
	} else {
		// Standard case: (NthRoot/2)^(-1) mod p
		s.NInv = MForm(ModExp(NthRoot>>1, Modulus-2, Modulus), Modulus, s.BRedConstant)
	}

	// 1.2 Computes Psi and PsiInv in Montgomery form
	PsiMont := MForm(ModExp(s.PrimitiveRoot, (Modulus-1)/NthRoot, Modulus), Modulus, s.BRedConstant)
	PsiInvMont := MForm(ModExp(s.PrimitiveRoot, Modulus-((Modulus-1)/NthRoot)-1, Modulus), Modulus, s.BRedConstant)

	s.RootsForward = make([]uint64, NthRoot>>1)
	s.RootsBackward = make([]uint64, NthRoot>>1)

	s.RootsForward[0] = MForm(1, Modulus, s.BRedConstant)
	s.RootsBackward[0] = MForm(1, Modulus, s.BRedConstant)

	half := NthRoot >> 1
	// If half is a power-of-two, we can keep bit-reversed order; otherwise, fill sequentially.
	if half != 0 && (half&(half-1)) == 0 {
		logHalf := int(bits.Len64(half) - 1)
		for j := uint64(1); j < half; j++ {
			prev := utils.BitReverse64(j-1, logHalf)
			cur := utils.BitReverse64(j, logHalf)
			s.RootsForward[cur] = MRed(s.RootsForward[prev], PsiMont, Modulus, s.MRedConstant)
			s.RootsBackward[cur] = MRed(s.RootsBackward[prev], PsiInvMont, Modulus, s.MRedConstant)
		}
	} else {
		for j := uint64(1); j < half; j++ {
			s.RootsForward[j] = MRed(s.RootsForward[j-1], PsiMont, Modulus, s.MRedConstant)
			s.RootsBackward[j] = MRed(s.RootsBackward[j-1], PsiInvMont, Modulus, s.MRedConstant)
		}
	}

	return
}

// PrimitiveRoot computes the smallest primitive root of the given prime q
// The unique factors of q-1 can be given to speed up the search for the root.
func PrimitiveRoot(q uint64, factors []uint64) (uint64, []uint64, error) {

	if factors != nil {
		if err := CheckFactors(q-1, factors); err != nil {
			return 0, factors, err
		}
	} else {

		factorsBig := factorization.GetFactors(new(big.Int).SetUint64(q - 1)) //Factor q-1, might be slow

		factors = make([]uint64, len(factorsBig))
		for i := range factors {
			factors[i] = factorsBig[i].Uint64()
		}
	}

	notFoundPrimitiveRoot := true

	var g uint64 = 2

	for notFoundPrimitiveRoot {
		g++
		for _, factor := range factors {
			// if for any factor of q-1, g^(q-1)/factor = 1 mod q, g is not a primitive root
			if ModExp(g, (q-1)/factor, q) == 1 {
				notFoundPrimitiveRoot = true
				break
			}
			notFoundPrimitiveRoot = false
		}
	}

	return g, factors, nil
}

// Find3NPrimitiveRoot finds a 3N-th primitive root for 3N NTT.
// For 3N NTT, we need ω such that ω^(3N) ≡ 1 (mod p) and ω^k ≢ 1 for proper divisors k of 3N.
func Find3NPrimitiveRoot(p uint64, NthRoot uint64, factors []uint64) (uint64, []uint64, error) {
	// First find a primitive root of p-1
	primRoot, primFactors, err := PrimitiveRoot(p, factors)
	if err != nil {
		return 0, factors, err
	}

	// For 3N NTT, we need ω^(3N) ≡ 1 (mod p)
	// Since primRoot^(p-1) ≡ 1 (mod p), we can use:
	// ω = primRoot^((p-1)/3N) mod p
	// This ensures ω^(3N) ≡ (primRoot^((p-1)/3N))^(3N) ≡ primRoot^(p-1) ≡ 1 (mod p)

	if (p-1)%NthRoot != 0 {
		return 0, primFactors, fmt.Errorf("invalid 3N configuration: (p-1) not divisible by 3N=%d", NthRoot)
	}

	omega := ModExp(primRoot, (p-1)/NthRoot, p)

	// Verify that omega is indeed a 3N-th primitive root
	if ModExp(omega, NthRoot, p) != 1 {
		return 0, primFactors, fmt.Errorf("computed omega is not a 3N-th root")
	}

	// Check that it's primitive (no smaller order)
	// Test key divisors of 3N, but skip 1 since NthRoot/1 = NthRoot should equal 1
	divisors := []uint64{2, 3, 4, 6, 8, 9, 12, 18, 24, 36}
	for _, d := range divisors {
		if NthRoot%d == 0 && d > 1 && d < NthRoot {
			if ModExp(omega, NthRoot/d, p) == 1 {
				return 0, primFactors, fmt.Errorf("omega has smaller order than 3N (fails for divisor %d, order %d)", d, NthRoot/d)
			}
		}
	}

	return omega, primFactors, nil
}

// CheckFactors checks that the given list of factors contains
// all the unique primes of m.
func CheckFactors(m uint64, factors []uint64) (err error) {

	for _, factor := range factors {

		if !IsPrime(factor) {
			return fmt.Errorf("composite factor")
		}

		for m%factor == 0 {
			m /= factor
		}
	}

	if m != 1 {
		return fmt.Errorf("incomplete factor list")
	}

	return
}

// CheckPrimitiveRoot checks that g is a valid primitive root mod q,
// given the factors of q-1.
func CheckPrimitiveRoot(g, q uint64, factors []uint64) (err error) {

	if err = CheckFactors(q-1, factors); err != nil {
		return
	}

	for _, factor := range factors {
		if ModExp(g, (q-1)/factor, q) == 1 {
			return fmt.Errorf("invalid primitive root")
		}
	}

	return
}

// subRingParametersLiteral is a struct to store the minimum information
// to uniquely identify a SubRing and be able to reconstruct it efficiently.
// This struct's purpose is to faciliate marshalling of SubRings.
type subRingParametersLiteral struct {
	Type          uint8    // Standard or ConjugateInvariant
	LogN          uint8    // Log2 of the ring degree
	NthRoot       uint8    // N/NthRoot
	Modulus       uint64   // Modulus
	Factors       []uint64 // Factors of Modulus-1
	PrimitiveRoot uint64   // Primitive root used
}

// ParametersLiteral returns the SubRingParametersLiteral of the SubRing.
func (s *SubRing) parametersLiteral() subRingParametersLiteral {
	Factors := make([]uint64, len(s.Factors))
	copy(Factors, s.Factors)
	return subRingParametersLiteral{
		/* #nosec G115 -- s.Type has is 0 or 1 */
		Type: uint8(s.Type()),
		/* #nosec G115 -- N cannot be negative if SubRing is valid */
		LogN: uint8(bits.Len64(uint64(s.N - 1))),
		/* #nosec G115 -- NthRoot cannot be negative if SubRing is valid */
		NthRoot:       uint8(int(s.NthRoot) / s.N),
		Modulus:       s.Modulus,
		Factors:       Factors,
		PrimitiveRoot: s.PrimitiveRoot,
	}
}

// newSubRingFromParametersLiteral creates a new SubRing from the provided subRingParametersLiteral.
func newSubRingFromParametersLiteral(p subRingParametersLiteral) (s *SubRing, err error) {

	s = new(SubRing)

	s.N = 1 << int(p.LogN)

	s.NTTTable = new(NTTTable)

	/* #nosec G115 -- deserialization from valid subring -> N and NthRoot cannot be negative */
	s.NthRoot = uint64(s.N) * uint64(p.NthRoot)

	s.Modulus = p.Modulus

	s.Factors = make([]uint64, len(p.Factors))
	copy(s.Factors, p.Factors)

	s.PrimitiveRoot = p.PrimitiveRoot

	/* #nosec G115 -- Modulus cannot be negative */
	s.Mask = (1 << uint64(bits.Len64(s.Modulus-1))) - 1

	// Computes the fast modular reduction parameters for the Ring
	s.BRedConstant = GenBRedConstant(s.Modulus)

	// If qi is not a power of 2, we can compute the MRed (otherwise, it
	// would return an error as there is no valid Montgomery form mod a power of 2)
	if (s.Modulus&(s.Modulus-1)) != 0 && s.Modulus != 0 {
		s.MRedConstant = GenMRedConstant(s.Modulus)
	}

	switch Type(p.Type) {
	case Standard:

		s.ntt = NewNumberTheoreticTransformerStandard(s, s.N)

		/* #nosec G115 -- library requires 64-bit system -> int = int64 */
		if int(s.NthRoot) < s.N<<1 {
			/* #nosec G115 -- library requires 64-bit system -> int = int64 */
			return nil, fmt.Errorf("invalid ring type: NthRoot must be at least 2N but is %dN", int(s.NthRoot)/s.N)
		}

	case ConjugateInvariant:

		s.ntt = NewNumberTheoreticTransformerConjugateInvariant(s, s.N)

		/* #nosec G115 -- library requires 64-bit system -> int = int64 */
		if int(s.NthRoot) < s.N<<2 {
			/* #nosec G115 -- library requires 64-bit system -> int = int64 */
			return nil, fmt.Errorf("invalid ring type: NthRoot must be at least 4N but is %dN", int(s.NthRoot)/s.N)
		}

	default:
		return nil, fmt.Errorf("invalid ring type")
	}

	return s, s.generateNTTConstants()
}
