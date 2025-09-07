package matrix_ckks

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Float interface for supported numeric types
type Float interface {
	float64 | complex128 | *big.Float | *bignum.Complex
}

// FloatSlice is an empty interface whose goal is to
// indicate that the expected input should be []Float.
type FloatSlice interface{}

// computeInputPermutation computes the input permutation for ternary rings.
// This is equivalent to the fft_input_permutation function from the Python reference.
// For numbers of the form 2^a * 3^b, this creates a permutation that optimizes
// the mixed-radix FFT computation order.
func computeInputPermutation(n int) []int {
	// Factor n into 2^a * 3^b form
	a, b := factorN3N(n)
	if (1<<a)*pow3(b) != n {
		panic(fmt.Sprintf("n = %d must be of form 2^a * 3^b, got 2^%d * 3^%d", n, a, b))
	}
	if a < 1 {
		panic(fmt.Sprintf("n must be even, got n = %d", n))
	}

	index := make([]int, n)
	length := 1
	index[0] = 0

	shift := n >> 1
	for i := 0; i < length; i++ {
		index[length+i] = index[i] + shift
	}
	length *= 2

	// Apply radix-3 layers
	for i := 0; i < b; i++ {
		shift /= 3
		baseLen := length
		for j := 0; j < baseLen; j++ {
			index[length+j] = index[j] + shift
		}
		length += baseLen
		for j := 0; j < baseLen; j++ {
			index[length+j] = index[j] + 2*shift
		}
		length += baseLen
	}

	// Apply remaining radix-2 layers
	for i := 0; i < a-1; i++ {
		shift /= 2
		baseLen := length
		for j := 0; j < baseLen; j++ {
			index[length+j] = index[j] + shift
		}
		length += baseLen
	}

	return index
}

// computeInversePermutation computes the inverse of a permutation.
func computeInversePermutation(perm []int) []int {
	inv := make([]int, len(perm))
	for i, p := range perm {
		inv[p] = i
	}
	return inv
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
		panic(fmt.Sprintf("n must be of form 2^a * 3^b, but has other factors: %d", n))
	}
	return a, b
}

// pow3 computes 3^b
func pow3(b int) int {
	result := 1
	for i := 0; i < b; i++ {
		result *= 3
	}
	return result
}

// Encoder is a type that implements the encoding and decoding interface for the Matrix CKKS scheme.
// It provides methods to encode/decode []complex128 and []float64 types into/from Plaintext types
// using 3N-ring structure instead of power-of-2 rings.
//
// The encoding works similarly to CKKS but adapts to the 3N-ring structure for better matrix operations.
type Encoder struct {
	parameters   *rlwe.Parameters3N
	prec         uint
	bigintCoeffs []*big.Int
	qHalf        *big.Int
	buff         ring.Poly
	n            int // Ring degree N = 2^a * 3^b
}

// NewEncoder creates a new [Encoder] from the target parameters.
// Optional field `precision` can be given. If precision is empty
// or <= 53, then float64 and complex128 types will be used to
// perform the encoding. Else *[big.Float] and *[bignum.Complex] will be used.
func NewEncoder(parameters *rlwe.Parameters3N, precision ...uint) (ecd *Encoder) {
	n := parameters.N()

	var prec uint
	if len(precision) != 0 && precision[0] != 0 {
		prec = precision[0]
	} else {
		prec = 53 // Default precision for float64
	}

	ecd = &Encoder{
		prec:         prec,
		parameters:   parameters,
		bigintCoeffs: make([]*big.Int, n),
		qHalf:        bignum.NewInt(0),
		buff:         parameters.RingQ().NewPoly(),
		n:            n,
	}

	for i := range ecd.bigintCoeffs {
		ecd.bigintCoeffs[i] = bignum.NewInt(0)
	}

	return ecd
}

// Prec returns the precision in bits used by the [Encoder].
func (ecd *Encoder) Prec() uint {
	return ecd.prec
}

// Parameters returns the parameters of the [Encoder].
func (ecd *Encoder) GetParameters() *rlwe.Parameters3N {
	return ecd.parameters
}

// EncodePolynomial encodes a polynomial (as uint64 coefficients) into a plaintext.
// This is for basic polynomial operations with proper scaling.
func (ecd *Encoder) EncodePolynomial(values []uint64, pt *rlwe.Plaintext) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Get the plaintext scale
	scale := pt.Scale.Uint64()
	ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
	modulus := ringQ.SubRings[0].Modulus

	// Encode values with proper fixed-point encoding
	for i, val := range values {
		// Scale the value by the plaintext scale
		scaledVal := val * scale

		// Apply proper modular arithmetic (centering around modulus)
		if scaledVal >= modulus>>1 {
			pt.Value.Coeffs[0][i] = modulus - scaledVal
		} else {
			pt.Value.Coeffs[0][i] = scaledVal
		}
	}

	// Zero out remaining coefficients
	for i := len(values); i < ecd.n; i++ {
		pt.Value.Coeffs[0][i] = 0
	}

	// Mark the plaintext as being in coefficient domain (not NTT domain)
	pt.IsNTT = false

	return nil
}

// DecodePolynomial extracts polynomial coefficients as uint64 values.
// This is for basic polynomial operations with proper scaling.
func (ecd *Encoder) DecodePolynomial(pt *rlwe.Plaintext, values []uint64) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Get the plaintext scale
	scale := pt.Scale.Uint64()

	// If the plaintext is in NTT domain, we need to convert it back to coefficient domain first
	if pt.IsNTT {
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		ringQ.INTT(pt.Value, ecd.buff)

		// Extract and decode coefficients from the converted polynomial
		modulus := ringQ.SubRings[0].Modulus

		for i := range values {
			var rawVal uint64
			if i < len(ecd.buff.Coeffs[0]) {
				rawVal = ecd.buff.Coeffs[0][i]
			} else {
				rawVal = 0
			}

			// Apply proper modular arithmetic (centering around modulus)
			var centeredVal uint64
			if rawVal >= modulus>>1 {
				centeredVal = modulus - rawVal
			} else {
				centeredVal = rawVal
			}

			// Decode by dividing by the scale with proper rounding (Standard Lattigo method)
			if scale > 0 {
				// Add scale/2 for rounding before division
				roundedVal := centeredVal + scale/2
				values[i] = roundedVal / scale
			} else {
				values[i] = 0
			}
		}
	} else {
		// Extract and decode coefficients directly if already in coefficient domain
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		modulus := ringQ.SubRings[0].Modulus

		for i := range values {
			var rawVal uint64
			if i < len(pt.Value.Coeffs[0]) {
				rawVal = pt.Value.Coeffs[0][i]
			} else {
				rawVal = 0
			}

			// Apply proper modular arithmetic (centering around modulus)
			var centeredVal uint64
			if rawVal >= modulus>>1 {
				centeredVal = modulus - rawVal
			} else {
				centeredVal = rawVal
			}

			// Decode by dividing by the scale with proper rounding (Standard Lattigo method)
			if scale > 0 {
				// Add scale/2 for rounding before division
				roundedVal := centeredVal + scale/2
				values[i] = roundedVal / scale
			} else {
				values[i] = 0
			}
		}
	}

	return nil
}

// ShallowCopy creates a shallow copy of this [Encoder] in which all the read-only data-structures are
// shared with the receiver and the temporary buffers are reallocated. The receiver and the returned
// [Encoder] can be used concurrently.
func (ecd *Encoder) ShallowCopy() *Encoder {
	return &Encoder{
		parameters:   ecd.parameters,
		prec:         ecd.prec,
		bigintCoeffs: make([]*big.Int, len(ecd.bigintCoeffs)),
		qHalf:        bignum.NewInt(0),
		buff:         ecd.parameters.RingQ().NewPoly(),
		n:            ecd.n,
	}
}
