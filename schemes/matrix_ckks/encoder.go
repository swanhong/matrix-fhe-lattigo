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

// Encoder is a type that implements the encoding and decoding interface for the Matrix CKKS scheme.
// It provides methods to encode/decode []complex128 and []float64 types into/from Plaintext types
// using 3N-ring structure instead of power-of-2 rings.
//
// The encoding works similarly to CKKS but adapts to the 3N-ring structure for better matrix operations.
type Encoder struct {
	parameters   Parameters
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
func NewEncoder(parameters Parameters, precision ...uint) (ecd *Encoder) {
	n := parameters.N()

	var prec uint
	if len(precision) != 0 && precision[0] != 0 {
		prec = precision[0]
	} else {
		prec = parameters.EncodingPrecision()
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
func (ecd *Encoder) GetParameters() Parameters {
	return ecd.parameters
}

// EncodePolynomial encodes a polynomial (as uint64 coefficients) into a plaintext.
// This is for basic polynomial operations without complex encoding.

// TODO: construct an encoding using iDFT
func (ecd *Encoder) EncodePolynomial(values []uint64, pt *rlwe.Plaintext) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// For debugging: don't use scale, just set coefficients directly
	// Set coefficients directly without scaling
	for i, val := range values {
		pt.Value.Coeffs[0][i] = val
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
// This is for basic polynomial operations without complex decoding.
func (ecd *Encoder) DecodePolynomial(pt *rlwe.Plaintext, values []uint64) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// For debugging: extract coefficients directly without scale division
	var rawValues []uint64

	// If the plaintext is in NTT domain, we need to convert it back to coefficient domain first
	if pt.IsNTT {
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		ringQ.INTT(pt.Value, ecd.buff)

		// Extract raw coefficients from the converted polynomial
		rawValues = make([]uint64, len(values))
		for i := range rawValues {
			if i < len(ecd.buff.Coeffs[0]) {
				rawValues[i] = ecd.buff.Coeffs[0][i]
			} else {
				rawValues[i] = 0
			}
		}
	} else {
		// Extract raw coefficients directly if already in coefficient domain
		rawValues = make([]uint64, len(values))
		for i := range rawValues {
			if i < len(pt.Value.Coeffs[0]) {
				rawValues[i] = pt.Value.Coeffs[0][i]
			} else {
				rawValues[i] = 0
			}
		}
	}

	// For debugging: just copy raw values directly
	copy(values, rawValues)

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
