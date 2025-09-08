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

// Encode encodes values into a plaintext, supporting both polynomial coefficients and complex vectors
func (ecd *Encoder) Encode(values interface{}, pt *rlwe.Plaintext) (err error) {
	switch values := values.(type) {
	case []uint64:
		// Polynomial coefficient encoding (original Matrix CKKS approach)
		return ecd.EncodePolynomial(values, pt)
	case []float64:
		// Float64 vector encoding (like original CKKS)
		return ecd.EncodeFloat64(values, pt)
	case []complex128:
		// Complex vector encoding (like original CKKS)
		return ecd.EncodeComplex(values, pt)
	default:
		return fmt.Errorf("cannot Encode: supported types are []uint64, []float64, or []complex128, but %T was given", values)
	}
}

// EncodePolynomial encodes a polynomial (as uint64 coefficients) into a plaintext.
// This function follows the original CKKS approach exactly.
func (ecd *Encoder) EncodePolynomial(values []uint64, pt *rlwe.Plaintext) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Convert uint64 values to float64 for encoding (same as original CKKS)
	floatValues := make([]float64, len(values))
	for i, val := range values {
		floatValues[i] = float64(val)
	}

	// Use the original CKKS encoding approach
	// This uses pt.Scale.Float64() directly, just like original CKKS
	Float64ToFixedPointCRT(ecd.parameters.RingQ().AtLevel(pt.Level()), floatValues, pt.Scale.Float64(), pt.Value.Coeffs)

	// Zero out remaining coefficients
	for i := len(values); i < ecd.n; i++ {
		for j := range pt.Value.Coeffs {
			pt.Value.Coeffs[j][i] = 0
		}
	}

	// Mark the plaintext as being in coefficient domain (not NTT domain)
	pt.IsNTT = false

	return nil
}

// EncodeFloat64 encodes a float64 vector into a plaintext (like original CKKS)
func (ecd *Encoder) EncodeFloat64(values []float64, pt *rlwe.Plaintext) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Use the original CKKS encoding approach
	Float64ToFixedPointCRT(ecd.parameters.RingQ().AtLevel(pt.Level()), values, pt.Scale.Float64(), pt.Value.Coeffs)

	// Zero out remaining coefficients
	for i := len(values); i < ecd.n; i++ {
		for j := range pt.Value.Coeffs {
			pt.Value.Coeffs[j][i] = 0
		}
	}

	// Mark the plaintext as being in coefficient domain (not NTT domain)
	pt.IsNTT = false

	return nil
}

// EncodeComplex encodes a complex128 vector into a plaintext (like original CKKS)
func (ecd *Encoder) EncodeComplex(values []complex128, pt *rlwe.Plaintext) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Convert complex128 to float64 (real parts only)
	floatValues := make([]float64, len(values))
	for i, val := range values {
		floatValues[i] = real(val)
	}

	// Use the original CKKS encoding approach
	Float64ToFixedPointCRT(ecd.parameters.RingQ().AtLevel(pt.Level()), floatValues, pt.Scale.Float64(), pt.Value.Coeffs)

	// Zero out remaining coefficients
	for i := len(values); i < ecd.n; i++ {
		for j := range pt.Value.Coeffs {
			pt.Value.Coeffs[j][i] = 0
		}
	}

	// Mark the plaintext as being in coefficient domain (not NTT domain)
	pt.IsNTT = false

	return nil
}

// Decode decodes a plaintext to values, supporting both polynomial coefficients and complex vectors
func (ecd *Encoder) Decode(pt *rlwe.Plaintext, values interface{}) (err error) {
	switch values := values.(type) {
	case *[]uint64:
		// Polynomial coefficient decoding (original Matrix CKKS approach)
		return ecd.DecodePolynomial(pt, *values)
	case *[]float64:
		// Float64 vector decoding (like original CKKS)
		return ecd.DecodeFloat64(pt, *values)
	case *[]complex128:
		// Complex vector decoding (like original CKKS)
		return ecd.DecodeComplex(pt, *values)
	default:
		return fmt.Errorf("cannot Decode: supported types are *[]uint64, *[]float64, or *[]complex128, but %T was given", values)
	}
}

// DecodePolynomial extracts polynomial coefficients as uint64 values.
// This function follows the original CKKS approach exactly.
func (ecd *Encoder) DecodePolynomial(pt *rlwe.Plaintext, values []uint64) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Convert to float64 for decoding (same as original CKKS)
	floatValues := make([]float64, len(values))

	// Handle NTT domain conversion if needed
	if pt.IsNTT {
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		ringQ.INTT(pt.Value, ecd.buff)
		err = ecd.polyToFloatNoCRT(ecd.buff.Coeffs[0], floatValues, pt.Scale, 0, ringQ)
	} else {
		err = ecd.polyToFloatNoCRT(pt.Value.Coeffs[0], floatValues, pt.Scale, 0, ecd.parameters.RingQ().AtLevel(pt.Level()))
	}

	if err != nil {
		return err
	}

	// Convert back to uint64
	for i, val := range floatValues {
		values[i] = uint64(val)
	}

	return nil
}

// DecodeFloat64 extracts float64 values (like original CKKS)
func (ecd *Encoder) DecodeFloat64(pt *rlwe.Plaintext, values []float64) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Handle NTT domain conversion if needed
	if pt.IsNTT {
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		ringQ.INTT(pt.Value, ecd.buff)
		return ecd.polyToFloatNoCRT(ecd.buff.Coeffs[0], values, pt.Scale, 0, ringQ)
	} else {
		return ecd.polyToFloatNoCRT(pt.Value.Coeffs[0], values, pt.Scale, 0, ecd.parameters.RingQ().AtLevel(pt.Level()))
	}
}

// DecodeComplex extracts complex128 values (like original CKKS)
func (ecd *Encoder) DecodeComplex(pt *rlwe.Plaintext, values []complex128) (err error) {
	if len(values) > ecd.n {
		return fmt.Errorf("too many values: %d > %d", len(values), ecd.n)
	}

	// Convert to float64 for decoding (real parts only)
	floatValues := make([]float64, len(values))

	// Handle NTT domain conversion if needed
	if pt.IsNTT {
		ringQ := ecd.parameters.RingQ().AtLevel(pt.Level())
		ringQ.INTT(pt.Value, ecd.buff)
		err = ecd.polyToFloatNoCRT(ecd.buff.Coeffs[0], floatValues, pt.Scale, 0, ringQ)
	} else {
		err = ecd.polyToFloatNoCRT(pt.Value.Coeffs[0], floatValues, pt.Scale, 0, ecd.parameters.RingQ().AtLevel(pt.Level()))
	}

	if err != nil {
		return err
	}

	// Convert back to complex128
	for i, val := range floatValues {
		values[i] = complex(val, 0) // Real parts only
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

// Float64ToFixedPointCRT encodes a slice of float64 values into CRT polynomial coefficients.
// This is adapted from the original CKKS implementation.
func Float64ToFixedPointCRT(r *ring.Ring, values []float64, scale float64, coeffs [][]uint64) {
	start := len(values)
	end := len(coeffs[0])

	for i := 0; i < start; i++ {
		SingleFloat64ToFixedPointCRT(r, i, values[i], scale, coeffs)
	}

	for i := start; i < end; i++ {
		SingleFloat64ToFixedPointCRT(r, i, 0, 0, coeffs)
	}
}

// SingleFloat64ToFixedPointCRT encodes a single float64 value into CRT polynomial coefficients.
// This is adapted from the original CKKS implementation.
func SingleFloat64ToFixedPointCRT(r *ring.Ring, i int, value float64, scale float64, coeffs [][]uint64) {
	if value == 0 {
		for j := range coeffs {
			coeffs[j][i] = 0
		}
		return
	}

	var isNegative bool
	var xFlo *big.Float
	var xInt *big.Int
	var c uint64

	isNegative = false

	if value < 0 {
		isNegative = true
		scale *= -1
	}

	value *= scale

	moduli := r.ModuliChain()[:r.Level()+1]

	if value >= 1.8446744073709552e+19 {
		xFlo = big.NewFloat(value)
		xInt = new(big.Int)
		xFlo.Int(xInt)
	} else {
		c = uint64(value)
		if isNegative {
			c = -c
		}
	}

	for j, qi := range moduli {
		if xFlo != nil {
			coeffs[j][i] = ring.CRed(xInt.Uint64(), qi)
		} else {
			coeffs[j][i] = ring.CRed(c, qi)
		}
	}
}

// polyToFloatNoCRT decodes a single-level CRT poly on a real valued FloatSlice.
// This is adapted from the original CKKS implementation.
func (ecd *Encoder) polyToFloatNoCRT(coeffs []uint64, values []float64, scale rlwe.Scale, logSlots int, r *ring.Ring) (err error) {
	Q := r.SubRings[0].Modulus
	slots := len(values)

	sf64 := scale.Float64()

	for i := 0; i < slots; i++ {
		if coeffs[i] >= Q>>1 {
			values[i] = -float64(Q-coeffs[i]) / sf64
		} else {
			values[i] = float64(coeffs[i]) / sf64
		}
	}

	return nil
}
