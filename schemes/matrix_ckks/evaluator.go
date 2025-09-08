package matrix_ckks

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Evaluator is a Matrix CKKS-specific evaluator that provides homomorphic operations
// optimized for 3N-ring structures. It uses the Matrix CKKS encoder and implements
// proper RNS rescaling operations.
type Evaluator struct {
	*Encoder
	parameters *rlwe.Parameters3N
	evk        rlwe.EvaluationKeySet
	buffQ      [3]ring.Poly // Memory buffers for intermediate computations
}

// NewEvaluator instantiates a new [Evaluator] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.Parameters3N] interface
//   - evk: *[rlwe.EvaluationKeySet] (can be nil for addition/subtraction only)
//
// output: an [Evaluator] for performing homomorphic operations.
func NewEvaluator(params *rlwe.Parameters3N, evk rlwe.EvaluationKeySet) *Evaluator {
	encoder := NewEncoder(params)

	// Initialize memory buffers
	ringQ := params.RingQ()
	buffQ := [3]ring.Poly{
		ringQ.NewPoly(),
		ringQ.NewPoly(),
		ringQ.NewPoly(),
	}

	return &Evaluator{
		Encoder:    encoder,
		parameters: params,
		evk:        evk,
		buffQ:      buffQ,
	}
}

// GetParameters returns the Matrix CKKS parameters.
func (eval *Evaluator) GetParameters() *rlwe.Parameters3N {
	return eval.parameters
}

// GetRLWEParameters returns the underlying RLWE parameters.
func (eval *Evaluator) GetRLWEParameters() *rlwe.Parameters {
	return eval.parameters.GetRLWEParameters()
}

// Add performs homomorphic addition of two ciphertexts and stores the result in ctOut.
// ctOut = ct0 + ct1
func (eval *Evaluator) Add(ct0, ct1, ctOut *rlwe.Ciphertext) (err error) {
	// Check that ciphertexts are at the same level
	if ct0.Level() != ct1.Level() {
		return fmt.Errorf("ciphertexts must be at the same level for addition")
	}

	level := ct0.Level()

	// Resize output to handle both input degrees
	maxDegree := max(ct0.Degree(), ct1.Degree())
	ctOut.Resize(maxDegree, level)

	// Copy metadata from first operand
	ctOut.Scale = ct0.Scale
	ctOut.LogDimensions = ct0.LogDimensions
	ctOut.IsBatched = ct0.IsBatched
	ctOut.IsNTT = ct0.IsNTT // Keep same domain as inputs

	// Get ring at the appropriate level
	ringQ := eval.parameters.RingQ().AtLevel(level)

	// Add coefficient polynomials element-wise
	minDegree := min(ct0.Degree(), ct1.Degree())

	// Add overlapping coefficients
	for i := 0; i <= minDegree; i++ {
		ringQ.Add(ct0.Value[i], ct1.Value[i], ctOut.Value[i])
	}

	// Copy non-overlapping coefficients from the longer ciphertext
	if ct0.Degree() > ct1.Degree() {
		for i := minDegree + 1; i <= ct0.Degree(); i++ {
			ctOut.Value[i].CopyLvl(level, ct0.Value[i])
		}
	} else if ct1.Degree() > ct0.Degree() {
		for i := minDegree + 1; i <= ct1.Degree(); i++ {
			ctOut.Value[i].CopyLvl(level, ct1.Value[i])
		}
	}

	return nil
}

// AddNew performs homomorphic addition of two ciphertexts and returns the result.
func (eval *Evaluator) AddNew(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	ctOut := NewCiphertext(eval.parameters, max(ct0.Degree(), ct1.Degree()), ct0.Level())
	err := eval.Add(ct0, ct1, ctOut)
	return ctOut, err
}

// Mul performs homomorphic multiplication of two ciphertexts and stores the result in ctOut.
// ctOut = ct0 * ct1
//
// This implements proper polynomial multiplication in the 3N-ring structure.
func (eval *Evaluator) Mul(ct0, ct1, ctOut *rlwe.Ciphertext) (err error) {
	// Check that ciphertexts are at the same level
	if ct0.Level() != ct1.Level() {
		return fmt.Errorf("ciphertexts must be at the same level for multiplication")
	}

	level := ct0.Level()

	// For polynomial multiplication, output degree is sum of input degrees
	outputDegree := ct0.Degree() + ct1.Degree()
	ctOut.Resize(outputDegree, level)

	// Copy metadata and update scale
	ctOut.Scale = ct0.Scale.Mul(ct1.Scale)
	ctOut.LogDimensions = ct0.LogDimensions
	ctOut.IsBatched = ct0.IsBatched
	ctOut.IsNTT = ct0.IsNTT

	// Get ring at the appropriate level
	ringQ := eval.parameters.RingQ().AtLevel(level)

	// Convert to NTT domain if needed
	if !ct0.IsNTT {
		ringQ.NTT(ct0.Value[0], ct0.Value[0])
		if ct0.Degree() == 1 {
			ringQ.NTT(ct0.Value[1], ct0.Value[1])
		}
		ct0.IsNTT = true
	}
	if !ct1.IsNTT {
		ringQ.NTT(ct1.Value[0], ct1.Value[0])
		if ct1.Degree() == 1 {
			ringQ.NTT(ct1.Value[1], ct1.Value[1])
		}
		ct1.IsNTT = true
	}

	// Perform polynomial multiplication in NTT domain
	if ct0.Degree() == 0 && ct1.Degree() == 0 {
		// Both are degree 0 (plaintext-like)
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[0], ctOut.Value[0])
	} else if ct0.Degree() == 0 && ct1.Degree() == 1 {
		// ct0 is degree 0, ct1 is degree 1
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[0], ctOut.Value[0])
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[1], ctOut.Value[1])
	} else if ct0.Degree() == 1 && ct1.Degree() == 0 {
		// ct0 is degree 1, ct1 is degree 0
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[0], ctOut.Value[0])
		ringQ.MulCoeffsMontgomery(ct0.Value[1], ct1.Value[0], ctOut.Value[1])
	} else if ct0.Degree() == 1 && ct1.Degree() == 1 {
		// Both are degree 1
		// ctOut[0] = ct0[0] * ct1[0]
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[0], ctOut.Value[0])

		// ctOut[1] = ct0[0] * ct1[1] + ct0[1] * ct1[0]
		ringQ.MulCoeffsMontgomery(ct0.Value[0], ct1.Value[1], ctOut.Value[1])
		ringQ.MulCoeffsMontgomeryThenAdd(ct0.Value[1], ct1.Value[0], ctOut.Value[1])

		// ctOut[2] = ct0[1] * ct1[1]
		ringQ.MulCoeffsMontgomery(ct0.Value[1], ct1.Value[1], ctOut.Value[2])
	} else {
		return fmt.Errorf("unsupported ciphertext degrees for multiplication: %d, %d", ct0.Degree(), ct1.Degree())
	}

	// Mark output as being in NTT domain
	ctOut.IsNTT = true

	// Convert back to coefficient domain for proper decryption
	ringQ.INTT(ctOut.Value[0], ctOut.Value[0])
	if ctOut.Degree() >= 1 {
		ringQ.INTT(ctOut.Value[1], ctOut.Value[1])
	}
	if ctOut.Degree() >= 2 {
		ringQ.INTT(ctOut.Value[2], ctOut.Value[2])
	}
	ctOut.IsNTT = false

	return nil
}

// MulNew performs homomorphic multiplication of two ciphertexts and returns the result.
func (eval *Evaluator) MulNew(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	outputDegree := ct0.Degree() + ct1.Degree()
	ctOut := NewCiphertext(eval.parameters, outputDegree, ct0.Level())
	err := eval.Mul(ct0, ct1, ctOut)
	return ctOut, err
}

// Rescale divides op0 by the last prime of the moduli chain and repeats this procedure
// levelsConsumedPerRescaling times. This follows the original CKKS implementation exactly.
//
// Returns an error if:
//   - Either op0 or opOut MetaData are nil
//   - The level of op0 is too low to enable a rescale
func (eval *Evaluator) Rescale(op0, opOut *rlwe.Ciphertext) (err error) {
	if op0.MetaData == nil || opOut.MetaData == nil {
		return fmt.Errorf("cannot Rescale: op0.MetaData or opOut.MetaData is nil")
	}

	params := eval.parameters
	nbRescales := eval.levelsConsumedPerRescaling()

	if op0.Level() <= nbRescales-1 {
		return fmt.Errorf("cannot Rescale: input Ciphertext level is too low")
	}

	if op0 != opOut {
		opOut.Resize(op0.Degree(), op0.Level()-nbRescales)
	}

	*opOut.MetaData = *op0.MetaData

	ringQ := params.RingQ().AtLevel(op0.Level())

	// Scale division by the moduli being removed (same as original CKKS)
	for i := 0; i < nbRescales; i++ {
		opOut.Scale = opOut.Scale.Div(rlwe.NewScale(ringQ.SubRings[op0.Level()-i].Modulus))
	}

	// Polynomial division using RNS (same as original CKKS)
	for i := range opOut.Value {
		ringQ.DivRoundByLastModulusManyNTT(nbRescales, op0.Value[i], eval.buffQ[0], opOut.Value[i])
	}

	if op0 == opOut {
		opOut.Resize(op0.Degree(), op0.Level()-nbRescales)
	}

	return nil
}

// RescaleNew rescales a ciphertext and returns the result.
func (eval *Evaluator) RescaleNew(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	nbRescales := eval.levelsConsumedPerRescaling()
	ctOut := NewCiphertext(eval.parameters, ct.Degree(), ct.Level()-nbRescales)
	err := eval.Rescale(ct, ctOut)
	return ctOut, err
}

// ModDown reduces the level of op0 by levels without rescaling (no scale division).
// This is equivalent to DropLevel but with proper RNS handling.
//
// Returns an error if:
//   - Either op0 or opOut MetaData are nil
//   - The level of op0 is too low to enable the operation
func (eval *Evaluator) ModDown(op0, opOut *rlwe.Ciphertext, levels int) (err error) {
	if op0.MetaData == nil || opOut.MetaData == nil {
		return fmt.Errorf("cannot ModDown: op0.MetaData or opOut.MetaData is nil")
	}

	if op0.Level() <= levels-1 {
		return fmt.Errorf("cannot ModDown: input Ciphertext level is too low")
	}

	if op0 != opOut {
		opOut.Resize(op0.Degree(), op0.Level()-levels)
	}

	*opOut.MetaData = *op0.MetaData

	// Copy the scale without modification (no division)
	opOut.Scale = op0.Scale

	// Simply copy the polynomial values (no division by moduli)
	for i := range opOut.Value {
		// Copy only the levels we want to keep
		for j := 0; j <= opOut.Level(); j++ {
			copy(opOut.Value[i].Coeffs[j], op0.Value[i].Coeffs[j])
		}
	}

	if op0 == opOut {
		opOut.Resize(op0.Degree(), op0.Level()-levels)
	}

	return nil
}

// ModDownNew reduces the level of op0 by levels without rescaling and returns the result.
func (eval *Evaluator) ModDownNew(ct *rlwe.Ciphertext, levels int) (*rlwe.Ciphertext, error) {
	ctOut := NewCiphertext(eval.parameters, ct.Degree(), ct.Level()-levels)
	err := eval.ModDown(ct, ctOut, levels)
	return ctOut, err
}

// DropLevel reduces the level of op0 by levels and returns the result in op0.
// No rescaling is applied during this procedure (equivalent to ModDown but in-place).
func (eval *Evaluator) DropLevel(op0 *rlwe.Ciphertext, levels int) {
	op0.Resize(op0.Degree(), op0.Level()-levels)
}

// DropLevelNew reduces the level of op0 by levels and returns the result in a newly created element.
// No rescaling is applied during this procedure.
func (eval *Evaluator) DropLevelNew(op0 *rlwe.Ciphertext, levels int) *rlwe.Ciphertext {
	opOut := op0.CopyNew()
	eval.DropLevel(opOut, levels)
	return opOut
}

// levelsConsumedPerRescaling returns the number of levels consumed per rescaling operation.
// For Matrix CKKS, we use 1 level per rescaling by default.
func (eval *Evaluator) levelsConsumedPerRescaling() int {
	// For Matrix CKKS, we typically consume 1 level per rescaling
	// This can be adjusted based on the specific parameter set
	return 1
}

// MulByConst multiplies a ciphertext by a constant (integer or float)
func (eval *Evaluator) MulByConst(ct *rlwe.Ciphertext, constant interface{}, ctOut *rlwe.Ciphertext) (err error) {
	// Check that ciphertexts are at the same level
	if ct.Level() != ctOut.Level() {
		return fmt.Errorf("ciphertexts must be at the same level for constant multiplication")
	}

	// Use the exact same approach as original CKKS
	// Get the level (minimum of input and output levels, same as original CKKS)
	level := ct.Level() // In our case, both are at the same level

	// Get the ring at the target level (same as original CKKS)
	ringQ := eval.parameters.RingQ().AtLevel(level)

	// Convert the constant to *bignum.Complex (following original CKKS approach)
	cmplxBig := bignum.ToComplex(constant, eval.parameters.EncodingPrecision())

	var scale rlwe.Scale
	if cmplxBig.IsInt() {
		// For integer constants, no scaling required
		scale = rlwe.NewScale(1)
	} else {
		// For non-integer constants, use current modulus as scaling factor
		scale = rlwe.NewScale(ringQ.SubRings[level].Modulus)

		// If multiple moduli are used per rescaling, multiply by additional moduli
		for i := 1; i < eval.levelsConsumedPerRescaling(); i++ {
			scale = scale.Mul(rlwe.NewScale(ringQ.SubRings[level-i].Modulus))
		}
	}

	// Convert the *bignum.Complex to RNS scalar representation
	RNSReal, RNSImag := eval.bigComplexToRNSScalar(ringQ, &scale.Value, cmplxBig)

	// RNS scalars are correctly computed

	// Process the RNS scalars the same way as original CKKS
	// Use the full ring's subrings, not the level-specific ring
	fullRingQ := eval.parameters.RingQ()
	for i, s := range fullRingQ.SubRings[:level+1] {
		RNSImag[i] = ring.MRed(RNSImag[i], s.RootsForward[1], s.Modulus, s.MRedConstant)
		RNSReal[i], RNSImag[i] = ring.CRed(RNSReal[i]+RNSImag[i], s.Modulus), ring.CRed(RNSReal[i]+s.Modulus-RNSImag[i], s.Modulus)
	}

	// Multiply each polynomial by the RNS scalar using the same approach as original CKKS
	for i := 0; i <= ct.Degree(); i++ {
		// Use the ring's double RNS scalar multiplication
		ringQ.MulDoubleRNSScalar(ct.Value[i], RNSReal, RNSImag, ctOut.Value[i])
	}

	// Copy metadata first
	*ctOut.MetaData = *ct.MetaData

	// Update the scale: multiply by the scaling factor used
	ctOut.Scale = ct.Scale.Mul(scale)

	return nil
}

// bigComplexToRNSScalar converts a *bignum.Complex to RNS scalar representation
// This is adapted from the original CKKS implementation
func (eval *Evaluator) bigComplexToRNSScalar(r *ring.Ring, scale *big.Float, cmplx *bignum.Complex) (RNSReal, RNSImag ring.RNSScalar) {
	if scale == nil {
		scale = new(big.Float).SetFloat64(1)
	}

	real := new(big.Int)
	if cmplx[0] != nil {
		r := new(big.Float).Mul(cmplx[0], scale)

		if cmp := cmplx[0].Cmp(new(big.Float)); cmp > 0 {
			r.Add(r, new(big.Float).SetFloat64(0.5))
		} else if cmp < 0 {
			r.Sub(r, new(big.Float).SetFloat64(0.5))
		}

		r.Int(real)
	}

	imag := new(big.Int)
	if cmplx[1] != nil {
		i := new(big.Float).Mul(cmplx[1], scale)

		if cmp := cmplx[1].Cmp(new(big.Float)); cmp > 0 {
			i.Add(i, new(big.Float).SetFloat64(0.5))
		} else if cmp < 0 {
			i.Sub(i, new(big.Float).SetFloat64(0.5))
		}

		i.Int(imag)
	}

	return r.NewRNSScalarFromBigint(real), r.NewRNSScalarFromBigint(imag)
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
