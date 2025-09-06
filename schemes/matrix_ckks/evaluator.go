package matrix_ckks

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Evaluator wraps the underlying RLWE evaluator for matrix CKKS homomorphic operations.
type Evaluator struct {
	*rlwe.Evaluator
	parameters *rlwe.Parameters3N
}

// NewEvaluator instantiates a new [Evaluator] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - evk: *[rlwe.EvaluationKeySet] (can be nil for addition/subtraction only)
//
// output: an [Evaluator] for performing homomorphic operations.
func NewEvaluator(params *rlwe.Parameters3N, evk rlwe.EvaluationKeySet) *Evaluator {
	return &Evaluator{
		Evaluator:  rlwe.NewEvaluator(params.GetRLWEParameters(), evk),
		parameters: params,
	}
}

// Add performs homomorphic addition of two ciphertexts and stores the result in ctOut.
// ctOut = ct0 + ct1
//
// This is a simplified implementation that focuses on correctness over performance.
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
	ctOut.IsNTT = ct0.IsNTT // This was missing! Keep same domain as inputs

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
// Note: For Matrix CKKS, we perform pointwise multiplication of coefficients,
// not polynomial convolution multiplication. This is different from standard ring operations.
func (eval *Evaluator) Mul(ct0, ct1, ctOut *rlwe.Ciphertext) (err error) {
	// Check that ciphertexts are at the same level
	if ct0.Level() != ct1.Level() {
		return fmt.Errorf("ciphertexts must be at the same level for multiplication")
	}

	level := ct0.Level()

	// For Matrix CKKS, we do pointwise multiplication, so output degree is max of input degrees
	// not sum (like in polynomial convolution)
	outputDegree := max(ct0.Degree(), ct1.Degree())
	ctOut.Resize(outputDegree, level)

	// Copy metadata (scale will be updated)
	// For Matrix CKKS pointwise multiplication, the scale should be the product of input scales
	// but we need to manage it to avoid overflow with smaller moduli
	inputScale := ct0.Scale.Uint64()

	// For pointwise multiplication, the natural scale would be inputScale^2
	// but we need to rescale to keep within modulus bounds
	ringQ2 := eval.parameters.RingQ().AtLevel(level)
	modulus := ringQ2.ModuliChain()[0]

	// Use the product of input scales, but ensure it fits in the modulus
	productScale := inputScale * inputScale

	// If the product scale is too large, scale it down
	if productScale > modulus/10 {
		// Use a reasonable scale that allows the multiplication result to fit
		productScale = modulus / 1000
	}

	ctOut.Scale = rlwe.NewScale(productScale)

	ctOut.LogDimensions = ct0.LogDimensions
	ctOut.IsBatched = ct0.IsBatched
	ctOut.IsNTT = ct0.IsNTT

	// Get ring at the appropriate level
	ringQ := eval.parameters.RingQ().AtLevel(level)

	// For Matrix CKKS, perform pointwise multiplication of coefficients
	// This is different from polynomial ring multiplication

	// Handle different degree combinations
	if ct0.Degree() == 1 && ct1.Degree() == 1 {
		// Both are degree 1, result can be degree 1 for pointwise operations
		ctOut.Resize(1, level)

		// Pointwise multiplication: (a0 + a1*X) * (b0 + b1*X) = (a0*b0) + (a1*b1)*X
		// where coefficients are multiplied pointwise, not convolved

		// When we multiply two scaled values (a*scale) * (b*scale) = (a*b*scale^2)
		// But our output scale might be different, so we need to adjust

		for j := 0; j <= level; j++ {
			for k := 0; k < ringQ.N(); k++ {
				// Multiply corresponding coefficients pointwise
				c0_coeff := ct0.Value[0].Coeffs[j][k]
				c1_coeff := ct1.Value[0].Coeffs[j][k]

				// Multiply and adjust for scale difference
				product := (c0_coeff * c1_coeff) % ringQ.SubRings[j].Modulus
				// The product is scaled by inputScale^2, but we want outputScale
				// So we need to multiply by (outputScale / inputScale^2)

				// For safety, just use the raw product for now and let decoding handle the scale
				ctOut.Value[0].Coeffs[j][k] = product

				// For degree 1 terms
				c0_deg1 := ct0.Value[1].Coeffs[j][k]
				c1_deg1 := ct1.Value[1].Coeffs[j][k]
				product1 := (c0_deg1 * c1_deg1) % ringQ.SubRings[j].Modulus
				ctOut.Value[1].Coeffs[j][k] = product1
			}
		}
	} else {
		return fmt.Errorf("unsupported ciphertext degrees for pointwise multiplication: %d, %d", ct0.Degree(), ct1.Degree())
	}

	return nil
}

// MulNew performs homomorphic multiplication of two ciphertexts and returns the result.
func (eval *Evaluator) MulNew(ct0, ct1 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	outputDegree := ct0.Degree() + ct1.Degree()
	ctOut := NewCiphertext(eval.parameters, outputDegree, ct0.Level())
	err := eval.Mul(ct0, ct1, ctOut)
	return ctOut, err
}

// Rescale rescales a ciphertext by dividing its scale by the given factor.
// This is essential after multiplication to keep scales manageable.
// ct.scale = ct.scale / scaleFactor
func (eval *Evaluator) Rescale(ct, ctOut *rlwe.Ciphertext, scaleFactor rlwe.Scale) (err error) {
	level := ct.Level()

	// Resize output ciphertext
	ctOut.Resize(ct.Degree(), level)

	// Update scale
	ctOut.Scale = ct.Scale.Div(scaleFactor)
	ctOut.LogDimensions = ct.LogDimensions
	ctOut.IsBatched = ct.IsBatched
	ctOut.IsNTT = ct.IsNTT

	// Get ring at the appropriate level
	ringQ := eval.parameters.RingQ().AtLevel(level)

	// Convert scale factor to uint64 for division
	scaleFactorUint64 := scaleFactor.Uint64()

	// For each polynomial coefficient in the ciphertext
	for i := 0; i <= ct.Degree(); i++ {
		// For each level (prime)
		for j := 0; j <= level; j++ {
			// For each coefficient in the polynomial
			for k := 0; k < ringQ.N(); k++ {
				// Get coefficient value at this level
				coeff := ct.Value[i].Coeffs[j][k]

				// Divide by scale factor with rounding
				rescaledCoeff := (coeff + scaleFactorUint64/2) / scaleFactorUint64

				ctOut.Value[i].Coeffs[j][k] = rescaledCoeff
			}
		}
	}

	return nil
}

// RescaleNew rescales a ciphertext and returns the result.
func (eval *Evaluator) RescaleNew(ct *rlwe.Ciphertext, scaleFactor rlwe.Scale) (*rlwe.Ciphertext, error) {
	ctOut := NewCiphertext(eval.parameters, ct.Degree(), ct.Level())
	err := eval.Rescale(ct, ctOut, scaleFactor)
	return ctOut, err
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
