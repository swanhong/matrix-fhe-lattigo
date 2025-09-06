package matrix_ckks

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils"
)

// MatrixDecryptor is a Matrix CKKS specific decryptor that uses pointwise operations
// instead of polynomial convolution multiplication.
type MatrixDecryptor struct {
	params *rlwe.Parameters3N
	ringQ  *ring.Ring
	sk     *rlwe.SecretKey
	buff   ring.Poly
}

// NewMatrixDecryptor creates a new Matrix CKKS decryptor.
func NewMatrixDecryptor(params *rlwe.Parameters3N, sk *rlwe.SecretKey) *MatrixDecryptor {
	ringQ := params.RingQ()

	return &MatrixDecryptor{
		params: params,
		ringQ:  ringQ,
		sk:     sk,
		buff:   ringQ.NewPoly(),
	}
}

// Decrypt decrypts a ciphertext using Matrix CKKS pointwise operations.
func (dec *MatrixDecryptor) Decrypt(ct *rlwe.Ciphertext, pt *rlwe.Plaintext) {
	level := utils.Min(ct.Level(), pt.Level())
	ringQ := dec.ringQ.AtLevel(level)

	// Resize plaintext
	pt.Resize(0, level)

	// Copy metadata
	*pt.MetaData = *ct.MetaData

	// For Matrix CKKS, decryption follows standard RLWE:
	// pt = ct[0] + sk * ct[1] (pointwise operations)

	// Start with ct[0]
	pt.Value.CopyLvl(level, ct.Value[0])

	// For degree 1 ciphertexts, add sk * ct[1]
	if ct.Degree() >= 1 {
		// For Matrix CKKS, we use pointwise multiplication instead of polynomial multiplication
		// pt = pt + sk * ct[1] (pointwise)

		// Create temporary buffer for sk * ct[1]
		skTimesCt1 := dec.buff

		// Pointwise multiplication: sk[i] * ct[1][i] for each coefficient i
		for j := 0; j <= level; j++ {
			for k := 0; k < ringQ.N(); k++ {
				skCoeff := dec.sk.Value.Q.Coeffs[j][k]
				ct1Coeff := ct.Value[1].Coeffs[j][k]
				// Use proper modular multiplication
				skTimesCt1.Coeffs[j][k] = ring.MRed(skCoeff, ct1Coeff, ringQ.SubRings[j].Modulus, ringQ.SubRings[j].MRedConstant)
			}
		}

		// Add: pt = pt + sk * ct[1]
		ringQ.Add(pt.Value, skTimesCt1, pt.Value)
	}

	// Reduce if needed
	ringQ.Reduce(pt.Value, pt.Value)

	// Set NTT domain based on ciphertext
	pt.IsNTT = ct.IsNTT
}

// DecryptNew decrypts a ciphertext and returns a new plaintext.
func (dec *MatrixDecryptor) DecryptNew(ct *rlwe.Ciphertext) *rlwe.Plaintext {
	pt := NewPlaintext(dec.params, ct.Level())
	dec.Decrypt(ct, pt)
	return pt
}
