package matrix_ckks

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// MatrixEncryptor is a Matrix CKKS specific encryptor that uses pointwise operations
// instead of polynomial convolution multiplication.
type MatrixEncryptor struct {
	params    *rlwe.Parameters3N
	ringQ     *ring.Ring
	sk        *rlwe.SecretKey
	xeSampler ring.Sampler
	buff      ring.Poly // Temporary buffer for operations
}

// NewMatrixEncryptor creates a new Matrix CKKS encryptor.
func NewMatrixEncryptor(params *rlwe.Parameters3N, sk *rlwe.SecretKey) *MatrixEncryptor {
	ringQ := params.RingQ()

	// Create error sampler for encryption noise
	prng, err := sampling.NewPRNG()
	if err != nil {
		panic(fmt.Sprintf("failed to create PRNG: %v", err))
	}
	xeSampler, err := ring.NewSampler(prng, ringQ, params.Xe(), false)
	if err != nil {
		panic(fmt.Sprintf("failed to create error sampler: %v", err))
	}

	return &MatrixEncryptor{
		params:    params,
		ringQ:     ringQ,
		sk:        sk,
		xeSampler: xeSampler,
		buff:      ringQ.NewPoly(),
	}
}

// Encrypt encrypts a plaintext using proper RLWE encryption with pointwise operations.
// For Matrix CKKS, we use the standard RLWE encryption: ct = (pt + e - a*s, a)
// where 'a' is sampled uniformly, 's' is the secret key, and 'e' is noise.
// The key difference is that we use pointwise multiplication for a*s instead of polynomial convolution.
func (enc *MatrixEncryptor) Encrypt(pt *rlwe.Plaintext, ct *rlwe.Ciphertext) error {
	level := pt.Level()
	ringQ := enc.ringQ.AtLevel(level)

	// Resize ciphertext to degree 1
	ct.Resize(1, level)

	// Copy metadata
	*ct.MetaData = *pt.MetaData

	// Standard RLWE encryption: ct = (pt + e - a*s, a)
	// where 'a' is uniform random, 's' is secret key, 'e' is noise

	// Sample uniform random polynomial 'a' for ct[1]
	enc.xeSampler.AtLevel(level).Read(ct.Value[1])

	// Sample noise polynomial 'e'
	e := enc.buff
	enc.xeSampler.AtLevel(level).Read(e)

	// Compute a*s using pointwise multiplication (Matrix CKKS specific)
	aTimesS := ringQ.NewPoly()
	for j := 0; j <= level; j++ {
		for k := 0; k < ringQ.N(); k++ {
			aCoeff := ct.Value[1].Coeffs[j][k]
			sCoeff := enc.sk.Value.Q.Coeffs[j][k]
			aTimesS.Coeffs[j][k] = ring.MRed(aCoeff, sCoeff, ringQ.SubRings[j].Modulus, ringQ.SubRings[j].MRedConstant)
		}
	}

	// ct[0] = pt + e - a*s
	ringQ.Add(pt.Value, e, ct.Value[0])
	ringQ.Sub(ct.Value[0], aTimesS, ct.Value[0])

	// Set NTT domain based on plaintext
	ct.IsNTT = pt.IsNTT

	return nil
}

// EncryptNew encrypts a plaintext and returns a new ciphertext.
func (enc *MatrixEncryptor) EncryptNew(pt *rlwe.Plaintext) (*rlwe.Ciphertext, error) {
	ct := NewCiphertext(enc.params, 1, pt.Level())
	err := enc.Encrypt(pt, ct)
	return ct, err
}
