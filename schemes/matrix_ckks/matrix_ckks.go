// Package matrix_ckks implements a matrix-friendly version of the CKKS scheme
// that uses 3N-ring structure instead of power-of-2 rings for better matrix operations.
//
// The main differences from standard CKKS:
// 1. Supports ring degrees of form N = 2^a * 3^b instead of just N = 2^k
// 2. Uses 3N-NTT for polynomial operations
// 3. Optimized for matrix-based computations
//
// The implementation provides all standard CKKS operations (encryption, decryption,
// homomorphic addition, multiplication) while leveraging the 3N-ring structure.
package matrix_ckks

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// NewPlaintext allocates a new [rlwe.Plaintext] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - level: the level of the plaintext
//
// output: a newly allocated [rlwe.Plaintext] at the specified level.
//
// Note: the user can update the field MetaData to set a specific scaling factor,
// plaintext dimensions (if applicable) or encoding domain, before encoding values
// on the created plaintext.
func NewPlaintext(params *rlwe.Parameters3N, level int) (pt *rlwe.Plaintext) {
	// Create plaintext using the RLWE method first to get proper metadata
	pt = rlwe.NewPlaintext(params.GetRLWEParameters(), level)

	// Replace the polynomial with one from our Matrix ring
	ringQ := params.RingQ().AtLevel(level)
	pt.Value = ringQ.NewPoly()

	// Set Matrix CKKS specific metadata
	pt.IsBatched = true
	pt.Scale = params.DefaultScale()
	pt.LogDimensions = params.LogMaxDimensions()
	return
}

// NewCiphertext allocates a new [rlwe.Ciphertext] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - degree: the degree of the ciphertext
//   - level: the level of the Ciphertext
//
// output: a newly allocated [rlwe.Ciphertext] of the specified degree and level.
func NewCiphertext(params *rlwe.Parameters3N, degree, level int) (ct *rlwe.Ciphertext) {
	// Create ciphertext using RLWE method first to get proper metadata
	ct = rlwe.NewCiphertext(params.GetRLWEParameters(), degree, level)

	// Replace the polynomials with ones from our Matrix ring
	ringQ := params.RingQ().AtLevel(level)
	ct.Value = make([]ring.Poly, degree+1)
	for i := range ct.Value {
		ct.Value[i] = ringQ.NewPoly()
	}

	// Set Matrix CKKS specific metadata
	ct.IsBatched = true
	ct.Scale = params.DefaultScale()
	ct.LogDimensions = params.LogMaxDimensions()
	return
}

// NewEncryptor instantiates a new [rlwe.Encryptor] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - key: *[rlwe.SecretKey] or *[rlwe.PublicKey]
//
// output: an [rlwe.Encryptor] instantiated with the provided key.
func NewEncryptor(params *rlwe.Parameters3N, key rlwe.EncryptionKey) *rlwe.Encryptor {
	return rlwe.NewEncryptor(params, key)
}

// NewDecryptor instantiates a new [rlwe.Decryptor] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - key: *[rlwe.SecretKey]
//
// output: an [rlwe.Decryptor] instantiated with the provided secret-key.
func NewDecryptor(params *rlwe.Parameters3N, key *rlwe.SecretKey) *rlwe.Decryptor {
	return rlwe.NewDecryptor(params, key)
}

// NewKeyGenerator instantiates a new [rlwe.KeyGenerator] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//
// output: an [rlwe.KeyGenerator].
func NewKeyGenerator(params *rlwe.Parameters3N) *rlwe.KeyGenerator {
	return rlwe.NewKeyGenerator(params)
}
