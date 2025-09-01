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
func NewPlaintext(params Parameters, level int) (pt *rlwe.Plaintext) {
	pt = rlwe.NewPlaintext(params, level)
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
func NewCiphertext(params Parameters, degree, level int) (ct *rlwe.Ciphertext) {
	ct = rlwe.NewCiphertext(params, degree, level)
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
func NewEncryptor(params Parameters, key rlwe.EncryptionKey) *rlwe.Encryptor {
	return rlwe.NewEncryptor(params, key)
}

// NewDecryptor instantiates a new [rlwe.Decryptor] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//   - key: *[rlwe.SecretKey]
//
// output: an [rlwe.Decryptor] instantiated with the provided secret-key.
func NewDecryptor(params Parameters, key *rlwe.SecretKey) *rlwe.Decryptor {
	return rlwe.NewDecryptor(params, key)
}

// NewKeyGenerator instantiates a new [rlwe.KeyGenerator] for matrix CKKS.
//
// inputs:
//   - params: an [rlwe.ParameterProvider] interface
//
// output: an [rlwe.KeyGenerator].
func NewKeyGenerator(params Parameters) *rlwe.KeyGenerator {
	return rlwe.NewKeyGenerator(params)
}
