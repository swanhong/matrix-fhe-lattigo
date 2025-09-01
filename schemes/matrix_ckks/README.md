# Matrix CKKS Scheme

Matrix CKKS is a variant of the CKKS homomorphic encryption scheme that uses 3N-friendly ring structures instead of power-of-2 rings. This enables better support for matrix operations.

## Key Differences from Standard CKKS

1. **Ring Structure**: Uses rings of degree N = 2^a * 3^b instead of N = 2^k
2. **NTT Support**: Designed to work with 3N-NTT (once integrated)
3. **Matrix Operations**: Better suited for certain matrix computations

## Current Status

✅ **Completed**:
- Basic parameter structure for 3N-friendly rings
- Parameter validation (N must be of form 2^a * 3^b)
- Basic encoder for polynomial coefficients
- Encryption/decryption framework using RLWE backend
- Example parameters and test cases

🚧 **In Progress**:
- Integration with 3N-NTT implementation
- Proper coefficient handling for 3N rings

📋 **Planned**:
- Matrix-optimized evaluator
- Specialized encodings for matrix operations
- Performance optimizations

## Usage Example

```go
```go
// Create parameters from literal
params, err := matrix_ckks.NewParametersFromLiteral(paramLiteral)

// Generate keys
kgen := matrix_ckks.NewKeyGenerator(params)
sk, pk := kgen.GenKeyPairNew()

// Create encoder and crypto objects
encoder := matrix_ckks.NewEncoder(params)
encryptor := matrix_ckks.NewEncryptor(params, pk)
decryptor := matrix_ckks.NewDecryptor(params, sk)

// Encrypt polynomial coefficients
inputPoly := []uint64{1, 2, 3, 4, 5, 6}
pt := matrix_ckks.NewPlaintext(params, params.MaxLevel())
encoder.EncodePolynomial(inputPoly, pt)

ct := matrix_ckks.NewCiphertext(params, 1, params.MaxLevel())
encryptor.Encrypt(pt, ct)

// Decrypt
ptDecrypted := matrix_ckks.NewPlaintext(params, ct.Level())
decryptor.Decrypt(ct, ptDecrypted)

outputPoly := make([]uint64, len(inputPoly))
encoder.DecodePolynomial(ptDecrypted, outputPoly)
```

## Testing

Run the tests with:

```bash
cd schemes/matrix_ckks
go test -v .
```

## Supported Ring Degrees

The scheme supports ring degrees N of the form 2^a * 3^b, such as:
- N = 6 (2^1 * 3^1)
- N = 12 (2^2 * 3^1) 
- N = 16 (2^4 * 3^0)
- N = 18 (2^1 * 3^2)
- N = 24 (2^3 * 3^1)
- N = 36 (2^2 * 3^2)
- N = 48 (2^4 * 3^1)
- N = 96 (2^5 * 3^1)

Note: Current implementation requires N ≥ 16 due to underlying RLWE constraints.
