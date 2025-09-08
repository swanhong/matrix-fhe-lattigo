package matrix_ckks

import (
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// TestMatrixCKKSBasicEncryptDecrypt tests the basic encrypt-decrypt pipeline
func TestMatrixCKKSBasicEncryptDecrypt(t *testing.T) {
	// Use Matrix CKKS parameters
	paramsLiteral := rlwe.ParametersLiteral3N{
		Order2:       3,
		Order3:       1,
		LogQ:         []int{55, 45, 45, 45, 45, 45, 45},
		LogP:         []int{60},
		Xe:           ring.Ternary{H: 1},
		Xs:           ring.Ternary{H: 1},
		RingType:     ring.Matrix,
		DefaultScale: rlwe.NewScale(1 << 45),
	}

	params, err := rlwe.NewParametersFromLiteral3N(paramsLiteral)
	if err != nil {
		t.Fatalf("Failed to create Matrix CKKS parameters: %v", err)
	}

	// Test complex vectors (like original CKKS)
	testValues := []complex128{1.0 + 0i, 2.0 + 0i, 3.0 + 0i, 4.0 + 0i}

	// Create Matrix CKKS components
	encoder := NewEncoder(&params)
	kgen := NewKeyGenerator(&params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)

	// Create plaintext and encode (like original CKKS)
	pt := NewPlaintext(&params, 6)
	err = encoder.Encode(testValues, pt)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Decrypt and decode result (like original CKKS)
	ptResult := NewPlaintext(&params, ct.Level())
	decryptor.Decrypt(ct, ptResult)
	resultValues := make([]complex128, len(testValues))
	err = encoder.Decode(ptResult, &resultValues)
	if err != nil {
		t.Fatalf("Result decoding failed: %v", err)
	}

	// Verify results
	tolerance := 0.1

	for i := 0; i < len(testValues); i++ {
		expected := real(testValues[i])
		actual := real(resultValues[i])

		if actual < expected-tolerance || actual > expected+tolerance {
			t.Errorf("Value %d: expected ~%.1f, got %.1f", i, expected, actual)
		}
	}
}

// TestMatrixCKKSAddition tests Matrix CKKS addition operation
func TestMatrixCKKSAddition(t *testing.T) {
	// Use Matrix CKKS parameters
	paramsLiteral := rlwe.ParametersLiteral3N{
		Order2:       3,
		Order3:       1,
		LogQ:         []int{55, 45, 45, 45, 45, 45, 45},
		LogP:         []int{60},
		Xe:           ring.Ternary{H: 1},
		Xs:           ring.Ternary{H: 1},
		RingType:     ring.Matrix,
		DefaultScale: rlwe.NewScale(1 << 45),
	}

	params, err := rlwe.NewParametersFromLiteral3N(paramsLiteral)
	if err != nil {
		t.Fatalf("Failed to create Matrix CKKS parameters: %v", err)
	}

	// Test complex vectors (like original CKKS)
	values1 := []complex128{1.0 + 0i, 2.0 + 0i, 3.0 + 0i, 4.0 + 0i}
	values2 := []complex128{5.0 + 0i, 6.0 + 0i, 7.0 + 0i, 8.0 + 0i}

	// Create Matrix CKKS components
	encoder := NewEncoder(&params)
	kgen := NewKeyGenerator(&params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil)

	// Create plaintexts and encode (like original CKKS)
	pt1 := NewPlaintext(&params, 6)
	err = encoder.Encode(values1, pt1)
	if err != nil {
		t.Fatalf("Encoding values1 failed: %v", err)
	}

	pt2 := NewPlaintext(&params, 6)
	err = encoder.Encode(values2, pt2)
	if err != nil {
		t.Fatalf("Encoding values2 failed: %v", err)
	}

	// Encrypt both plaintexts
	ct1, err := encryptor.EncryptNew(pt1)
	if err != nil {
		t.Fatalf("Encryption of values1 failed: %v", err)
	}

	ct2, err := encryptor.EncryptNew(pt2)
	if err != nil {
		t.Fatalf("Encryption of values2 failed: %v", err)
	}

	// Test addition
	ctSum := NewCiphertext(&params, ct1.Degree(), ct1.Level())
	err = evaluator.Add(ct1, ct2, ctSum)
	if err != nil {
		t.Fatalf("Addition failed: %v", err)
	}

	// Decrypt and decode result (like original CKKS)
	ptResult := NewPlaintext(&params, ctSum.Level())
	decryptor.Decrypt(ctSum, ptResult)
	resultValues := make([]complex128, len(values1))
	err = encoder.Decode(ptResult, &resultValues)
	if err != nil {
		t.Fatalf("Result decoding failed: %v", err)
	}

	// Calculate expected values
	expectedValues := make([]complex128, len(values1))
	for i := 0; i < len(values1); i++ {
		expectedValues[i] = values1[i] + values2[i]
	}

	// Verify results
	tolerance := 0.1

	for i := 0; i < len(values1); i++ {
		expected := real(expectedValues[i])
		actual := real(resultValues[i])

		if actual < expected-tolerance || actual > expected+tolerance {
			t.Errorf("Value %d: expected ~%.1f, got %.1f", i, expected, actual)
		}
	}
}

// TestMatrixCKKSComplexVectorConstantMultiplication tests Matrix CKKS with complex vectors like original CKKS
func TestMatrixCKKSComplexVectorConstantMultiplication(t *testing.T) {
	// Use Matrix CKKS parameters
	paramsLiteral := rlwe.ParametersLiteral3N{
		Order2:       3,
		Order3:       1,
		LogQ:         []int{55, 45, 45, 45, 45, 45, 45},
		LogP:         []int{60},
		Xe:           ring.Ternary{H: 1},
		Xs:           ring.Ternary{H: 1},
		RingType:     ring.Matrix,
		DefaultScale: rlwe.NewScale(1 << 45),
	}

	params, err := rlwe.NewParametersFromLiteral3N(paramsLiteral)
	if err != nil {
		t.Fatalf("Failed to create Matrix CKKS parameters: %v", err)
	}

	// Test complex vectors (like original CKKS)
	testValues := []complex128{1.0 + 0i, 2.0 + 0i, 3.0 + 0i, 4.0 + 0i}
	constant := 2.0

	// Create Matrix CKKS components
	encoder := NewEncoder(&params)
	kgen := NewKeyGenerator(&params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil)

	// Create plaintext and encode (like original CKKS)
	pt := NewPlaintext(&params, 6)
	err = encoder.Encode(testValues, pt)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Test constant multiplication
	ctScaled := NewCiphertext(&params, ct.Degree(), ct.Level())
	err = evaluator.MulByConst(ct, constant, ctScaled)
	if err != nil {
		t.Fatalf("Constant multiplication failed: %v", err)
	}

	// Decrypt and decode result (like original CKKS)
	ptResult := NewPlaintext(&params, ctScaled.Level())
	decryptor.Decrypt(ctScaled, ptResult)
	resultValues := make([]complex128, len(testValues))
	err = encoder.Decode(ptResult, &resultValues)
	if err != nil {
		t.Fatalf("Result decoding failed: %v", err)
	}

	// Calculate expected values
	expectedValues := make([]complex128, len(testValues))
	for i, val := range testValues {
		expectedValues[i] = val * complex(constant, 0)
	}

	// Verify results
	tolerance := 0.1

	for i := 0; i < len(testValues); i++ {
		expected := real(expectedValues[i])
		actual := real(resultValues[i])

		if actual < expected-tolerance || actual > expected+tolerance {
			t.Errorf("Value %d: expected ~%.1f, got %.1f", i, expected, actual)
		}
	}
}

// TestMatrixCKKSMultipleConstants tests Matrix CKKS with multiple different constants
func TestMatrixCKKSMultipleConstants(t *testing.T) {
	// Use Matrix CKKS parameters
	paramsLiteral := rlwe.ParametersLiteral3N{
		Order2:       3,
		Order3:       1,
		LogQ:         []int{55, 45, 45, 45, 45, 45, 45},
		LogP:         []int{60},
		Xe:           ring.Ternary{H: 1},
		Xs:           ring.Ternary{H: 1},
		RingType:     ring.Matrix,
		DefaultScale: rlwe.NewScale(1 << 45),
	}

	params, err := rlwe.NewParametersFromLiteral3N(paramsLiteral)
	if err != nil {
		t.Fatalf("Failed to create Matrix CKKS parameters: %v", err)
	}

	// Test complex vectors (like original CKKS)
	testValues := []complex128{1.0 + 0i, 2.0 + 0i, 3.0 + 0i}
	constants := []float64{2.0, 3.0, 4.0}

	// Create Matrix CKKS components
	encoder := NewEncoder(&params)
	kgen := NewKeyGenerator(&params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil)

	// Create plaintext and encode (like original CKKS)
	pt := NewPlaintext(&params, 6)
	err = encoder.Encode(testValues, pt)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Test each constant
	for _, constant := range constants {
		// Test constant multiplication
		ctScaled := NewCiphertext(&params, ct.Degree(), ct.Level())
		err = evaluator.MulByConst(ct, constant, ctScaled)
		if err != nil {
			t.Fatalf("Constant multiplication failed for %.1f: %v", constant, err)
		}

		// Decrypt and decode result (like original CKKS)
		ptResult := NewPlaintext(&params, ctScaled.Level())
		decryptor.Decrypt(ctScaled, ptResult)
		resultValues := make([]complex128, len(testValues))
		err = encoder.Decode(ptResult, &resultValues)
		if err != nil {
			t.Fatalf("Result decoding failed for %.1f: %v", constant, err)
		}

		// Calculate expected values
		expectedValues := make([]complex128, len(testValues))
		for i, val := range testValues {
			expectedValues[i] = val * complex(constant, 0)
		}

		// Verify results
		tolerance := 0.1

		for i := 0; i < len(testValues); i++ {
			expected := real(expectedValues[i])
			actual := real(resultValues[i])

			if actual < expected-tolerance || actual > expected+tolerance {
				t.Errorf("Constant %.1f, Value %d: expected ~%.1f, got %.1f", constant, i, expected, actual)
			}
		}
	}
}

// TestMatrixCKKSIntegerConstants tests Matrix CKKS with integer constants
func TestMatrixCKKSIntegerConstants(t *testing.T) {
	// Use Matrix CKKS parameters
	paramsLiteral := rlwe.ParametersLiteral3N{
		Order2:       3,
		Order3:       1,
		LogQ:         []int{55, 45, 45, 45, 45, 45, 45},
		LogP:         []int{60},
		Xe:           ring.Ternary{H: 1},
		Xs:           ring.Ternary{H: 1},
		RingType:     ring.Matrix,
		DefaultScale: rlwe.NewScale(1 << 45),
	}

	params, err := rlwe.NewParametersFromLiteral3N(paramsLiteral)
	if err != nil {
		t.Fatalf("Failed to create Matrix CKKS parameters: %v", err)
	}

	// Test complex vectors (like original CKKS)
	testValues := []complex128{1.0 + 0i, 2.0 + 0i, 3.0 + 0i, 4.0 + 0i}
	integerConstants := []int{2, 3, 4}

	// Create Matrix CKKS components
	encoder := NewEncoder(&params)
	kgen := NewKeyGenerator(&params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil)

	// Create plaintext and encode (like original CKKS)
	pt := NewPlaintext(&params, 6)
	err = encoder.Encode(testValues, pt)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Test each integer constant
	for _, constant := range integerConstants {
		// Test constant multiplication
		ctScaled := NewCiphertext(&params, ct.Degree(), ct.Level())
		err = evaluator.MulByConst(ct, constant, ctScaled)
		if err != nil {
			t.Fatalf("Constant multiplication failed for %d: %v", constant, err)
		}

		// Decrypt and decode result (like original CKKS)
		ptResult := NewPlaintext(&params, ctScaled.Level())
		decryptor.Decrypt(ctScaled, ptResult)
		resultValues := make([]complex128, len(testValues))
		err = encoder.Decode(ptResult, &resultValues)
		if err != nil {
			t.Fatalf("Result decoding failed for %d: %v", constant, err)
		}

		// Calculate expected values
		expectedValues := make([]complex128, len(testValues))
		for i, val := range testValues {
			expectedValues[i] = val * complex(float64(constant), 0)
		}

		// Verify results
		tolerance := 0.1

		for i := 0; i < len(testValues); i++ {
			expected := real(expectedValues[i])
			actual := real(resultValues[i])

			if actual < expected-tolerance || actual > expected+tolerance {
				t.Errorf("Integer constant %d, Value %d: expected ~%.1f, got %.1f", constant, i, expected, actual)
			}
		}
	}
}
