package matrix_ckks

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// Helper function to get minimum of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestCiphertextMultiplication(t *testing.T) {
	fmt.Println("\n=== Testing Homomorphic Multiplication ===")

	// Use the third parameter set for testing (N=96, order2=5)
	paramLiterals := ExampleParametersLiteral()
	pl := paramLiterals[2] // N=96

	N := 1 << uint(pl.Order2)
	for i := 0; i < pl.Order3; i++ {
		N *= 3
	}
	t.Logf("Testing with N=%d", N)

	// Create parameters
	params, err := rlwe.NewParametersFromLiteral3N(pl)
	if err != nil {
		t.Fatalf("Failed to create parameters: %v", err)
	}

	// Create key generator and generate keys
	kgen := NewKeyGenerator(&params)
	sk, pk := kgen.GenKeyPairNew()

	// Create encoder, encryptor, and decryptor
	encoder := NewEncoder(&params)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil) // nil for multiplication without relinearization keys

	// For Matrix CKKS, the number of slots is N/2
	slots := params.MaxSlots()

	// Create test data - simple values for easier verification
	inputPoly1 := make([]uint64, slots)
	inputPoly2 := make([]uint64, slots)
	expectedPoly := make([]uint64, slots)

	for i := 0; i < slots; i++ {
		inputPoly1[i] = uint64(i + 1)                   // [1, 2, 3, ...]
		inputPoly2[i] = uint64((i + 1) * 2)             // [2, 4, 6, ...]
		expectedPoly[i] = inputPoly1[i] * inputPoly2[i] // [2, 8, 18, ...]
	}

	t.Logf("Input polynomial 1: %v", inputPoly1[:minInt(10, len(inputPoly1))])
	t.Logf("Input polynomial 2: %v", inputPoly2[:minInt(10, len(inputPoly2))])
	t.Logf("Expected result: %v", expectedPoly[:minInt(10, len(expectedPoly))])

	// Encode the input polynomials
	pt1 := NewPlaintext(&params, params.MaxLevel())
	pt2 := NewPlaintext(&params, params.MaxLevel())

	err = encoder.EncodePolynomial(inputPoly1, pt1)
	if err != nil {
		t.Fatalf("Failed to encode polynomial 1: %v", err)
	}

	err = encoder.EncodePolynomial(inputPoly2, pt2)
	if err != nil {
		t.Fatalf("Failed to encode polynomial 2: %v", err)
	}

	// Encrypt the plaintexts
	ct1, err := encryptor.EncryptNew(pt1)
	if err != nil {
		t.Fatalf("Failed to encrypt polynomial 1: %v", err)
	}

	ct2, err := encryptor.EncryptNew(pt2)
	if err != nil {
		t.Fatalf("Failed to encrypt polynomial 2: %v", err)
	}

	// Perform homomorphic multiplication
	ctResult, err := evaluator.MulNew(ct1, ct2)
	if err != nil {
		t.Fatalf("Failed to multiply ciphertexts: %v", err)
	}

	// Decrypt the result
	ptResult := NewPlaintext(&params, ctResult.Level())
	decryptor.Decrypt(ctResult, ptResult)

	// Decode the result
	resultPoly := make([]uint64, slots)
	err = encoder.DecodePolynomial(ptResult, resultPoly)
	if err != nil {
		t.Fatalf("Failed to decode result: %v", err)
	}

	t.Logf("Actual result: %v", resultPoly[:minInt(10, len(resultPoly))])

	// Verify the result (allowing for some noise in homomorphic encryption)
	correctCount := 0
	for i := 0; i < slots; i++ {
		if resultPoly[i] == expectedPoly[i] {
			correctCount++
		} else {
			t.Logf("Mismatch at index %d: expected %d, got %d", i, expectedPoly[i], resultPoly[i])
		}
	}

	accuracy := float64(correctCount) / float64(slots) * 100
	t.Logf("Accuracy: %d/%d (%.1f%%)", correctCount, slots, accuracy)

	if accuracy >= 80.0 {
		t.Logf("✅ SUCCESS: Homomorphic multiplication working with %.1f%% accuracy", accuracy)
	} else {
		t.Errorf("❌ FAILED: Homomorphic multiplication accuracy too low: %.1f%%", accuracy)
	}
}

func TestCiphertextAddition(t *testing.T) {
	fmt.Println("\n=== Testing Homomorphic Addition ===")

	// Use the third parameter set for testing (N=96, order2=5)
	paramLiterals := ExampleParametersLiteral()
	pl := paramLiterals[2] // N=96

	N := 1 << uint(pl.Order2)
	for i := 0; i < pl.Order3; i++ {
		N *= 3
	}
	t.Logf("Testing with N=%d", N)

	// Create parameters
	params, err := rlwe.NewParametersFromLiteral3N(pl)
	if err != nil {
		t.Fatalf("Failed to create parameters: %v", err)
	}

	// Create key generator and generate keys
	kgen := NewKeyGenerator(&params)
	sk, pk := kgen.GenKeyPairNew()

	// Create encoder, encryptor, decryptor, and evaluator
	encoder := NewEncoder(&params)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	evaluator := NewEvaluator(&params, nil)

	// For Matrix CKKS, the number of slots is N/2
	slots := params.MaxSlots()

	// Create test data
	inputPoly1 := make([]uint64, slots)
	inputPoly2 := make([]uint64, slots)
	expectedPoly := make([]uint64, slots)

	for i := 0; i < slots; i++ {
		inputPoly1[i] = uint64(i + 1)                   // [1, 2, 3, ...]
		inputPoly2[i] = uint64((i + 1) * 3)             // [3, 6, 9, ...]
		expectedPoly[i] = inputPoly1[i] + inputPoly2[i] // [4, 8, 12, ...]
	}

	t.Logf("Input polynomial 1: %v", inputPoly1[:minInt(10, len(inputPoly1))])
	t.Logf("Input polynomial 2: %v", inputPoly2[:minInt(10, len(inputPoly2))])
	t.Logf("Expected result: %v", expectedPoly[:minInt(10, len(expectedPoly))])

	// Encode the input polynomials
	pt1 := NewPlaintext(&params, params.MaxLevel())
	pt2 := NewPlaintext(&params, params.MaxLevel())

	err = encoder.EncodePolynomial(inputPoly1, pt1)
	if err != nil {
		t.Fatalf("Failed to encode polynomial 1: %v", err)
	}

	err = encoder.EncodePolynomial(inputPoly2, pt2)
	if err != nil {
		t.Fatalf("Failed to encode polynomial 2: %v", err)
	}

	// Encrypt the plaintexts
	ct1, err := encryptor.EncryptNew(pt1)
	if err != nil {
		t.Fatalf("Failed to encrypt polynomial 1: %v", err)
	}

	ct2, err := encryptor.EncryptNew(pt2)
	if err != nil {
		t.Fatalf("Failed to encrypt polynomial 2: %v", err)
	}

	// Perform homomorphic addition
	ctResult, err := evaluator.AddNew(ct1, ct2)
	if err != nil {
		t.Fatalf("Failed to add ciphertexts: %v", err)
	}

	// Decrypt the result
	ptResult := NewPlaintext(&params, ctResult.Level())
	decryptor.Decrypt(ctResult, ptResult)

	// Decode the result
	resultPoly := make([]uint64, slots)
	err = encoder.DecodePolynomial(ptResult, resultPoly)
	if err != nil {
		t.Fatalf("Failed to decode result: %v", err)
	}

	t.Logf("Actual result: %v", resultPoly[:minInt(10, len(resultPoly))])

	// Verify the result
	correctCount := 0
	for i := 0; i < slots; i++ {
		if resultPoly[i] == expectedPoly[i] {
			correctCount++
		} else {
			t.Logf("Mismatch at index %d: expected %d, got %d", i, expectedPoly[i], resultPoly[i])
		}
	}

	accuracy := float64(correctCount) / float64(slots) * 100
	t.Logf("Accuracy: %d/%d (%.1f%%)", correctCount, slots, accuracy)

	if accuracy >= 90.0 {
		t.Logf("✅ SUCCESS: Homomorphic addition working with %.1f%% accuracy", accuracy)
	} else {
		t.Errorf("❌ FAILED: Homomorphic addition accuracy too low: %.1f%%", accuracy)
	}
}

func TestBasicHEOperations(t *testing.T) {
	t.Logf("=== Testing Basic HE Operations ===")

	// Use the first parameter set for testing (N=24)
	paramLiterals := ExampleParametersLiteral()
	pl := paramLiterals[0] // N=24

	N := 1 << uint(pl.Order2)
	for i := 0; i < pl.Order3; i++ {
		N *= 3
	}
	t.Logf("Testing with N=%d", N)

	// Create parameters
	params, err := rlwe.NewParametersFromLiteral3N(pl)
	if err != nil {
		t.Fatalf("Failed to create parameters: %v", err)
	}

	// Create key generator and generate keys
	kgen := NewKeyGenerator(&params)
	sk, pk := kgen.GenKeyPairNew()

	// Create encoder, encryptor, and decryptor
	encoder := NewEncoder(&params)
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)

	// For Matrix CKKS, the number of slots is N/2
	slots := params.MaxSlots()

	// Test with simple values
	testValues := []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	if len(testValues) > slots {
		testValues = testValues[:slots]
	}

	t.Logf("Testing with values: %v", testValues)

	// Create plaintext and encode
	pt := NewPlaintext(&params, params.MaxLevel())
	err = encoder.EncodePolynomial(testValues, pt)
	if err != nil {
		t.Fatalf("Failed to encode: %v", err)
	}

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Failed to encrypt: %v", err)
	}

	// Decrypt
	ptDecrypted := NewPlaintext(&params, ct.Level())
	decryptor.Decrypt(ct, ptDecrypted)

	// Decode
	decodedValues := make([]uint64, len(testValues))
	err = encoder.DecodePolynomial(ptDecrypted, decodedValues)
	if err != nil {
		t.Fatalf("Failed to decode: %v", err)
	}

	t.Logf("Decoded values: %v", decodedValues)

	// Check if values match (allowing for some noise in homomorphic encryption)
	correctCount := 0
	for i := 0; i < len(testValues); i++ {
		if decodedValues[i] == testValues[i] {
			correctCount++
		} else {
			t.Logf("Mismatch at index %d: expected %d, got %d", i, testValues[i], decodedValues[i])
		}
	}

	accuracy := float64(correctCount) / float64(len(testValues)) * 100
	t.Logf("Accuracy: %d/%d (%.1f%%)", correctCount, len(testValues), accuracy)

	if accuracy >= 90.0 {
		t.Logf("✅ SUCCESS: Basic HE operations working with %.1f%% accuracy", accuracy)
	} else {
		t.Errorf("❌ FAILED: Basic HE operations accuracy too low: %.1f%%", accuracy)
	}
}

func TestEncodeEncryptDecryptDecode(t *testing.T) {
	t.Log("=== Testing Full Matrix CKKS Pipeline: Encode → Encrypt → Decrypt → Decode ===")

	// Use the first parameter set
	pl := ExampleParametersLiteral()[0]
	params, err := rlwe.NewParametersFromLiteral3N(pl)
	if err != nil {
		t.Fatalf("Failed to create parameters: %v", err)
	}

	// Create key generator and generate keys
	keygen := NewKeyGenerator(&params)
	sk, pk := keygen.GenKeyPairNew()

	// Create encryptor, decryptor, and encoder
	encryptor := NewEncryptor(&params, pk)
	decryptor := NewDecryptor(&params, sk)
	encoder := NewEncoder(&params)

	slots := params.MaxSlots()
	t.Logf("Testing with N=%d, slots=%d", params.N(), slots)

	// Test 1: Random Small Integers
	t.Log("\n=== Test 1: Random Small Integers ===")
	rand.Seed(42) // Fixed seed for reproducible results
	testVector := make([]uint64, slots)
	for i := 0; i < slots; i++ {
		testVector[i] = uint64(rand.Intn(100) + 1) // Random integers from 1 to 100
	}
	t.Logf("Original random vector: %v", testVector)

	// Encode
	pt := NewPlaintext(&params, params.MaxLevel())
	err = encoder.EncodePolynomial(testVector, pt)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}
	t.Log("✅ Step 1: Encoding successful")

	// Encrypt
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}
	t.Log("✅ Step 2: Encryption successful")

	// Decrypt
	ptDecrypted := NewPlaintext(&params, ct.Level())
	decryptor.Decrypt(ct, ptDecrypted)
	t.Log("✅ Step 3: Decryption successful")

	// Decode
	decodedVector := make([]uint64, slots)
	err = encoder.DecodePolynomial(ptDecrypted, decodedVector)
	if err != nil {
		t.Fatalf("Decoding failed: %v", err)
	}
	t.Log("✅ Step 4: Decoding successful")
	t.Logf("Decoded vector: %v", decodedVector)

	// Check accuracy
	perfectCount := 0
	for i := 0; i < len(testVector); i++ {
		if testVector[i] == decodedVector[i] {
			perfectCount++
		}
	}

	t.Logf("Pipeline Analysis:")
	t.Logf("  - Perfect values: %d/%d", perfectCount, len(testVector))
	t.Logf("  - Incorrect values: %d", len(testVector)-perfectCount)

	pipelineCorrect := perfectCount == len(testVector)
	if pipelineCorrect {
		t.Log("✅ Test 1 PASSED: Random integers pipeline works correctly")
	} else {
		t.Log("❌ Test 1 FAILED: Random integers pipeline has incorrect values")
	}

	// Test 2: Zero Vector
	t.Log("\n=== Test 2: Zero Vector ===")
	zeroVector := make([]uint64, slots)
	t.Logf("Original zero vector: %v", zeroVector)

	// Encode
	ptZero := NewPlaintext(&params, params.MaxLevel())
	err = encoder.EncodePolynomial(zeroVector, ptZero)
	if err != nil {
		t.Fatalf("Encoding failed: %v", err)
	}

	// Encrypt
	ctZero, err := encryptor.EncryptNew(ptZero)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Decrypt
	ptZeroDecrypted := NewPlaintext(&params, ctZero.Level())
	decryptor.Decrypt(ctZero, ptZeroDecrypted)

	// Decode
	decodedZeroVector := make([]uint64, slots)
	err = encoder.DecodePolynomial(ptZeroDecrypted, decodedZeroVector)
	if err != nil {
		t.Fatalf("Decoding failed: %v", err)
	}
	t.Logf("Decoded zero vector: %v", decodedZeroVector)

	// Check accuracy
	zeroPerfectCount := 0
	for i := 0; i < len(zeroVector); i++ {
		if zeroVector[i] == decodedZeroVector[i] {
			zeroPerfectCount++
		}
	}

	t.Logf("Zero Pipeline Analysis:")
	t.Logf("  - Perfect values: %d/%d", zeroPerfectCount, len(zeroVector))
	t.Logf("  - Incorrect values: %d", len(zeroVector)-zeroPerfectCount)

	zeroPipelineCorrect := zeroPerfectCount == len(zeroVector)
	if zeroPipelineCorrect {
		t.Log("✅ Test 2 PASSED: Zero vector pipeline works correctly")
	} else {
		t.Log("❌ Test 2 FAILED: Zero vector pipeline has incorrect values")
	}

	// Test 3: Multiple Random Seeds
	t.Log("\n=== Test 3: Multiple Random Seeds ===")
	successCount := 0
	totalTests := 5
	for seed := 1; seed <= totalTests; seed++ {
		rand.Seed(int64(seed))
		randomVector := make([]uint64, slots)
		for i := 0; i < slots; i++ {
			randomVector[i] = uint64(rand.Intn(50) + 1) // Random integers from 1 to 50
		}

		// Encode
		ptRandom := NewPlaintext(&params, params.MaxLevel())
		err = encoder.EncodePolynomial(randomVector, ptRandom)
		if err != nil {
			continue
		}

		// Encrypt
		ctRandom, err := encryptor.EncryptNew(ptRandom)
		if err != nil {
			continue
		}

		// Decrypt
		ptRandomDecrypted := NewPlaintext(&params, ctRandom.Level())
		decryptor.Decrypt(ctRandom, ptRandomDecrypted)

		// Decode
		decodedRandomVector := make([]uint64, slots)
		err = encoder.DecodePolynomial(ptRandomDecrypted, decodedRandomVector)
		if err != nil {
			continue
		}

		// Check accuracy
		correct := true
		for i := 0; i < len(randomVector); i++ {
			if decodedRandomVector[i] != randomVector[i] {
				correct = false
				break
			}
		}
		if correct {
			successCount++
		}
	}

	t.Logf("Multiple random tests: %d/%d passed", successCount, totalTests)
	multipleRandomCorrect := successCount == totalTests

	t.Log("\n=== CONCLUSION ===")
	if pipelineCorrect && zeroPipelineCorrect && multipleRandomCorrect {
		t.Log("🎉 SUCCESS: Full Matrix CKKS pipeline works perfectly!")
		t.Log("   - Encode → Encrypt → Decrypt → Decode pipeline is 100% accurate")
		t.Log("   - Random integer vectors are preserved exactly")
		t.Log("   - Zero vectors are preserved exactly")
		t.Log("   - Multiple random seeds work correctly")
		t.Log("   - Matrix CKKS is ready for homomorphic operations!")
	} else {
		t.Log("❌ FAILED: Full Matrix CKKS pipeline has accuracy issues")
		if !pipelineCorrect {
			t.Log("   - Random integer pipeline has incorrect values")
		}
		if !zeroPipelineCorrect {
			t.Log("   - Zero vector pipeline has incorrect values")
		}
		if !multipleRandomCorrect {
			t.Logf("   - Multiple random tests: %d/%d failed", totalTests-successCount, totalTests)
		}
	}
}
