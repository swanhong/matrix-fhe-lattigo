package matrix_ckks

import (
	"github.com/tuneinsight/lattigo/v6/ring"
)

// ExampleParametersLiteral returns example parameters for testing Matrix CKKS.
// These parameters use 3N-friendly ring structures.
func ExampleParametersLiteral() []ParametersLiteral {
	return []ParametersLiteral{
		// Small example: N=24 (2^3 * 3^1)
		{
			N:               24,
			PowerOf2:        3,             // 2-factor: 2^3 = 8
			PowerOf3:        1,             // 3-factor: 3^1 = 3, so N = 8*3 = 24
			LogNthRoot:      0,             // Will be computed automatically
			LogQ:            []int{30, 20}, // Two primes for basic operations
			LogP:            []int{20},     // One auxiliary prime
			Xe:              ring.Ternary{H: 12},
			Xs:              ring.Ternary{H: 12},
			RingType:        ring.Matrix,
			LogDefaultScale: 20,
		},
		// Medium example: N=48 (2^4 * 3^1)
		{
			N:               48,
			PowerOf2:        4,                 // 2-factor: 2^4 = 16
			PowerOf3:        1,                 // 3-factor: 3^1 = 3, so N = 16*3 = 48
			LogNthRoot:      0,                 // Will be computed automatically
			LogQ:            []int{50, 40, 40}, // Three primes (will try to find 3N-friendly ones)
			LogP:            []int{60},         // One auxiliary prime
			Xe:              ring.Ternary{H: 192},
			Xs:              ring.Ternary{H: 192},
			RingType:        ring.Matrix,
			LogDefaultScale: 40,
		},
		// Larger example: N=96 (2^5 * 3^1)
		{
			N:               96,
			PowerOf2:        5,                     // 2-factor: 2^5 = 32
			PowerOf3:        1,                     // 3-factor: 3^1 = 3, so N = 32*3 = 96
			LogNthRoot:      0,                     // Will be computed automatically
			LogQ:            []int{40, 30, 30, 30}, // Larger moduli for security
			LogP:            []int{40},             // Larger auxiliary prime
			Xe:              ring.Ternary{H: 8},    // Much smaller noise (8 instead of 32)
			Xs:              ring.Ternary{H: 8},    // Much smaller secret weight
			RingType:        ring.Matrix,
			LogDefaultScale: 10, // Keep scale small for debugging
		},
	}
}
