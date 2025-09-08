package matrix_ckks

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// ExampleParametersLiteral returns example parameters for testing Matrix CKKS.
// These parameters use 3N-friendly ring structures.
func ExampleParametersLiteral() []rlwe.ParametersLiteral3N {
	return []rlwe.ParametersLiteral3N{
		// Small example: N=24 (2^3 * 3^1) - Proper RNS-CKKS design
		{
			Order2:       3,                  // 2-factor: 2^3 = 8
			Order3:       1,                  // 3-factor: 3^1 = 3, so N = 8*3 = 24
			NthRoot:      0,                  // Will be computed automatically
			LogQ:         []int{30, 30},      // Two 30-bit moduli for proper rescaling
			LogP:         []int{40},          // Larger auxiliary prime
			Xe:           ring.Ternary{H: 1}, // Minimal noise
			Xs:           ring.Ternary{H: 1}, // Minimal secret weight
			RingType:     ring.Matrix,
			DefaultScale: rlwe.NewScale(1 << 20), // Smaller scale to avoid modular wraparound
		},
		// Medium example: N=48 (2^4 * 3^1)
		{
			Order2:       4,                 // 2-factor: 2^4 = 16
			Order3:       1,                 // 3-factor: 3^1 = 3, so N = 16*3 = 48
			NthRoot:      0,                 // Will be computed automatically
			LogQ:         []int{50, 40, 40}, // Three primes (will try to find 3N-friendly ones)
			LogP:         []int{60},         // One auxiliary prime
			Xe:           ring.Ternary{H: 192},
			Xs:           ring.Ternary{H: 192},
			RingType:     ring.Matrix,
			DefaultScale: rlwe.NewScale(1 << 40),
		},
		// Larger example: N=96 (2^5 * 3^1)
		{
			Order2:       5,                     // 2-factor: 2^5 = 32
			Order3:       1,                     // 3-factor: 3^1 = 3, so N = 32*3 = 96
			NthRoot:      0,                     // Will be computed automatically
			LogQ:         []int{40, 30, 30, 30}, // Larger moduli for security
			LogP:         []int{40},             // Larger auxiliary prime
			Xe:           ring.Ternary{H: 8},    // Much smaller noise (8 instead of 32)
			Xs:           ring.Ternary{H: 8},    // Much smaller secret weight
			RingType:     ring.Matrix,
			DefaultScale: rlwe.NewScale(1 << 10),
		},
	}
}
