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
			LogNthRoot:      0,             // Will be computed automatically
			LogQ:            []int{30, 20}, // Two primes for basic operations
			LogP:            []int{20},     // One auxiliary prime
			Xe:              ring.Ternary{H: 12},
			Xs:              ring.Ternary{H: 12},
			RingType:        ring.Standard,
			LogDefaultScale: 20,
		},
		// Medium example: N=48 (2^4 * 3^1)
		{
			N:               48,
			LogNthRoot:      0,                 // Will be computed automatically
			LogQ:            []int{50, 40, 40}, // Three primes
			LogP:            []int{60},         // One auxiliary prime
			Xe:              ring.Ternary{H: 192},
			Xs:              ring.Ternary{H: 192},
			RingType:        ring.Standard,
			LogDefaultScale: 40,
		},
		// Larger example: N=96 (2^5 * 3^1)
		{
			N:               96,
			LogNthRoot:      0,                     // Will be computed automatically
			LogQ:            []int{55, 45, 45, 45}, // Four primes for more operations
			LogP:            []int{60},             // One auxiliary prime
			Xe:              ring.Ternary{H: 192},
			Xs:              ring.Ternary{H: 192},
			RingType:        ring.Standard,
			LogDefaultScale: 40,
		},
	}
}

// ExampleParameters returns example parameters for testing Matrix CKKS.
func ExampleParameters() []Parameters {
	var params []Parameters
	literals := ExampleParametersLiteral()

	for _, pl := range literals {
		p, err := NewParametersFromLiteral(pl)
		if err == nil {
			params = append(params, p)
		}
	}

	return params
}
