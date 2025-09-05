package matrix_ckks

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// PrecisionMode is a variable that defines how many primes (one
// per machine word) are required to store initial message values.
// This also sets how many primes are consumed per rescaling.
//
// There are currently two modes supported:
//   - PREC64 (one 64-bit word)
//   - PREC128 (two 64-bit words)
//
// PREC64 is the default mode and supports reference plaintext scaling
// factors of up to 2^{64}, while PREC128 scaling factors of up to 2^{128}.
//
// The PrecisionMode is chosen automatically based on the provided initial
// `LogDefaultScale` value provided by the user.
type PrecisionMode int

const (
	NTTFlag = true
	PREC64  = PrecisionMode(0)
	PREC128 = PrecisionMode(1)
)

// ParametersLiteral is a literal representation of Matrix CKKS parameters.
// It has public fields and is used to express unchecked user-defined parameters
// literally into Go programs. The [NewParametersFromLiteral] function is used to
// generate the actual checked parameters from the literal representation.
//
// The main difference from regular CKKS is that this uses 3N-friendly ring structure
// where N = 2^a * 3^b instead of just powers of 2.
//
// Users must set the polynomial degree (in actual value, N) and the coefficient modulus, by either setting
// the Q and P fields to the desired moduli chain, or by setting the LogQ and LogP fields to
// the desired moduli sizes (in log_2). Users must also specify a default initial scale for the plaintexts.
//
// Optionally, users may specify the error variance (Sigma), the secrets' density (H), the ring
// type (RingType). If left unset, standard default values for these field are substituted at parameter creation.
type ParametersLiteral3N struct {
	Order2          int
	Order3          int                         // Ring degree (must be of form 2^a * 3^b)
	NthRoot         int                         // Log2 of the 2N-th primitive root modulus
	Q               []uint64                    // Coefficient modulus (product of primes q_i)
	P               []uint64                    // Auxiliary modulus for RNS (product of primes p_j)
	LogQ            []int                       `json:",omitempty"` // Log2 sizes of Q primes
	LogP            []int                       `json:",omitempty"` // Log2 sizes of P primes
	Xe              ring.DistributionParameters // Error distribution
	Xs              ring.DistributionParameters // Secret distribution
	RingType        ring.Type                   // Ring type (Standard or ConjugateInvariant)
	LogDefaultScale int                         // Log2 of the default plaintext scale
}

// GetRLWEParametersLiteral returns the [rlwe.ParametersLiteral] from the target [matrix_ckks.ParameterLiteral].
func (p ParametersLiteral3N) GetRLWEParametersLiteral3N() rlwe.ParametersLiteral3N {
	// For 3N-ring, we should use LogN = 0 to indicate that we want to use
	// the actual N value directly rather than 2^LogN. The RLWE layer
	// will automatically detect the 3N structure and use the appropriate NTT.

	// Since RLWE expects LogN but we have actual N, we need to compute
	// the equivalent LogN that gives us at least N coefficients

	// logN := 0// The code snippet you provided is calculating the log base 2 of the ring degree `N` in
	// order to determine the appropriate value for the `LogN` parameter.

	// However, for 3N rings, we should try to use the N value directly
	// by setting appropriate LogNthRoot to indicate 3N structure
	return rlwe.ParametersLiteral3N{
		Order2:       p.Order2, // Power-of-2 that accommodates our N
		Order3:       p.Order3, // Power-of-2 that accommodates our N
		NthRoot:      1,        // Will be set automatically by Ring construction
		Q:            p.Q,
		P:            p.P,
		LogQ:         p.LogQ,
		LogP:         p.LogP,
		Xe:           p.Xe,
		Xs:           p.Xs,
		RingType:     p.RingType,
		NTTFlag:      NTTFlag,
		DefaultScale: rlwe.NewScale(math.Exp2(float64(p.LogDefaultScale))),
	}
}

// Parameters represents a parameter set for the Matrix CKKS cryptosystem. Its fields are private and
// immutable. See [ParametersLiteral] for user-specified parameters.
type Parameters3N struct {
	rlwe.Parameters3N
}

// MaxDimensions returns the maximum dimension of the matrix that can be SIMD packed in a single plaintext polynomial.
// TODO: Catch ring.Matrix case
func (p Parameters3N) MaxDimensions() ring.Dimensions {
	switch p.RingType() {
	case ring.Matrix:
		power3 := 1
		for i := 0; i < p.Order3(); i++ {
			power3 *= 3
		}
		return ring.Dimensions{Rows: p.Order2() >> 2, Cols: power3 << 1}

	default:
		panic("cannot MaxDimensions: invalid ring type")
	}
}

// MaxSlots returns the total number of entries (slots) that a plaintext can store.
// This value is obtained by multiplying all dimensions from MaxDimensions.
func (p Parameters3N) MaxSlots() int {
	dims := p.MaxDimensions()
	return dims.Rows * dims.Cols
}

// MaxLevel returns the maximum ciphertext level
func (p Parameters3N) MaxLevel() int {
	return p.QCount() - 1
}

// NewParametersFromLiteral instantiate a set of Matrix CKKS parameters from a [ParametersLiteral] specification.
// It returns the empty parameters [Parameters]{} and a non-nil error if the specified parameters are invalid.
//
// The main validation is that N must be of the form 2^a * 3^b and we create 3N rings directly.
//
// See [rlwe.NewParametersFromLiteral] for default values of the other optional fields.
func NewParametersFromLiteral3N(pl ParametersLiteral3N) (Parameters3N, error) {
	// Validate that N is of form 2^a * 3^b
	if !isValid3NForm(pl.N) {
		return Parameters3N{}, fmt.Errorf("N=%d must be of form 2^a * 3^b", pl.N)
	}

	// Create the underlying RLWE parameters (this will use power-of-2 rings)
	rlweParams, err := rlwe.NewParametersFromLiteral(pl.GetRLWEParametersLiteral3N())
	if err != nil {
		return Parameters{}, fmt.Errorf("cannot NewParametersFromLiteral: %w", err)
	}

	if pl.LogDefaultScale > 128 {
		return Parameters{}, fmt.Errorf("cannot NewParametersFromLiteral: LogDefaultScale=%d > 128 or < 0", pl.LogDefaultScale)
	}

	// Create 3N rings directly using our N value
	var ringQ, ringP *ring.Ring

	// Build moduli from the parameter literal
	var modQ, modP []uint64
	if len(pl.Q) > 0 {
		modQ = pl.Q
	} else {
		// Generate 3N-friendly moduli from LogQ
		for _, logq := range pl.LogQ {
			// Find a 3N-friendly prime for this bit size
			primes, err := ring.Find3NRNSPrimes(pl.N, logq, 1, 1<<16)
			if err != nil {
				return Parameters{}, fmt.Errorf("failed to find 3N-friendly prime for LogQ=%d: %w", logq, err)
			}
			modQ = append(modQ, primes[0])
		}
	}

	if len(pl.P) > 0 {
		modP = pl.P
	} else if len(pl.LogP) > 0 {
		// Generate 3N-friendly moduli from LogP
		for _, logp := range pl.LogP {
			// Find a 3N-friendly prime for this bit size
			primes, err := ring.Find3NRNSPrimes(pl.N, logp, 1, 1<<16)
			if err != nil {
				return Parameters{}, fmt.Errorf("failed to find 3N-friendly prime for LogP=%d: %w", logp, err)
			}
			modP = append(modP, primes[0])
		}
	}

	// Create 3N rings directly
	if ringQ, err = ring.NewRingFromType(pl.N, modQ, pl.RingType); err != nil {
		return Parameters{}, fmt.Errorf("failed to create 3N ringQ: %w", err)
	}

	if len(modP) > 0 {
		if ringP, err = ring.NewRingFromType(pl.N, modP, pl.RingType); err != nil {
			return Parameters{}, fmt.Errorf("failed to create 3N ringP: %w", err)
		}
	}

	return Parameters{
		Parameters: rlweParams,
		n:          pl.N,
		ringQ:      ringQ,
		ringP:      ringP,
	}, nil
}

// isValid3NForm checks if n is of the form 2^a * 3^b
func isValid3NForm(n int) bool {
	if n <= 0 {
		return false
	}

	// Remove all factors of 2
	for n%2 == 0 {
		n /= 2
	}

	// Remove all factors of 3
	for n%3 == 0 {
		n /= 3
	}

	// Should be left with 1 if it was only 2s and 3s
	return n == 1
}

// MarshalBinary encodes the parameters into a binary form.
func (p Parameters) MarshalBinary() (data []byte, err error) {
	return p.Parameters.MarshalBinary()
}

// UnmarshalBinary decodes a binary form into parameters.
func (p *Parameters) UnmarshalBinary(data []byte) (err error) {
	return p.Parameters.UnmarshalBinary(data)
}

// MarshalJSON encodes the parameters into JSON.
func (p Parameters) MarshalJSON() (data []byte, err error) {
	return json.Marshal(p.ParametersLiteral())
}

// UnmarshalJSON decodes JSON into parameters.
func (p *Parameters) UnmarshalJSON(data []byte) (err error) {
	var pl ParametersLiteral
	if err = json.Unmarshal(data, &pl); err != nil {
		return err
	}
	*p, err = NewParametersFromLiteral(pl)
	return err
}

// ParametersLiteral returns the [ParametersLiteral] of the target [Parameters].
func (p Parameters) ParametersLiteral() ParametersLiteral {
	pl := p.Parameters.ParametersLiteral()
	return ParametersLiteral{
		N:               p.n,
		LogNthRoot:      pl.LogNthRoot,
		Q:               pl.Q,
		P:               pl.P,
		LogQ:            pl.LogQ,
		LogP:            pl.LogP,
		Xe:              pl.Xe,
		Xs:              pl.Xs,
		RingType:        pl.RingType,
		LogDefaultScale: int(math.Log2(pl.DefaultScale.Float64())),
	}
}

// LogDefaultScale returns the log2 of the default plaintext
// scaling factor (rounded to the nearest integer).
func (p Parameters) LogDefaultScale() int {
	return int(math.Round(math.Log2(p.DefaultScale().Float64())))
}

// EncodingPrecision returns the encoding precision in bits of the plaintext values which
// is max(53, log2(DefaultScale)).
func (p Parameters) EncodingPrecision() (prec uint) {
	if log2scale := math.Log2(p.DefaultScale().Float64()); log2scale <= 53 {
		prec = 53
	} else {
		prec = uint(log2scale)
	}

	return
}

// PrecisionMode returns the precision mode of the parameters.
// This value can be [ckks.PREC64] or [ckks.PREC128].
func (p Parameters) PrecisionMode() PrecisionMode {
	if p.LogDefaultScale() <= 64 {
		return PREC64
	}
	return PREC128
}

// LevelsConsumedPerRescaling returns the number of levels (i.e. primes)
// consumed per rescaling. This value is 1 if the precision mode is PREC64
// and is 2 if the precision mode is PREC128.
func (p Parameters) LevelsConsumedPerRescaling() int {
	switch p.PrecisionMode() {
	case PREC128:
		return 2
	default:
		return 1
	}
}
