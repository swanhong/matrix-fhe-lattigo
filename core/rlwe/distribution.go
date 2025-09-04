package rlwe

import (
	"math"

	"github.com/tuneinsight/lattigo/v6/ring"
)

type Distribution struct {
	ring.DistributionParameters
	Std      float64
	AbsBound float64
}

func NewDistribution(params ring.DistributionParameters, logN int) (d Distribution) {
	d.DistributionParameters = params
	switch params := params.(type) {
	case ring.DiscreteGaussian:
		d.Std = params.Sigma
		d.AbsBound = params.Bound
	case ring.Ternary:
		if params.P != 0 {
			d.Std = math.Sqrt(1 - params.P)
		} else {
			d.Std = math.Sqrt(float64(params.H) / (math.Exp2(float64(logN)) - 1))
		}
		d.AbsBound = 1
	default:
		// Sanity check
		panic("invalid dist")
	}
	return
}

func NewDistribution3N(params ring.DistributionParameters, order2 int, order3 int) (d Distribution) {

	ringDim := 1 << uint(order2)  // 2^order2
	for i := 0; i < order3; i++ { // 3^order3
		ringDim *= 3
	}

	d.DistributionParameters = params
	switch params := params.(type) {
	case ring.DiscreteGaussian:
		d.Std = params.Sigma
		d.AbsBound = params.Bound
	case ring.Ternary:
		if params.P != 0 {
			d.Std = math.Sqrt(1 - params.P)
		} else {
			d.Std = math.Sqrt(float64(params.H) / float64(ringDim-1))
		}
		d.AbsBound = 1
	default:
		// Sanity check
		panic("invalid dist")
	}
	return
}
