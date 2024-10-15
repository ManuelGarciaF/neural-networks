package nn

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type ReLU struct{}

var _ Layer = ReLU{}

func (ReLU) Forward(in *t.Tensor) *t.Tensor {
	in.Apply(func(v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	})
	return in
}
