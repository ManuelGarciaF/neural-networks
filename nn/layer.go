package nn

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	Forward(in *t.Tensor) *t.Tensor

	// Returns the gradient for the layer and accumulates gradients for parameters.
	Backward(nextLayerGradient *t.Tensor) *t.Tensor

	// Updates de parameters based on accumulated gradients, uses numberSamples to
	// average the gradients before applying.
	// TODO check if this is necessary with a small enough learningRate
	UpdateParams(numberSamples int, learningRate float64)

	ClearGradients()
}
