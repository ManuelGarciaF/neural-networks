package nn

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	Forward(in *t.Tensor) *t.Tensor

	// Returns the gradients for the current and previous layers.
	ComputeGradients(nextLayerGradient *t.Tensor) (LayerGrad, *t.Tensor)

	// Updates parameters based on gradients.
	UpdateParams(grads []LayerGrad, learningRate float64)
}

// Stores the gradient for a layer with respect to its parameters.
type LayerGrad interface {
	// TODO Maybe move some logic here
}
