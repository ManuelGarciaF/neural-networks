package nn

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	// Returns the finished activation and the layer's cached state to be used during backpropagation.
	Forward(in *t.Tensor) (*t.Tensor, LayerState)

	// Returns the gradients for the current and previous layers.
	ComputeGradients(state LayerState, nextLayerGradient *t.Tensor) (LayerGrad, *t.Tensor)

	// Updates parameters based on gradients.
	UpdateParams(grads []LayerGrad, learningRate float64)
}

// Stores the gradient for a layer with respect to its parameters.
type LayerGrad interface {
	layerGrad() // Marker method
}

// Each layer type needs different values cached during its forwarding for backpropagation
type LayerState interface {
	layerState() // Marker method
}
