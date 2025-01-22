package nn

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	// Forward Returns the finished activation and the layer's cached state to be used during backpropagation.
	Forward(in *t.Tensor) (*t.Tensor, LayerState)

	// ComputeGradients returns the gradients for the current and previous layers.
	ComputeGradients(state LayerState, nextLayerGradient *t.Tensor, gradClipping float64) (LayerGrad, *t.Tensor)

	// UpdateParams updates layer parameters based on gradients.
	UpdateParams(grads []LayerGrad, learningRate float64)
}

// LayerGrad stores the gradient for a layer with respect to its parameters.
type LayerGrad interface {
	layerGrad() // Marker method
}

// LayerState stores the cached values from forwarding to be reused for backpropagation
type LayerState interface {
	layerState() // Marker method
}
