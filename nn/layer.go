package nn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	// Forward Returns the finished activation and the layer's cached state to be used during backpropagation.
	Forward(in *t.Tensor) (*t.Tensor, LayerState)

	// ComputeGradients returns the weight gradient for the current layer and the activation gradient for previous layers.
	ComputeGradients(state LayerState, nextLayerGradient *t.Tensor, gradClipping float64) (LayerGrad, *t.Tensor)

	// UpdateParams updates layer parameters based on gradient.
	UpdateParams(grads LayerGrad, learningRate float64)

	// Serialization methods
	save(w io.Writer) error // Writes the current layer (including name and constructor params)
}

// LayerGrad stores the gradient for a layer with respect to its parameters.
type LayerGrad interface {
	Add(another LayerGrad)
	Scale(factor float64)
}

// LayerState stores the cached values from forwarding to be reused for backpropagation
type LayerState interface {
	layerState() // Marker method
}

type layerType byte

const (
	FULLY_CONNECTED_LAYER layerType = iota
)

func loadLayer(r io.Reader) (Layer, error) {
	// Read type of layer
	var t layerType
	err := binary.Read(r, binary.LittleEndian, &t)
	if err != nil {
		return nil, err
	}

	// Call layer specific load function
	switch layerType(t) {
	case FULLY_CONNECTED_LAYER:
		return loadFullyConnectedLayer(r)
	default:
		return nil, errors.New(fmt.Sprint("Invalid layer type found: ", t))
	}
}
