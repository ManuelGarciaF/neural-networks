package nn

import (
	"math"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type NeuralNetwork struct {
	Layers []Layer
}

type TrainingSample struct{In, Out *t.Tensor} // Both column vectors

// Arch is a list of layer sizes, including input and output
func NewNeuralNetwork(arch []int) *NeuralNetwork {
	layers := make([]Layer, 0, (len(arch)-1)*2)

	for i := 0; i < len(arch)-1; i++ {
		layers = append(layers, NewFullyConnectedLayer(arch[i], arch[i+1]))
		layers = append(layers, ReLU{})
	}

	return &NeuralNetwork{Layers: layers}
}

func (n *NeuralNetwork) Forward(input *t.Tensor) *t.Tensor {
	activations := make([]*t.Tensor, len(n.Layers)+1)
	activations[0] = input
	for i, l := range n.Layers {
		activations[i+1] = l.Forward(activations[i])
	}
	return activations[len(activations)-1]
}

// Mean squared error
func (n *NeuralNetwork) Loss(samples []TrainingSample) float64 {
	sum := float64(0)
	for _, sample := range samples {
		// Add up the loss for each example
		actual := n.Forward(sample.In)
		for i := 0; i < actual.Dim(1); i++ {
			sum += math.Pow(sample.Out.At(i)-actual.At(i), 2)
		}
	}
	return sum/float64(len(samples))
}
