package nn

import (
	"fmt"
	"math"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

const LOG_PROGRESS = true

type NeuralNetwork struct {
	Layers []Layer
}

type TrainingSample struct{ In, Out *t.Tensor } // Both column vectors

// Arch is a list of layer sizes, including input and output
func NewMLP(arch []int, actF ActivationFunction) *NeuralNetwork {
	layers := make([]Layer, 0, len(arch)-1)

	for i := 0; i < len(arch)-1; i++ {
		layers = append(layers, NewFullyConnectedLayer(arch[i], arch[i+1], actF))
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
func (n *NeuralNetwork) AverageLoss(samples []TrainingSample) float64 {
	sum := 0.0
	for _, s := range samples {
		actual := n.Forward(s.In)
		for i := 0; i < actual.Rows(); i++ {
			sum += math.Pow(s.Out.At(i)-actual.At(i), 2)
		}
	}
	return sum / float64(len(samples))
}

func (n *NeuralNetwork) BackpropStep(samples []TrainingSample, learningRate float64) {
	if learningRate <= 0 {
		return
	}

	for _, l := range n.Layers {
		l.ClearGradients()
	}

	// TODO add stochastic gradient descent, also add concurrency for each sample in the same pass
	for _, s := range samples {

		a := n.Forward(s.In)
		// Initial gradient for output layer is
		// dL/da_k = 2*(a_k - y_k) for MSE loss
		a_grad := t.ScalarMult(t.Sub(a, s.Out), 2)

		for l := len(n.Layers) - 1; l >= 0; l-- {
			a_grad = n.Layers[l].Backward(a_grad)
		}
	}

	for _, l := range n.Layers {
		l.UpdateParams(len(samples), learningRate)
	}
}

func (n *NeuralNetwork) Train(data []TrainingSample, epochs int, learningRate float64)  {
	currLoss := n.AverageLoss(data)

	for i := 0; i < epochs; i++ {
		n.BackpropStep(data, learningRate)

		newLoss := n.AverageLoss(data)
		if newLoss > currLoss {
			learningRate *= 0.95 // Reduce learning rate when loss increases (we "skipped" over the minimum)
		}
		currLoss = newLoss

		if LOG_PROGRESS && (epochs < 10 || i%(epochs/10) == 0) {
			fmt.Printf("iter:%7d - Loss: %7.5f\n", i, currLoss)
		}

	}
}
