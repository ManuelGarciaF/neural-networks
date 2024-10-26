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

	/* FIXME have to remove activations from the layer to avoid race conditions when executing concurrently
		// Each layer will have one gradient per sample. It's easier to have a channel per layer.
		gradientChannels := make([]chan LayerGrad, len(n.Layers))
		for i := range gradientChannels {
			gradientChannels[i] = make(chan LayerGrad, len(samples))
		}

		var wg sync.WaitGroup
		// TODO stochastic gradient descent (divide in batches)
		for _, sample := range samples {
			// Create goroutines for each sample
			wg.Add(1)
			go func() {
				defer wg.Done()

				a := n.Forward(sample.In) // Forward to cache activations
				// Initial gradient for output layer is
				// dL/da_k = 2*(a_k - y_k) for MSE loss
				aGrad := t.ScalarMult(t.Sub(a, sample.Out), 2)

				for l := len(n.Layers) - 1; l >= 0; l-- {
					// Get the gradients from layer l
					var paramsGrad LayerGrad
					paramsGrad, aGrad = n.Layers[l].Backward(aGrad)

					// Send the parameters gradient to its layer's channel
					gradientChannels[l] <- paramsGrad
				}
			}()
		}
		wg.Wait()
		// Close channels
		for _, c := range gradientChannels {
			close(c)
		}

		// Update params of each layer
		for layerIndex, layer := range n.Layers {
			// Collect gradients for this layer
			grads := make([]LayerGrad, 0, len(samples))
			for g := range gradientChannels[layerIndex] {
				grads = append(grads, g)
			}

			layer.UpdateParams(grads, learningRate)
		}
	*/

	// Non concurrent version
	gradientLists := make([][]LayerGrad, len(n.Layers))
	for l := range gradientLists {
		gradientLists[l] = make([]LayerGrad, 0, len(samples))
	}
	for _, sample := range samples {
		a := n.Forward(sample.In) // Forward to cache activations
		// Initial gradient for output layer is
		// dL/da_k = 2*(a_k - y_k) for MSE loss
		aGrad := t.ScalarMult(t.Sub(a, sample.Out), 2)

		for l := len(n.Layers) - 1; l >= 0; l-- {
			// Get the gradients from layer l
			var paramsGrad LayerGrad
			paramsGrad, aGrad = n.Layers[l].Backward(aGrad)
			gradientLists[l] = append(gradientLists[l], paramsGrad)
		}
	}
	// Apply updates
	for i, layer := range n.Layers {
		layer.UpdateParams(gradientLists[i], learningRate)
	}
}

func (n *NeuralNetwork) Train(data []TrainingSample, epochs int, learningRate float64) {
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
