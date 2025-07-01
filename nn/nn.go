package nn

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type NeuralNetwork struct {
	Layers                []Layer
	GradientClippingLimit float64
}

type Sample struct{ In, Out *t.Tensor } // Both column vectors

// NewMLP (Multi-Layer Perceptron) Creates a network of fully connected layers.
// Arch is a list of layer sizes, including input and output
func NewMLP(
	arch []int,
	actF ActivationFunction,
	outputActF ActivationFunction,
	gradientClippingLimit float64,
) *NeuralNetwork {
	layers := make([]Layer, 0, len(arch)-1)

	for i := 0; i < len(arch)-2; i++ {
		layers = append(layers, NewFullyConnectedLayer(arch[i], arch[i+1], actF))
	}

	layers = append(layers, NewFullyConnectedLayer(arch[len(arch)-2], arch[len(arch)-1], outputActF))

	return &NeuralNetwork{Layers: layers, GradientClippingLimit: gradientClippingLimit}
}

func (n *NeuralNetwork) Forward(input *t.Tensor) (*t.Tensor, []LayerState) {
	activations := make([]*t.Tensor, len(n.Layers)+1)
	states := make([]LayerState, len(n.Layers))
	activations[0] = input
	for i, l := range n.Layers {
		activations[i+1], states[i] = l.Forward(activations[i])
	}
	return activations[len(activations)-1], states
}

func (n *NeuralNetwork) AverageLoss(samples []Sample) float64 {
	// Mean squared error
	sum := 0.0
	for _, s := range samples {
		actual, _ := n.Forward(s.In)
		for i := 0; i < actual.Rows(); i++ {
			sum += math.Pow(s.Out.At(i)-actual.At(i), 2)
		}
	}
	return sum / float64(len(samples))
}

func (n *NeuralNetwork) TrainSingleThreaded(data []Sample, epochs int, startingLearningRate float64, decay float64, verboseSteps int) {
	for i := 0; i < epochs; i++ {
		// Decay the learningRate
		learningRate := startingLearningRate / (1.0 + decay*float64(i))

		n.BackpropStepSingleThreaded(data, learningRate)

		if verboseSteps > 0 && i%(epochs/verboseSteps) == 0 {
			fmt.Printf("iter:%7d - lr:%3d - Loss: %7.5f\n", i, learningRate, n.AverageLoss(data))
		}
	}

}

func (n *NeuralNetwork) BackpropStepSingleThreaded(samples []Sample, learningRate float64) {
	if learningRate <= 0 {
		return
	}
	// A gradient per layer
	gradientAccums := make([]LayerGrad, len(n.Layers))

	for _, sample := range samples {
		activation, states := n.Forward(sample.In)

		lossGrad := computeLossGradient(activation, sample.Out)

		grads := n.Backward(states, lossGrad)

		for layer := range gradientAccums {
			if gradientAccums[layer] == nil {
				gradientAccums[layer] = grads[layer]
			} else {
				gradientAccums[layer].Add(grads[layer])
			}
		}
	}
	// Apply updates
	for i, layer := range n.Layers {
		// Normalize gradient
		gradientAccums[i].Scale(1.0 / float64(len(samples)))

		layer.UpdateParams(gradientAccums[i], learningRate)
	}
}

type NetworkGrad = []LayerGrad // Makes it easier to think about

// Train the network using mini-batch SGD.
// Remember that using the verbose option is really expensive since it calculates the global loss
func (n *NeuralNetwork) TrainConcurrent(
	samples []Sample,
	epochs int,
	startingLearningRate float64,
	decay float64,
	batchSize int,
	workers int,
	verboseSteps int,
) {
	if workers == 0 {
		workers = runtime.NumCPU()
	}
	if batchSize == 0 {
		batchSize = len(samples)
	}

	// Create workers
	workChan := make(chan []Sample, workers)    // A list of samples per worker
	gradChan := make(chan NetworkGrad, workers) // Each worker produces gradients for the whole network
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			backpropWorker(n, workChan, gradChan)
		}()
	}

	// Number of samples to send to each worker
	workSize := ceilingDiv(batchSize, workers)
	numSubBatches := ceilingDiv(batchSize, workSize)

	for epoch := 0; epoch < epochs; epoch++ {
		// Decay the learning rate.
		learningRate := startingLearningRate / (1.0 + decay*float64(epoch))

		batch := randomSubset(samples, batchSize)

		// Send part of the samples to each worker
		for start := 0; start < batchSize; start += workSize {
			end := min(start+workSize, batchSize)
			workChan <- batch[start:end] // Send part of the batch off to a worker
		}

		// Collect results
		networkGrad := make([]LayerGrad, len(n.Layers)) // Network grad for accumulating results.
		for i := 0; i < numSubBatches; i++ {
			subBatchNetworkGrad := <-gradChan
			for layer := range subBatchNetworkGrad { // Add layer by layer
				if networkGrad[layer] == nil {
					networkGrad[layer] = subBatchNetworkGrad[layer]
				} else {
					networkGrad[layer].Add(subBatchNetworkGrad[layer])
				}
			}
		}

		// Apply updates
		for i, layer := range n.Layers {
			// Normalize gradient
			networkGrad[i].Scale(1.0 / float64(batchSize))

			layer.UpdateParams(networkGrad[i], learningRate)
		}

		if verboseSteps > 0 && epoch%(epochs/verboseSteps) == 0 {
			fmt.Printf("iter:%7d - lr:%1.4f - Batch Loss: %7.5f\n", epoch, learningRate, n.AverageLoss(batch))
		}
	}

	// Wait until workers finished
	close(workChan)
	wg.Wait()
}

// Backward returns the list of gradients for each successive layer.
func (n *NeuralNetwork) Backward(states []LayerState, lossGradient *t.Tensor) []LayerGrad {
	gradientList := make([]LayerGrad, len(n.Layers))

	actGrad := lossGradient // Gradient respect the last activation.
	for layer := len(n.Layers) - 1; layer >= 0; layer-- {
		gradientList[layer], actGrad = n.Layers[layer].ComputeGradients(
			states[layer],
			actGrad,
			n.GradientClippingLimit,
		)
	}

	return gradientList
}

// Shouldn't need a mutex on the NN since there should never be pending work while updating parameters.
func backpropWorker(n *NeuralNetwork, workChan <-chan []Sample, gradChan chan<- NetworkGrad) {
	// Get our work
	for samples := range workChan {
		// Process batch of samples
		batchGrads := make([]LayerGrad, len(n.Layers))
		for _, sample := range samples {
			// Forward
			activation, states := n.Forward(sample.In)

			// Backward
			lossGradient := computeLossGradient(activation, sample.Out)
			gradients := n.Backward(states, lossGradient)

			// Accummulate partial results
			for layer := range batchGrads {
				if batchGrads[layer] == nil {
					batchGrads[layer] = gradients[layer]
				} else {
					batchGrads[layer].Add(gradients[layer])
				}
			}
		}

		// Send results back
		gradChan <- batchGrads
	}
}

func computeLossGradient(output, expectedOutput *t.Tensor) *t.Tensor {
	// Initial gradient for output layer is
	// dL/da_k = 2*(a_k - y_k) for MSE loss
	return t.ScalarMult(t.Sub(output, expectedOutput), 2)
}
