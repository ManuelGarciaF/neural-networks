package nn

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type NeuralNetwork struct {
	Layers                []Layer
	GradientClippingLimit float64
}

type Sample struct{ In, Out *t.Tensor } // Both column vectors

// Arch is a list of layer sizes, including input and output
func NewMLP(
	arch []int,
	actF ActivationFunction,
	outputNeedsActivation bool,
	gradientClippingLimit float64,
) *NeuralNetwork {
	layers := make([]Layer, 0, len(arch)-1)

	for i := 0; i < len(arch)-2; i++ {
		layers = append(layers, NewFullyConnectedLayer(arch[i], arch[i+1], actF))
	}

	var outputActF ActivationFunction = NoActF{}
	if outputNeedsActivation {
		outputActF = actF
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

func (n *NeuralNetwork) TrainSingleThreaded(data []Sample, epochs int, learningRate float64, verbose bool) {
	for i := 0; i < epochs; i++ {
		n.BackpropStepSingleThreaded(data, learningRate)
		if verbose && (epochs <= 50 || i%(epochs/50) == 0) {
			fmt.Printf("iter:%7d - Loss: %7.5f\n", i, n.AverageLoss(data))
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

func (n *NeuralNetwork) TrainConcurrent(
	samples []Sample,
	epochs int,
	learningRate float64,
	batchSize int,
	maxGoroutines int,
	verbose bool,
) {
	if maxGoroutines == 0 {
		maxGoroutines = runtime.NumCPU()
	}

	// Shuffle data to avoid biases
	rand.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})

	for epoch := 0; epoch < epochs; epoch++ {
		// gradientChan := make(chan []LayerGrad, maxGoroutines)
		var wg sync.WaitGroup

		// Process in batches
		for batchStart := 0; batchStart < len(samples); batchStart += batchSize {
			batchEnd := min(batchStart+batchSize, len(samples))
			batchSamples := samples[batchStart:batchEnd]

			wg.Add(1)
			go func(batch []Sample) {
				defer wg.Done()

				// Accumulate gradients of the batch.
				batchGrads := make([]LayerGrad, len(n.Layers))
				for _, sample := range batch {
					activation, states := n.Forward(sample.In)
					lossGradient := computeLossGradient(activation, sample.Out)
					gradients := n.Backward(states, lossGradient)

					for layer := range batchGrads {
						if batchGrads[layer] == nil {
							batchGrads[layer] = gradients[layer]
						} else {
							batchGrads[layer].Add(gradients[layer])
						}
					}
				}

			}(batchSamples)

		}
	}
}

// Backward returns the list of gradients for each successive layer
func (n *NeuralNetwork) Backward(states []LayerState, lossGradient *t.Tensor) []LayerGrad {
	gradientList := make([]LayerGrad, len(n.Layers))

	actGrad := lossGradient // Gradient respect the last activation
	for layer := len(n.Layers) - 1; layer >= 0; layer-- {
		gradientList[layer], actGrad = n.Layers[layer].ComputeGradients(
			states[layer],
			actGrad,
			n.GradientClippingLimit,
		)
	}

	return gradientList
}

func computeLossGradient(output, expectedOutput *t.Tensor) *t.Tensor {
	// Initial gradient for output layer is
	// dL/da_k = 2*(a_k - y_k) for MSE loss
	return t.ScalarMult(t.Sub(output, expectedOutput), 2)
}
