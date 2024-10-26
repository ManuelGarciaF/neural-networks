package nn

import (
	"math"
	"math/rand"

	"github.com/ManuelGarciaF/neural-networks/assert"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type FullyConnectedLayer struct {
	Weights *t.Tensor // MxN array
	Bias    *t.Tensor // N length vector
	actF    ActivationFunction

	lastInput *t.Tensor // Stored for backpropagation.
	lastZ     *t.Tensor // Z is the partial activation = Wx + b, stored for backpropagation.

	wGradAcum *t.Tensor // Accumulated values of the gradients during a backpropagation batch
	bGradAcum *t.Tensor
}

var _ Layer = &FullyConnectedLayer{}

func NewFullyConnectedLayer(inSize, outSize int, actF ActivationFunction) *FullyConnectedLayer {
	l := &FullyConnectedLayer{
		Weights:   t.New(outSize, inSize),
		Bias:      t.New(outSize, 1),
		actF:      actF,
		lastInput: nil,
		lastZ:     nil,
		wGradAcum: nil,
		bGradAcum: nil,
	}

	// Best for sigmoid
	l.initializeXavier(inSize, outSize)

	return l
}

func (l *FullyConnectedLayer) initializeXavier(in, out int) {
	limit := math.Sqrt(6.0 / float64(in+out))

	for i := range l.Weights.Data {
		l.Weights.Data[i] = (rand.Float64()*2.0 - 0.5) * limit
	}
	for i := range l.Bias.Data {
		l.Bias.Data[i] = 0.0
	}
}

func (l *FullyConnectedLayer) Forward(in *t.Tensor) *t.Tensor {
	assert.Equal(in.Cols(), 1, "Input must be a column vector")
	assert.Equal(l.Weights.Cols(), in.Rows(), "Input must be the right size")

	l.lastInput = in

	z := t.Add(t.MatMul(l.Weights, in), l.Bias)
	l.lastZ = z

	return t.Apply(z, l.actF.Apply)
}

// We calculate the gradients for each sample and accumulate them.
func (l *FullyConnectedLayer) Backward(nextLayerGrad *t.Tensor) *t.Tensor {
	// Next layer's gradient multiplied element wise with the activation function's derivative,
	// allows to calculate gradients with matrix operations.
	// dL/da^(prev)_k = Sum_j(dL/da_j * actF'(z_j) * w_jk) = delta * w^T
	// dL/dw_jk = dL/da_j * actF'(z_j) * input_k
	// dL/db_j = dL/da_j * actF'(z_j)
	actFDerivatives := t.Apply(l.lastZ, l.actF.Derivative)
	delta := t.ElementMult(nextLayerGrad, actFDerivatives)

	// Weight gradient turns out to be just delta * input^T.
	wgrad := t.MatMul(delta, t.MatTranspose(l.lastInput))
	l.wGradAcum.AddInPlace(wgrad)
	// Similarly, bias gradient is just delta.
	l.bGradAcum.AddInPlace(delta)

	// Finally, the gradient of the loss respecting this layer's input:
	return t.MatMul(t.MatTranspose(l.Weights), delta)
}

func (l *FullyConnectedLayer) UpdateParams(numberSamples int, learningRate float64) {
	assert.GreaterThan(numberSamples, 0, "Must be positive")
	assert.GreaterThan(learningRate, 0, "Must be positive")

	// Average gradient over training examples
	wGradAvg := t.ScalarMult(l.wGradAcum, 1.0/float64(numberSamples))
	bGradAvg := t.ScalarMult(l.bGradAcum, 1.0/float64(numberSamples))

	// Modify parameters according to gradients and learning rate
	l.Weights.SubInPlace(t.ScalarMult(wGradAvg, learningRate))
	l.Bias.SubInPlace(t.ScalarMult(bGradAvg, learningRate))
}

func (l *FullyConnectedLayer) ClearGradients() {
	l.wGradAcum = t.New(l.Weights.Shape...)
	l.bGradAcum = t.New(l.Bias.Shape...)
}
