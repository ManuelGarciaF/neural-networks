package nn

import (
	"math"
	"math/rand"

	"github.com/ManuelGarciaF/neural-networks/assert"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type FullyConnectedLayer struct {
	Weights *t.Tensor // MxN array
	Biases  *t.Tensor // N length vector
	actF    ActivationFunction
}

var _ Layer = &FullyConnectedLayer{}

type FullyConnectedLayerGradient struct {
	Weights *t.Tensor
	Biases  *t.Tensor
}

var _ LayerGrad = FullyConnectedLayerGradient{}

func (FullyConnectedLayerGradient) layerGrad() {}

type FullyConnectedLayerState struct {
	Input *t.Tensor
	Z     *t.Tensor // Z is the partial activation = Wx + b, stored for backpropagation.
}

var _ LayerState = FullyConnectedLayerState{}

func (FullyConnectedLayerState) layerState() {}

func NewFullyConnectedLayer(inSize, outSize int, actF ActivationFunction) *FullyConnectedLayer {
	l := &FullyConnectedLayer{
		Weights: t.New(outSize, inSize),
		Biases:  t.New(outSize, 1),
		actF:    actF,
	}

	// Initialize weights and biases
	l.initializeXavier(inSize, outSize)

	return l
}

func (l *FullyConnectedLayer) initializeXavier(in, out int) {
	limit := math.Sqrt(6.0 / float64(in+out))

	for i := range l.Weights.Data {
		l.Weights.Data[i] = (rand.Float64() - 0.5) * 2.0 * limit // Random number between [-limit, limit]
	}
	for i := range l.Biases.Data {
		l.Biases.Data[i] = 0.0
	}
}

func (l *FullyConnectedLayer) Forward(in *t.Tensor) (*t.Tensor, LayerState) {
	assert.Equal(in.Cols(), 1, "Input must be a column vector")
	assert.Equal(l.Weights.Cols(), in.Rows(), "Input must be the right size")

	z := t.Add(t.MatMul(l.Weights, in), l.Biases)

	// Cached values
	state := FullyConnectedLayerState{
		Input: in,
		Z:     z,
	}

	return t.Apply(z, l.actF.Apply), state
}

func (l *FullyConnectedLayer) ComputeGradients(s LayerState, nextLayerGrad *t.Tensor) (LayerGrad, *t.Tensor) {
	/* The formulas for each element of the gradient are:

	   dL/da^(prev)_k = Sum_j(dL/da_j * actF'(z_j) * w_jk)
	   dL/dw_jk       =       dL/da_j * actF'(z_j) * input_k
	   dL/db_j        =       dL/da_j * actF'(z_j)

	   We call this common factor (turned into a column vector)

	   delta = dL/da_j * actF'(z_j)

	   With it we can calculate the gradients using matrix operations.
	*/
	state, ok := s.(FullyConnectedLayerState)
	assert.True(ok, "State must match layer type")

	actFDerivatives := t.Apply(state.Z, l.actF.Derivative)
	delta := t.ElementMult(nextLayerGrad, actFDerivatives)

	// Clip delta to avoid infinite gradients.
	MAX_NORM := 1.0
	norm := delta.ColVectorNorm2()
	if norm > MAX_NORM {
		delta.ScaleInPlace(MAX_NORM / norm)
	}

	parameterGrad := FullyConnectedLayerGradient{
		// Weight gradient turns out to be just delta * input^T.
		Weights: t.MatMul(delta, t.MatTranspose(state.Input)),
		// Similarly, the bias gradient is just delta.
		Biases: delta,
	}
	// Finally, the gradient of the loss respecting the previous layer's output.
	prevLayerGrad := t.MatMul(t.MatTranspose(l.Weights), delta)

	// Check for NaN and inf
	assert.False(parameterGrad.Weights.Any(math.IsNaN), "NaN in weight gradients")
	assert.False(parameterGrad.Biases.Any(math.IsNaN), "NaN in bias gradients")
	assert.False(parameterGrad.Weights.Any(isInf), "Inf in weight gradients")
	assert.False(parameterGrad.Biases.Any(isInf), "Inf in bias gradients")

	return parameterGrad, prevLayerGrad
}

func (l *FullyConnectedLayer) UpdateParams(grads []LayerGrad, learningRate float64) {
	assert.GreaterThan(learningRate, 0, "Must be positive")

	// Average gradients over training examples
	wGradAvg := t.New(l.Weights.Shape...) // Empty tensors with the correct shape
	bGradAvg := t.New(l.Biases.Shape...)
	for _, individualGrad := range grads {
		individualGrad, ok := individualGrad.(FullyConnectedLayerGradient)
		assert.True(ok, "Gradient must match layer type")

		wGradAvg.AddInPlace(individualGrad.Weights)
		bGradAvg.AddInPlace(individualGrad.Biases)
	}
	wGradAvg.ScaleInPlace(1.0 / float64(len(grads)))
	bGradAvg.ScaleInPlace(1.0 / float64(len(grads)))

	// Modify parameters according to gradients and learning rate
	l.Weights.SubInPlace(t.ScalarMult(wGradAvg, learningRate))
	l.Biases.SubInPlace(t.ScalarMult(bGradAvg, learningRate))
}

func (l *FullyConnectedLayer) UpdateParamsMomentum(grads []LayerGrad, learningRate float64, prevGrads []LayerGrad, momentumConstant float64) {
	panic("not implemented")
}

func isInf(v float64) bool {
	return math.IsInf(v, 0)
}
