package nn

import (
	"math/rand/v2"

	"github.com/ManuelGarciaF/neural-networks/assert"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type FullyConnectedLayer struct {
	Weights *t.Tensor // MxN array
	Bias    *t.Tensor // N length vector
}

var _ Layer = &FullyConnectedLayer{}

func NewFullyConnectedLayer(inSize, outSize int) *FullyConnectedLayer {
	l := &FullyConnectedLayer{
		Weights: t.New(outSize, inSize),
		Bias:    t.New(outSize, 1),
	}
	l.initialize()
	return l
}

func (l *FullyConnectedLayer) initialize() {
	for i := range l.Weights.Data {
		l.Weights.Data[i] = (rand.Float64() - 0.5) * 4 // TODO may need to do something extra here
	}
	for i := range l.Bias.Data {
		l.Bias.Data[i] = (rand.Float64() - 0.5) * 4 // TODO may need to do something extra here
	}
}

func (l *FullyConnectedLayer) Forward(in *t.Tensor) *t.Tensor {
	assert.Equal(in.Dim(1), 1, "Input must be a column vector")
	assert.Equal(l.Weights.Dim(1), in.Dim(0), "Input must be the right size")

	return t.Add(t.MatMul(l.Weights, in), l.Bias)
}
