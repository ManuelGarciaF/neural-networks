package layer

import (
	"math/rand"

	"github.com/ManuelGarciaF/neural-networks/assert"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type FullyConnectedLayer struct {
	weights *t.Tensor // MxN array
	bias    *t.Tensor // N length vector
}

var _ Linear = &FullyConnectedLayer{}

func NewFullyConnectedLayer(outSize, inSize int) *FullyConnectedLayer {
	l := &FullyConnectedLayer{
		weights: t.New(outSize, inSize),
		bias:    t.New(outSize),
	}
	l.initialize()
	return l
}

func (l *FullyConnectedLayer) initialize() {
	for i := range l.weights.Data {
		l.weights.Data[i] = rand.Float64() // TODO may need to do something extra here
	}
	for i := range l.bias.Data {
		l.bias.Data[i] = rand.Float64() // TODO may need to do something extra here
	}
}

func (l *FullyConnectedLayer) Forward(in *t.Tensor) *t.Tensor {
	assert.Equal(in.Dim(1), 1, "Input must be a column vector")
	assert.Equal(l.weights.Dim(1), in.Dim(0), "Input must be the right size")

	return t.Add(t.MatMul(l.weights, in), l.bias)
}
