package nn

import "math"

type ActivationFunction interface {
	Apply(v float64) float64
	Derivative(v float64) float64
}

type ReLU struct{}

var _ ActivationFunction = ReLU{}

func (ReLU) Apply(v float64) float64 {
	return max(0, v)
}

func (ReLU) Derivative(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return 1
}

type LeakyReLU struct {
	Alpha float64
}

var _ ActivationFunction = LeakyReLU{}

func (l LeakyReLU) Apply(v float64) float64 {
	return max(l.Alpha*v, v)
}

func (l LeakyReLU) Derivative(v float64) float64 {
	if v <= 0 {
		return l.Alpha
	}
	return 1
}

type Sigmoid struct{}

var _ ActivationFunction = Sigmoid{}

func (Sigmoid) Apply(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func (Sigmoid) Derivative(v float64) float64 {
	sx := 1.0 / (1.0 + math.Exp(-v))
	return sx * (1 - sx)
}
