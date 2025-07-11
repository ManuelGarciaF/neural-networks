package nn

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Poor man's enums with methods. Can't believe I miss java.

type activationFunctionType byte

const (
	RELU activationFunctionType = iota
	SIGMOID
	NO_ACT_F
)

type ActivationFunction interface {
	Apply(v float64) float64
	Derivative(v float64) float64
	actFType() activationFunctionType
}

func loadActF(r io.Reader) (ActivationFunction, error) {
	// Read type
	var t activationFunctionType
	err := binary.Read(r, binary.LittleEndian, &t)
	if err != nil {
		return nil, err
	}

	switch activationFunctionType(t) {
	case RELU:
		return ReLU{}, nil
	case SIGMOID:
		return Sigmoid{}, nil
	case NO_ACT_F:
		return NoActF{}, nil
	default:
		return nil, errors.New(
			fmt.Sprint("Invalid activation function type found: ", t),
		)
	}

}

type ReLU struct{}

var _ ActivationFunction = ReLU{}

func (ReLU) actFType() activationFunctionType { return RELU }

func (ReLU) Apply(v float64) float64 {
	return max(0, v)
}

func (ReLU) Derivative(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return 1
}

type Sigmoid struct{}

var _ ActivationFunction = Sigmoid{}

func (Sigmoid) actFType() activationFunctionType { return SIGMOID }

func (Sigmoid) Apply(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

func (s Sigmoid) Derivative(v float64) float64 {
	sv := s.Apply(v)
	return sv * (1 - sv)
}

// NoActF Used for the last layer
type NoActF struct{}

var _ ActivationFunction = NoActF{}

func (NoActF) actFType() activationFunctionType { return NO_ACT_F }

func (NoActF) Apply(v float64) float64 {
	return v
}

func (NoActF) Derivative(v float64) float64 {
	return 1
}
