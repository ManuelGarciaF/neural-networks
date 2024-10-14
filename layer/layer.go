package layer

import (
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

type Layer interface {
	Forward(in *t.Tensor) *t.Tensor // May modify the in tensor
}
