package nn

import (
	"testing"

	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

func BenchmarkXor(b *testing.B) {
	data := []Sample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(0)},
	}

	for b.Loop() {
		n := NewMLP([]int{2, 2, 1}, Sigmoid{}, true, 1.0)
		n.TrainSingleThreaded(data, 1000, 1.0, 1e-4, 0)
	}
}
