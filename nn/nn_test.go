package nn_test

import (
	"math"
	"testing"

	"github.com/ManuelGarciaF/neural-networks/nn"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

func TestNetworks(te *testing.T) {
	tests := []struct {
		name      string
		arch      []int
		outputAct bool
		data      []nn.Sample
		epochs    int
		learnRate float64
		threshold float64
	}{
		{
			name:      "AND",
			arch:      []int{2, 1},
			outputAct: true,
			learnRate: 1.0,
			epochs:    1000,
			threshold: 0.1,
			data: []nn.Sample{
				{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
				{In: t.ColumnVector(0, 1), Out: t.Scalar(0)},
				{In: t.ColumnVector(1, 0), Out: t.Scalar(0)},
				{In: t.ColumnVector(1, 1), Out: t.Scalar(1)},
			},
		},
		{
			name:      "XOR",
			arch:      []int{2, 2, 1},
			outputAct: true,
			learnRate: 1.0,
			epochs:    30000,
			threshold: 0.1,
			data: []nn.Sample{
				{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
				{In: t.ColumnVector(0, 1), Out: t.Scalar(1)},
				{In: t.ColumnVector(1, 0), Out: t.Scalar(1)},
				{In: t.ColumnVector(1, 1), Out: t.Scalar(0)},
			},
		},
		{
			name:      "Adder",
			arch:      []int{2, 1},
			outputAct: false,
			learnRate: 0.1,
			epochs:    10000,
			threshold: 0.5,
			data: []nn.Sample{
				{In: t.ColumnVector(1, 1), Out: t.Scalar(2)},
				{In: t.ColumnVector(2, 5), Out: t.Scalar(7)},
				{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
				{In: t.ColumnVector(-1, 1), Out: t.Scalar(0)},
			},
		},
	}

	for _, tt := range tests {
		te.Run(tt.name, func(te *testing.T) {
			network := nn.NewMLP(tt.arch, nn.Sigmoid{}, tt.outputAct, 1.0)
			network.TrainSingleThreaded(tt.data, tt.epochs, tt.learnRate, false)

			finalLoss := network.AverageLoss(tt.data)
			if finalLoss > tt.threshold {
				te.Errorf("Final loss %f exceeds threshold %f", finalLoss, tt.threshold)
			}

			// Test predictions
			for _, sample := range tt.data {
				out, _ := network.Forward(sample.In)
				for i, v := range out.Data {
					diff := math.Abs(v - sample.Out.Data[i])
					if diff > tt.threshold {
						te.Errorf("Prediction error too high: got %f, want %f", v, sample.Out.Data[i])
					}
				}
			}
		})
	}
}
