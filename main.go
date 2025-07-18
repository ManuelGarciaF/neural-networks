package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/ManuelGarciaF/neural-networks/nn"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

func main() {
	verbose := false
	if len(os.Args) > 1 && os.Args[1] == "-v" {
		verbose = true
	}

	adder(verbose)
	and(verbose)
	xor(verbose)
}

func adder(verbose bool) {
	fmt.Println("--------------------------------------")
	fmt.Println("Adder:")
	data := []nn.Sample{
		{In: t.ColumnVector(1, 1), Out: t.Scalar(2)},
		{In: t.ColumnVector(2, 5), Out: t.Scalar(7)},
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(-1, 1), Out: t.Scalar(0)},
	}
	// n := NaiveTraining([]int{2, 1}, data, 0.01, 20000)
	n := nn.NewMLP([]int32{2, 1}, nn.Sigmoid{}, nn.NoActF{}, 1.0)
	n.TrainConcurrent(data, 2000, 1.0, 0.01, 0, 0, verbose)
	testNN(n, data)
}

func and(verbose bool) {
	fmt.Println("--------------------------------------")
	fmt.Println("AND:")
	data := []nn.Sample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(1)},
	}
	// n := NaiveTraining([]int{2, 1}, data, 0.01, 200000)
	n := nn.NewMLP([]int32{2, 1}, nn.Sigmoid{}, nn.Sigmoid{}, 1.0)
	n.TrainConcurrent(data, 5000, 0.5, 1e-4, 0, 0, verbose)

	testNN(n, data)
}

func xor(verbose bool) {
	fmt.Println("--------------------------------------")
	fmt.Println("XOR:")
	data := []nn.Sample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(0)},
	}

	n := nn.NewMLP([]int32{2, 2, 1}, nn.Sigmoid{}, nn.NoActF{}, 1.0)
	n.TrainConcurrent(data, 5000, 0.5, 1e-4, 0, 0, verbose)
	testNN(n, data)
}

func NaiveTraining(arch []int32, data []nn.Sample, learningRate float64, epochs int) *nn.NeuralNetwork {
	n := nn.NewMLP(arch, nn.Sigmoid{}, nn.NoActF{}, 1.0)

	loss := n.AverageLoss(data)
	for i := range epochs {
		// Jiggle the parameters a bit
		n2 := nn.NewMLP(arch, nn.Sigmoid{}, nn.NoActF{}, 1.0) // New network to store updated values
		for lnum, l := range n.Layers {
			l, ok := l.(*nn.FullyConnectedLayer)
			l2, _ := n2.Layers[lnum].(*nn.FullyConnectedLayer)
			if !ok { // A MLP should only have fully connected layers
				panic("Unreachable")
			}
			for i, w0 := range l.Weights.Data {
				l2.Weights.Data[i] = w0 + (rand.Float64()-0.5)*learningRate
			}
			for i, b0 := range l.Biases.Data {
				l2.Biases.Data[i] = b0 + (rand.Float64()-0.5)*learningRate
			}
		}
		newLoss := n2.AverageLoss(data)
		if newLoss < loss {
			n = n2
			loss = newLoss
			fmt.Printf("iter:%7d - Loss: %7.5f\n", i, loss)
		}
	}

	return n
}

func testNN(n *nn.NeuralNetwork, data []nn.Sample) {
	fmt.Println("----------------------------")
	fmt.Printf("Final loss: %1.12f\n", n.AverageLoss(data))

	// fmt.Println("Final weights:")
	// PrintNN(n)

	// Check results
	fmt.Println()
	fmt.Println("Sample results:")
	for i, sample := range data {
		out, _ := n.Forward(sample.In)
		sample.In.PrintMatrix(fmt.Sprintf("  [%2d] In: ", i))
		out.PrintMatrix("       Out:")
		sample.Out.PrintMatrix("       Expected:")
	}
	fmt.Println()
}

func PrintNN(n *nn.NeuralNetwork) {
	for i, l := range n.Layers {
		l, ok := l.(*nn.FullyConnectedLayer)
		if !ok {
			continue
		}
		l.Weights.PrintMatrix(fmt.Sprintf("%d) Weights", i))
		l.Biases.PrintMatrix(fmt.Sprintf("%d) Bias", i))
	}
}
