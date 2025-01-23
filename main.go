package main

import (
	"fmt"
	"math/rand"
	"strconv"

	"github.com/ManuelGarciaF/neural-networks/nn"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

func main() {
	// Disable asserts for speed
	// assert.ASSERT_ENABLE = false

	adder()
	and()
	xor()
}

func adder() {
	fmt.Println("Adder:")
	data := []nn.Sample{
		{In: t.ColumnVector(1, 1), Out: t.Scalar(2)},
		{In: t.ColumnVector(2, 5), Out: t.Scalar(7)},
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(-1, 1), Out: t.Scalar(0)},
	}
	// n := naiveTraining([]int{2, 1}, data, 0.01, 20000)
	n := nn.NewMLP([]int{2, 1}, nn.Sigmoid{}, false, 1.0)
	n.Train(data, 10*1000, 0.1, false)
	testNN(n, data)
}

func and() {
	fmt.Println("AND:")
	data := []nn.Sample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(1)},
	}
	// n := naiveTraining([]int{2, 1}, data, 0.01, 200000)
	n := nn.NewMLP([]int{2, 1}, nn.Sigmoid{}, true, 1.0)
	n.Train(data, 10*1000, 1.0, false)

	testNN(n, data)
}

func xor() {
	fmt.Println("XOR:")
	data := []nn.Sample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(0)},
	}

	n := nn.NewMLP([]int{2, 2, 1}, nn.Sigmoid{}, true, 1.0)
	n.Train(data, 100*1000, 1.0, false)
	testNN(n, data)
}

func naiveTraining(arch []int, outputNeedsActivation bool, data []nn.Sample, learningRate float64, epochs int) *nn.NeuralNetwork {
	n := nn.NewMLP(arch, nn.Sigmoid{}, outputNeedsActivation, 1.0)

	fmt.Println("Initial parameters:")
	printNN(n)

	// Naive training
	loss := n.AverageLoss(data)
	for i := 0; i < epochs; i++ {
		// Juggle the parameters a bit
		n2 := nn.NewMLP(arch, nn.Sigmoid{}, outputNeedsActivation, 1.0) // New network to store updated values
		for lnum, l := range n.Layers {
			l, ok := l.(*nn.FullyConnectedLayer)
			l2, _ := n2.Layers[lnum].(*nn.FullyConnectedLayer)
			if !ok { // All layers are fullyConnected for now
				continue
			}
			for i := range l.Weights.Data {
				l2.Weights.Data[i] += (rand.Float64() - 0.5) * learningRate
			}
			for i := range l.Biases.Data {
				l2.Biases.Data[i] += (rand.Float64() - 0.5) * learningRate
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
	fmt.Println("Final loss:", n.AverageLoss(data))

	fmt.Println("Final weights:")
	printNN(n)

	// Check results
	fmt.Println()
	fmt.Println("Sample results:")
	for _, sample := range data {
		out, _ := n.Forward(sample.In)
		fmt.Printf("\tin: %+v, out: %v, expected %v\n", sample.In.Data, out.Data, sample.Out.Data)
	}
}

func printNN(n *nn.NeuralNetwork) {
	for i, l := range n.Layers {
		l, ok := l.(*nn.FullyConnectedLayer)
		if !ok {
			continue
		}
		l.Weights.PrintMatrix(strconv.Itoa(i) + ") Weights")
		l.Biases.PrintMatrix(strconv.Itoa(i) + ") Bias")
	}
}
