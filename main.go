package main

import (
	"fmt"
	"math/rand"
	"strconv"

	"github.com/ManuelGarciaF/neural-networks/nn"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

func main() {
	// adder()
	// and()
	xor()
}

func adder() {
	fmt.Println("Adder:")
	data := []nn.TrainingSample{
		{In: t.ColumnVector(1, 1), Out: t.Scalar(2)},
		{In: t.ColumnVector(2, 5), Out: t.Scalar(7)},
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(-1, 1), Out: t.Scalar(0)},
	}
	n := naiveTraining(
		[]int{2, 1},
		data,
		1e-6,
		20000,
	)
	testNN(n, data)
}

func and() {
	fmt.Println("AND:")
	data := []nn.TrainingSample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(1)},
	}
	n := naiveTraining(
		[]int{2, 1},
		data,
		1e-6,
		20000,
	)
	testNN(n, data)
}

func xor() {
	fmt.Println("XOR:")
	data := []nn.TrainingSample{
		{In: t.ColumnVector(0, 0), Out: t.Scalar(0)},
		{In: t.ColumnVector(0, 1), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 0), Out: t.Scalar(1)},
		{In: t.ColumnVector(1, 1), Out: t.Scalar(0)},
	}
	n := naiveTraining([]int{2, 2, 1}, data, 1e-6, 50000)
	testNN(n, data)
}

func naiveTraining(arch []int, data []nn.TrainingSample, epsilon float64, iterations int) *nn.NeuralNetwork {
	n := nn.NewNeuralNetwork(arch)

	fmt.Println("Initial parameters:")
	printNN(n)

	// Naive training
	for i := 0; i < iterations; i++ {
		loss := n.Loss(data)

		if i%(iterations/10) == 0 {
			fmt.Printf("iter: %d - Loss: %v\n", i, loss)
		}

		// Juggle the parameters a bit
		n2 := nn.NewNeuralNetwork(arch) // New network to store updated values
		for lnum, l := range n.Layers {
			l, ok := l.(*nn.FullyConnectedLayer)
			l2, _ := n2.Layers[lnum].(*nn.FullyConnectedLayer)
			if !ok {
				continue
			}
			for i := range l.Weights.Data {
				l2.Weights.Data[i] += (rand.Float64() - 0.5) * epsilon
			}
			for i := range l.Bias.Data {
				l2.Bias.Data[i] += (rand.Float64() - 0.5) * epsilon
			}
		}
		newLoss := n2.Loss(data)
		if newLoss < loss {
			n = n2
		}
	}

	return n
}

func testNN(n *nn.NeuralNetwork, data []nn.TrainingSample) {
	fmt.Println("------------------------------")
	fmt.Println("Final loss:", n.Loss(data))

	fmt.Println("Final weights:")
	printNN(n)

	// Check results
	fmt.Println()
	fmt.Println("Sample results:")
	for _, sample := range data {
		out := n.Forward(sample.In)
		fmt.Printf("\tin: %+v, out: %+v, expected %+v\n", sample.In.Data, out.Data, sample.Out.Data)
	}
}

func printNN(n *nn.NeuralNetwork) {
	for i, l := range n.Layers {
		l, ok := l.(*nn.FullyConnectedLayer)
		if !ok {
			continue
		}
		l.Weights.PrintMatrix("Weights(" + strconv.Itoa(i/2) + "):")
		l.Bias.PrintMatrix("Bias(" + strconv.Itoa(i/2) + "):")
	}
}
