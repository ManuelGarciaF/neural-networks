package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/ManuelGarciaF/neural-networks/assert"
	"github.com/ManuelGarciaF/neural-networks/nn"
	t "github.com/ManuelGarciaF/neural-networks/tensor"
)

const ImageSize = 28

// Conversion from number to the expected output
var LabelVectors = map[byte]*t.Tensor{
	0: t.ColumnVector(1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
	1: t.ColumnVector(0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
	2: t.ColumnVector(0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
	3: t.ColumnVector(0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
	4: t.ColumnVector(0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
	5: t.ColumnVector(0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
	6: t.ColumnVector(0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
	7: t.ColumnVector(0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
	8: t.ColumnVector(0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
	9: t.ColumnVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
}

func main() {
	// Example of using a MLP to learn the mnist digit recognition dataset
	// The dataset contains 60000 handwriten digits from 0 to 9 in 28*28 grayscale images

	// Downloaded from:
	// https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	// https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
	// https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
	// https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s [train|run]\n", os.Args[0])
		os.Exit(1)
	}
	switch os.Args[1] {
	case "t", "train":
		model := train()
		err := model.SaveToFile("mnist.nn")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error saving model: %s\n", err.Error())
		}
		run(model)

	case "r", "run":
		if len(os.Args) != 3 {
			fmt.Fprintf(os.Stderr, "Usage: %s run modelpath\n", os.Args[0])
			os.Exit(1)
		}
		path := os.Args[2]
		model, err := nn.LoadFromFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading model: %s\n", err.Error())
			os.Exit(2)
		}
		run(model)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		fmt.Fprintf(os.Stderr, "Usage: %s [train|run]\n", os.Args[0])
		os.Exit(1)
	}
}

func train() *nn.NeuralNetwork {
	// Build training data
	trainSamples := 50000
	trainImgs := readImgs("./train-images.idx3-ubyte", trainSamples)
	trainLabels := readLabels("./train-labels.idx1-ubyte", trainSamples)

	trainData := make([]nn.Sample, trainSamples)
	for i := range trainData {
		trainData[i] = nn.Sample{
			In:  t.WithData([]int32{ImageSize * ImageSize}, trainImgs[i]),
			Out: LabelVectors[trainLabels[i]],
		}
	}

	model := nn.NewMLP([]int32{
		ImageSize * ImageSize, // Input pixels
		256,
		256,
		10, // Outputs
	}, nn.Sigmoid{}, nn.Sigmoid{}, 1.0)

	fmt.Println("Starting Training")
	model.TrainConcurrent(trainData, 10, 0.25, 0.1, 32, 0, true)

	return model
}

func run(model *nn.NeuralNetwork) {
	testSamples := 500
	testImgs := readImgs("./t10k-images.idx3-ubyte", testSamples)
	testLabels := readLabels("./t10k-labels.idx1-ubyte", testSamples)

	testData := make([]nn.Sample, testSamples)
	for i := range testData {
		testData[i] = nn.Sample{
			In:  t.WithData([]int32{ImageSize * ImageSize}, testImgs[i]),
			Out: LabelVectors[testLabels[i]],
		}
	}

	fmt.Println("----------------------------")
	fmt.Println("Final loss:", model.AverageLoss(testData))

	// Calculate accuracy
	correctGuesses := 0
	for _, s := range testData {
		forward, _ := model.Forward(s.In)
		if maxIndex(forward.Data) == maxIndex(s.Out.Data) {
			correctGuesses++
		}
	}
	accuracy := (float64(correctGuesses) / float64(len(testData))) * 100
	fmt.Println("Accuracy: ", accuracy, "%")

	fmt.Println("Example outputs: ")

	test := randomSubset(testData, 4)
	for _, s := range test {
		fmt.Println("Digit: ")
		printDigit(s.In.Data)
		actual, _ := model.Forward(s.In)
		fmt.Println("Expected: ", maxIndex(s.Out.Data), " - Model's guess: ", maxIndex(actual.Data))
	}
}

func printDigit(img []float64) {
	for i := range ImageSize*ImageSize {
		b := img[i]
		if b > 0.85 {
			fmt.Print("#")
		} else if b > 0.5 {
			fmt.Print("*")
		} else if b > 0 {
			fmt.Print(".")
		} else {
			fmt.Print(" ")
		}

		if i%ImageSize == 0 {
			fmt.Println()
		}
	}
	fmt.Println()
}

func readImgs(path string, num int) [][]float64 {
	imgFile := must(os.Open(path))
	defer imgFile.Close()
	header := make([]byte, 16)
	must(imgFile.Read(header))

	imgs := make([][]float64, 0, num)
	for range num {
		size := 1 * ImageSize * ImageSize
		buf := make([]byte, size)
		n := must(imgFile.Read(buf))
		assert.Equal(n, size, "Error reading from file")

		// Convert byte to float from 0 to 1
		img := make([]float64, size)
		for j, b := range buf {
			img[j] = float64(b) / 256
		}
		imgs = append(imgs, img)
	}
	return imgs
}

func readLabels(path string, num int) []byte {
	labelFile := must(os.Open(path))
	defer labelFile.Close()
	header := make([]byte, 8)
	must(labelFile.Read(header))

	labels := make([]byte, num)
	n := must(labelFile.Read(labels))
	assert.Equal(n, num, "Error reading from file")
	return labels
}

func randomSubset[T any](ts []T, n int) []T {
	if n > len(ts) {
		n = len(ts)
	}

	perm := rand.Perm(len(ts))
	subset := make([]T, n)

	for i := 0; i < n; i++ {
		subset[i] = ts[perm[i]]
	}
	return subset
}

func maxIndex(values []float64) int {
	highestIndex := 0
	highestVal := float64(0)
	for i := range values {
		if values[i] > highestVal {
			highestIndex = i
			highestVal = values[i]
		}
	}
	return highestIndex
}

func must[T any](t T, err error) T {
	if err != nil {
		panic(err)
	}
	return t
}
