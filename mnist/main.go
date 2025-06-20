package main

import (
	"fmt"
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

	// Build training data
	trainSamples := 10000
	trainImgs := readImgs("./train-images.idx3-ubyte", trainSamples)
	trainLabels := readLabels("./train-labels.idx1-ubyte", trainSamples)

	trainData := make([]nn.Sample, trainSamples)
	for i := range trainData {
		trainData[i] = nn.Sample{
			In:  t.WithData([]int{ImageSize * ImageSize}, trainImgs[i]),
			Out: LabelVectors[trainLabels[i]],
		}
	}

	testSamples := 500
	testImgs := readImgs("./t10k-images.idx3-ubyte", testSamples)
	testLabels := readLabels("./t10k-labels.idx1-ubyte", testSamples)

	testData := make([]nn.Sample, testSamples)
	for i := range testData {
		testData[i] = nn.Sample{
			In:  t.WithData([]int{ImageSize * ImageSize}, testImgs[i]),
			Out: LabelVectors[testLabels[i]],
		}
	}

	model := nn.NewMLP([]int{
		ImageSize * ImageSize, // Input pixels
		256,
		256,
		10,                    // Outputs
	}, nn.ReLU{}, true, 1.0)

	model.TrainSingleThreaded(testData, 9, 0.25, 0, 9)

	fmt.Println("----------------------------")
	fmt.Println("Final loss:", model.AverageLoss(testData))
}

func printDigit(img []float64) {
	for i := 0; i < ImageSize*ImageSize; i++ {
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
	imgFile := assert.Must(os.Open(path))
	defer imgFile.Close()
	header := make([]byte, 16)
	assert.Must(imgFile.Read(header))

	imgs := make([][]float64, 0, num)
	for i := 0; i < num; i++ {
		size := 1 * ImageSize * ImageSize
		buf := make([]byte, size)
		n := assert.Must(imgFile.Read(buf))
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
	labelFile := assert.Must(os.Open(path))
	defer labelFile.Close()
	header := make([]byte, 8)
	assert.Must(labelFile.Read(header))

	labels := make([]byte, num)
	n := assert.Must(labelFile.Read(labels))
	assert.Equal(n, num, "Error reading from file")
	return labels
}
