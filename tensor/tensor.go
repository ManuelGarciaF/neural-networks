package tensor

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ManuelGarciaF/neural-networks/assert"
)

type Tensor struct {
	Data    []float64
	shape   []int
	strides []int
}

func New(shape ...int) *Tensor {
	if len(shape) == 0 {
		return &Tensor{
			Data:    make([]float64, 1),
			shape:   []int{},
			strides: []int{},
		}
	}

	// Total size is product of all sizes
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	// Strides are built up from the end
	strides := make([]int, len(shape))
	strides[len(shape)-1] = 1
	// Stride grows by the size of that dimension.
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return &Tensor{
		Data:    make([]float64, size),
		shape:   shape,
		strides: strides,
	}
}

func WithData(shape []int, data []float64) *Tensor {
	t := New(shape...)
	assert.Equal(len(t.Data), len(data), "The provided data does not match the required length")
	copy(t.Data, data)
	return t
}

func Scalar(v float64) *Tensor {
	t := New(1)
	t.Set(v, 0)
	return t
}

func RowVector(data ...float64) *Tensor {
	assert.GreaterThan(len(data), 0, "A vector can't have size 0")
	return WithData([]int{1, len(data)}, data)
}

func ColumnVector(data ...float64) *Tensor {
	assert.GreaterThan(len(data), 0, "A vector can't have size 0")
	return WithData([]int{len(data), 1}, data)
}

func (t *Tensor) Dims() int {
	return len(t.shape)
}

func (t *Tensor) Dim(i int) int {
	assert.GreaterThanOrEqual(i, 0, "Invalid dimension index")

	if i >= t.Dims() {
		return 1
	}
	return t.shape[i]
}

func (t *Tensor) Set(v float64, indices ...int) {
	t.Data[t.getDataIndex(indices)] = v
}

func (t *Tensor) At(indices ...int) float64 {
	return t.Data[t.getDataIndex(indices)]
}

func EqDims(t1, t2 *Tensor) bool {
	maxDims := max(t1.Dims(), t2.Dims())

	for i := 0; i < maxDims; i++ {
		if t1.Dim(i) != t2.Dim(i) {
			return false
		}

	}
	return true
}

func Eq(t1, t2 *Tensor) bool {
	return EqDims(t1, t2) && slices.Equal(t1.Data, t2.Data)
}

func MatMul(left, right *Tensor) *Tensor {
	assert.LessThanOrEqual(left.Dims(), 2, "Element is not a matrix")
	assert.LessThanOrEqual(right.Dims(), 2, "Element is not a matrix")

	leftRows, leftCols := left.Dim(0), left.Dim(1)
	rightRows, rightCols := right.Dim(0), right.Dim(1)

	assert.Equal(leftCols, rightRows, "Matrix dimensions do not match")

	outRows := leftRows
	outCols := rightCols
	sumLen := leftCols

	out := New(outRows, outCols)
	for row := 0; row < outRows; row++ {
		for col := 0; col < outCols; col++ {
			val := float64(0)
			for i := 0; i < sumLen; i++ {
				val += left.At(row, i) * right.At(i, col)
			}
			out.Set(val, row, col)
		}
	}

	return out
}

// The first argument is modified
func Add(t1, t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	for i := range t1.Data {
		t1.Data[i] += t2.Data[i]
	}

	return t1
}

// Applies a function in place to each element of the tensor
func (t *Tensor) Apply(f func(v float64) float64) {
	for i, v := range t.Data {
		t.Data[i] = f(v)
	}
}

func (t *Tensor) ScalarMult(s float64) {
	for _, v := range t.Data {
		v *= s
	}
}

func (t *Tensor) AddElem(v float64) {
	for i := range t.Data {
		t.Data[i] += v
	}
}

func (t *Tensor) getDataIndex(indices []int) int {
	dataIndex := 0
	for dim := 0; dim < t.Dims(); dim++ {
		index := 0
		if dim < len(indices) {
			index = indices[dim]
		}
		assert.True(index >= 0 && index < t.Dim(dim), "Index out of bounds")

		dataIndex += index * t.strides[dim]
	}

	// Just in case, check there are no extra non-0 indices
	for i := t.Dims(); i < len(indices); i++ {
		assert.Equal(0, indices[i], "Index out of bounds")
	}

	return dataIndex
}

// Really horrible, don't look
func (t *Tensor) PrintMatrix(prefix string) {
	fmt.Print(prefix, " ")
	for row := 0; row < t.Dim(0); row++ {
		if row > 0{
			fmt.Print(strings.Repeat(" ", len(prefix)+1))
		}
		if t.Dim(0) == 1 {
			fmt.Print("[")
		} else if row == 0 {
			fmt.Print("┌")
		} else if row == t.Dim(0)-1 {
			fmt.Print("└")
		} else {
			fmt.Print("│")
		}
		for col := 0; col < t.Dim(1); col++ {
			fmt.Printf("%8.4f ", t.At(row, col))
		}
		if t.Dim(0) == 1 {
			fmt.Println("]")
		} else if row == 0 {
			fmt.Println("┐")
		} else if row == t.Dim(0)-1 {
			fmt.Println("┘")
		} else {
			fmt.Println("│")
		}
	}
}
