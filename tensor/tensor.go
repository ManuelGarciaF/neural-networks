package tensor

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"

	"github.com/ManuelGarciaF/neural-networks/assert"
)

type Tensor struct {
	Data    []float64
	Shape   []int32
	strides []int32
}

func New(shape ...int32) *Tensor {
	if len(shape) == 0 {
		return &Tensor{
			Data:    make([]float64, 1),
			Shape:   make([]int32, 0),
			strides: []int32{1},
		}
	}

	// Total size is product of all sizes
	size := int32(1)
	for _, dim := range shape {
		size *= dim
	}

	// Strides are built up from the end
	strides := make([]int32, len(shape))
	strides[len(shape)-1] = 1
	// Stride grows by the size of that dimension.
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}

	return &Tensor{
		Data:    make([]float64, size),
		Shape:   shape,
		strides: strides,
	}
}

func WithData(shape []int32, data []float64) *Tensor {
	t := New(shape...)
	assert.Equal(len(t.Data), len(data), "The provided data does not match the required length")
	copy(t.Data, data)
	return t
}

func Scalar(v float64) *Tensor {
	t := New()
	t.Set(v)
	return t
}

func RowVector(data ...float64) *Tensor {
	assert.GreaterThan(len(data), 0, "A vector can't have size 0")
	size := int32(len(data))
	return WithData([]int32{1, size}, data)
}

func ColumnVector(data ...float64) *Tensor {
	assert.GreaterThan(len(data), 0, "A vector can't have size 0")
	size := int32(len(data))
	return WithData([]int32{size, 1}, data)
}

func (t *Tensor) Dims() int {
	return len(t.Shape)
}

func (t *Tensor) Dim(i int) int32 {
	assert.GreaterThanOrEqual(i, 0, "Invalid dimension index")

	if i >= t.Dims() {
		return 1
	}
	return t.Shape[i]
}

func (t *Tensor) Set(v float64, indices ...int32) {
	t.Data[t.getDataIndex(indices)] = v
}

func (t *Tensor) At(indices ...int32) float64 {
	return t.Data[t.getDataIndex(indices)]
}

func (t *Tensor) Rows() int32 {
	return t.Dim(0)
}

func (t *Tensor) Cols() int32 {
	return t.Dim(1)
}

func (t *Tensor) Copy() *Tensor {
	data := make([]float64, len(t.Data))
	copy(data, t.Data)
	shape := make([]int32, len(t.Shape))
	copy(shape, t.Shape)
	strides := make([]int32, len(t.strides))
	copy(strides, t.strides)

	return &Tensor{Data: data, Shape: shape, strides: strides}
}

func EqDims(t1, t2 *Tensor) bool {
	maxDims := max(t1.Dims(), t2.Dims())

	for i := range maxDims {
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
	assert.Equal(left.Cols(), right.Rows(), "Matrix dimensions do not match")

	outRows := left.Rows()
	outCols := right.Cols()
	sumLen := left.Cols()

	out := New(outRows, outCols)
	for row := range outRows {
		for col := range outCols {
			val := 0.0
			for i := range sumLen {
				val += left.At(row, i) * right.At(i, col)
			}
			out.Set(val, row, col)
		}
	}

	return out
}

// The receiver is modified
func (t1 *Tensor) AddInPlace(t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	for i := range t1.Data {
		t1.Data[i] += t2.Data[i]
	}

	return t1
}

func Add(t1, t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	new := t1.Copy()
	for i := range new.Data {
		new.Data[i] += t2.Data[i]
	}

	return new
}

// The receiver is modified
func (t1 *Tensor) SubInPlace(t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	for i := range t1.Data {
		t1.Data[i] -= t2.Data[i]
	}

	return t1
}

func Sub(t1, t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	new := t1.Copy()
	for i := range new.Data {
		new.Data[i] -= t2.Data[i]
	}

	return new
}

func ElementMult(t1, t2 *Tensor) *Tensor {
	assert.True(EqDims(t1, t2), "Tensors do not have the same shape")

	new := t1.Copy()
	for i := range new.Data {
		new.Data[i] *= t2.Data[i]
	}

	return new
}

// Applies a function to each element of the tensor.
func Map(t *Tensor, f func(v float64) float64) *Tensor {
	a := t.Copy()
	for i, v := range a.Data {
		a.Data[i] = f(v)
	}
	return a
}

func AddToElems(t *Tensor, v float64) *Tensor {
	out := t.Copy()
	for i := range out.Data {
		out.Data[i] += v
	}
	return out
}

func ScalarMult(t *Tensor, v float64) *Tensor {
	out := t.Copy()
	for i := range out.Data {
		out.Data[i] *= v
	}
	return out
}

func (t *Tensor) ScaleInPlace(v float64) *Tensor {
	for i := range t.Data {
		t.Data[i] *= v
	}
	return t
}

func (t *Tensor) ColVectorNorm1() float64 {
	assert.Equal(t.Cols(), 1, "Not a column vector")

	sum := 0.0
	for _, v := range t.Data {
		sum += math.Abs(v)
	}
	return sum
}

func (t *Tensor) ColVectorNorm2() float64 {
	assert.Equal(t.Cols(), 1, "Not a column vector")

	sum := 0.0
	for _, v := range t.Data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func MatTranspose(t *Tensor) *Tensor {
	assert.LessThanOrEqual(t.Dims(), 2, "Element is not a matrix")

	new := New(t.Cols(), t.Rows())
	for r := int32(0); r < t.Rows(); r++ {
		for c := int32(0); c < t.Cols(); c++ {
			new.Set(t.At(r, c), c, r)
		}
	}

	return new
}

func (t *Tensor) MatrixNormInf() float64 {
	assert.LessThanOrEqual(t.Dims(), 2, "Element is not a matrix")

	// Max row sum
	max := 0.0
	for r := int32(0); r < t.Rows(); r++ {
		sum := 0.0
		for c := int32(0); c < t.Cols(); c++ {
			sum += math.Abs(t.At(r, c))
		}
		if sum > max {
			max = sum
		}
	}
	return max
}

func (t *Tensor) Contains(v float64) bool {
	return slices.Contains(t.Data, v)
}

func (t *Tensor) Any(f func(v float64) bool) bool {
	return slices.ContainsFunc(t.Data, f)
}

func (t *Tensor) IsFinite() bool {
	return !(t.Any(math.IsNaN) && t.Any(isInf))

}

// Serialization functions
func (t *Tensor) Save(w io.Writer) error {
	// We just need to save the data and shapes, strides is computed on creation.

	// Shape
	dims := int32(len(t.Shape)) // Cast to keep standard sizes
	err := binary.Write(w, binary.LittleEndian, dims)
	if err != nil {
		return err
	}
	err = binary.Write(w, binary.LittleEndian, t.Shape)
	if err != nil {
		return err
	}

	// Data
	size := int32(len(t.Data))
	err = binary.Write(w, binary.LittleEndian, size)
	if err != nil {
		return err
	}
	err = binary.Write(w, binary.LittleEndian, t.Data)
	if err != nil {
		return err
	}

	return nil
}

func Load(r io.Reader) (*Tensor, error) {
	// Read dims
	var dims int32
	err := binary.Read(r, binary.LittleEndian, &dims)
	if err != nil {
		return nil, err
	}
	shape := make([]int32, dims)
	err = binary.Read(r, binary.LittleEndian, shape)
	if err != nil {
		return nil, err
	}

	var dataSize int32
	err = binary.Read(r, binary.LittleEndian, &dataSize)
	if err != nil {
		return nil, err
	}
	data := make([]float64, dataSize)
	err = binary.Read(r, binary.LittleEndian, data)
	if err != nil {
		return nil, err
	}

	return WithData(shape, data), nil
}

func isInf(v float64) bool {
	return math.IsInf(v, 0)
}

func (t *Tensor) PrintMatrix(prefix string) {
	fmt.Print(prefix, " ")
	pad := strings.Repeat(" ", len(prefix)+1)

	for row := int32(0); row < t.Rows(); row++ {
		// Align after prefix
		if row > 0 {
			fmt.Print(pad)
		}
		// Starting marker
		if t.Rows() == 1 {
			fmt.Print("[")
		} else if row == 0 {
			fmt.Print("┌")
		} else if row == t.Rows()-1 {
			fmt.Print("└")
		} else {
			fmt.Print("│")
		}
		// Contents
		for col := int32(0); col < t.Dim(1); col++ {
			fmt.Printf("%7.5f ", t.At(row, col))
		}
		// Ending marker
		if t.Rows() == 1 {
			fmt.Println("]")
		} else if row == 0 {
			fmt.Println("┐")
		} else if row == t.Rows()-1 {
			fmt.Println("┘")
		} else {
			fmt.Println("│")
		}
	}
}

func (t *Tensor) getDataIndex(indices []int32) int32 {
	// Unrolled loop for efficiency
	switch t.Dims() {
	case 0:
		return 0
	case 1:
		return t.getDataIndex1D(indices)
	case 2:
		return t.getDataIndex2D(indices)
	default: // For more complex tensors, compute using a loop
		dataIndex := int32(0)
		for dim := 0; dim < t.Dims(); dim++ {
			index := int32(0)
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
}

func (t *Tensor) getDataIndex2D(indices []int32) int32 {
	i := indices[0]
	assert.True(i >= 0 && i < t.Shape[0], "Index out of bounds")

	// Sometimes only 1 index is given, we assume the other is 0
	if len(indices) == 1 {
		return i * t.strides[0]
	}

	j := indices[1]
	assert.True(j >= 0 && j < t.Shape[1], "Index out of bounds")

	return i*t.strides[0] + j*t.strides[1]
}

func (t *Tensor) getDataIndex1D(indices []int32) int32 {
	i := indices[0]
	assert.True(i >= 0 && i < t.Shape[0], "Index out of bounds")
	return i
}
