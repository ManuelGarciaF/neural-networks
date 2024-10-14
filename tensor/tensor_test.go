package tensor

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		wantDim  int
		wantSize int
	}{
		{"Scalar", []int{}, 0, 1},
		{"Vector", []int{3}, 1, 3},
		{"Matrix", []int{2, 3}, 2, 6},
		{"3D Tensor", []int{2, 3, 4}, 3, 24},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := New(tt.shape...)
			if tensor.Dims() != tt.wantDim {
				t.Errorf("New(%v).Dim() = %v, want %v", tt.shape, tensor.Dims(), tt.wantDim)
			}
			if len(tensor.Data) != tt.wantSize {
				t.Errorf("len(New(%v).Data) = %v, want %v", tt.shape, len(tensor.Data), tt.wantSize)
			}
			if !reflect.DeepEqual(tensor.shape, tt.shape) {
				t.Errorf("New(%v).Shape = %v, want %v", tt.shape, tensor.shape, tt.shape)
			}
		})
	}
}

func TestSetAndAt(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		indices []int
		value   float64
	}{
		{"Scalar", []int{}, []int{}, 42},
		{"Vector", []int{3}, []int{1}, 42},
		{"Matrix", []int{2, 3}, []int{1, 2}, 42},
		{"3D Tensor", []int{2, 3, 4}, []int{1, 2, 3}, 42},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := New(tt.shape...)
			tensor.Set(tt.value, tt.indices...)
			if got := tensor.At(tt.indices...); got != tt.value {
				t.Errorf("After Set(%v, %v), At(%v) = %v, want %v", tt.value, tt.indices, tt.indices, got, tt.value)
			}
		})
	}
}

func TestSetAndAtPanic(t *testing.T) {
	tensor := New(2, 3)

	testCases := []struct {
		name    string
		indices []int
	}{
		{"Too Few Indices", []int{1}},
		{"Too Many Indices", []int{1, 2, 3}},
		{"Out of Bounds", []int{2, 3}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Set(%v) did not panic", tc.indices)
				}
			}()
			tensor.Set(42, tc.indices...)
		})

		t.Run(tc.name+" (At)", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("At(%v) did not panic", tc.indices)
				}
			}()
			tensor.At(tc.indices...)
		})
	}
}

func TestMatMul(t *testing.T) {
	tests := []struct {
		name  string
		left  *Tensor
		right *Tensor
		want  *Tensor
	}{
		{
			name:  "2x3 * 3x2",
			left:  WithData([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
			right: WithData([]int{3, 2}, []float64{7, 8, 9, 10, 11, 12}),
			want:  WithData([]int{2, 2}, []float64{58, 64, 139, 154}),
		},
		{
			name:  "3x2 * 2x3",
			left:  WithData([]int{3, 2}, []float64{1, 2, 3, 4, 5, 6}),
			right: WithData([]int{2, 3}, []float64{7, 8, 9, 10, 11, 12}),
			want:  WithData([]int{3, 3}, []float64{27, 30, 33, 61, 68, 75, 95, 106, 117}),
		},
		{
			name:  "1x2 * 2x1",
			left:  RowVector([]float64{2, 3}),
			right: ColumnVector([]float64{4, 5}),
			want:  Scalar(23),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MatMul(tt.left, tt.right)
			if !Eq(got, tt.want) {
				t.Errorf("MatMul() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatMulPanic(t *testing.T) {
	testCases := []struct {
		name  string
		left  *Tensor
		right *Tensor
	}{
		{"Incompatible Dimensions", New(2, 3), New(2, 3)},
		{"3D Tensor", New(2, 3, 4), New(4, 2)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("MatMul(%v, %v) did not panic", tc.left, tc.right)
				}
			}()
			MatMul(tc.left, tc.right)
		})
	}
}

func TestEq(t *testing.T) {
	tests := []struct {
		name string
		t1   *Tensor
		t2   *Tensor
		want bool
	}{
		{name: "exact", t1: Scalar(1), t2: Scalar(1), want: true},
		{name: "exact", t1: Scalar(1), t2: Scalar(1.2), want: false},
		{
			name: "scalar == 1 element rowVector",
			t1:   Scalar(1),
			t2:   RowVector([]float64{1}),
			want: true,
		},
		{
			name: "scalar == 1 element colVector",
			t1:   Scalar(1),
			t2:   ColumnVector([]float64{1}),
			want: true,
		},
		{
			name: "scalar == 1x1 matrix",
			t1:   Scalar(1),
			t2:   WithData([]int{1, 1}, []float64{1}),
			want: true,
		},
		{
			name: "2 element row != 2 element col",
			t1:   RowVector([]float64{1,2}),
			t2:   ColumnVector([]float64{1,2}),
			want: false,
		},
		{
			name: "2x2 matrixes",
			t1:   WithData([]int{2,2}, []float64{1,2,3,4}),
			t2:   WithData([]int{2,2}, []float64{1,2,3,4}),
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Eq(tt.t1, tt.t2); got != tt.want {
				t.Errorf("Eq() = %v, want %v", got, tt.want)
			}
		})
	}
}
