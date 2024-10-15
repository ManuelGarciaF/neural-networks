package nn

import (
	"testing"

	ts "github.com/ManuelGarciaF/neural-networks/tensor"
)

func TestFullyConnectedLayer_Forward(t *testing.T) {
	type fields struct {
		weights *ts.Tensor
		bias    *ts.Tensor
	}
	type args struct {
		in *ts.Tensor
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *ts.Tensor
	}{
		{
			name: "1x1 forward",
			fields: fields{
				weights: ts.Scalar(5),
				bias:    ts.Scalar(0.5),
			},
			args: args{
				in: ts.Scalar(2),
			},
			want: ts.Scalar(5*2 + 0.5),
		},
		{
			name: "1x2 forward",
			fields: fields{
				weights: ts.RowVector(1, 2),
				bias:    ts.Scalar(0.5),
			},
			args: args{
				in: ts.ColumnVector(3, 4),
			},
			want: ts.Scalar(1*3 + 2*4 + 0.5),
		},
		{
			name: "2x2 forward",
			fields: fields{
				weights: ts.WithData([]int{2, 2}, []float64{1, 2, 3, 4}),
				bias:    ts.ColumnVector(0.5, 0.2),
			},
			args: args{
				in: ts.ColumnVector(5, 6),
			},
			want: ts.ColumnVector(1*5+2*6+0.5, 3*5+4*6+0.2),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &FullyConnectedLayer{
				Weights: tt.fields.weights,
				Bias:    tt.fields.bias,
			}
			if got := l.Forward(tt.args.in); !ts.Eq(got, tt.want) {
				t.Errorf("FullyConnectedLayer.Forward() = %v, want %v", got.Data, tt.want.Data)
			}
		})
	}
}
