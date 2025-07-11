//go:build noasserts

package assert

import (
	"cmp"
)

func True(condition bool, msg string) {
}

func False(condition bool, msg string) {
}

func Equal[T comparable](a, b T, msg string) {
}

func NotEqual[T comparable](a, b T, msg string) {
}

func GreaterThan[T cmp.Ordered](a, b T, msg string) {
}

func GreaterThanOrEqual[T cmp.Ordered](a, b T, msg string) {
}

func LessThan[T cmp.Ordered](a, b T, msg string) {
}

func LessThanOrEqual[T cmp.Ordered](a, b T, msg string) {
}

// Must asserts that the err value is not nil
func Must[T any](t T, err error) T {
	if err != nil {
		panic(err)
	}
	return t
}
