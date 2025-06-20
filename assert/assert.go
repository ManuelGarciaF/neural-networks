//go:build !noasserts

package assert

import "fmt"

func True(condition bool, msg string) {
	if !condition {
		panic(fmt.Sprintf("Assertion failed: %v", msg))
	}
}

func False(condition bool, msg string) {
	if condition {
		panic(fmt.Sprintf("Assertion failed: %v", msg))
	}
}

func Equal[T comparable](a, b T, msg string) {
	if a != b {
		panic(fmt.Sprintf("Assertion failed: %s (%v != %v)", msg, a, b))
	}
}

func NotEqual[T comparable](a, b T, msg string) {
	if a == b {
		panic(fmt.Sprintf("Assertion failed: %s (expected != %v)", msg, b))
	}
}

func GreaterThan[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a <= b {
		panic(fmt.Sprintf("Assertion failed: %s (%v <= %v)", msg, a, b))
	}
}

func GreaterThanOrEqual[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a < b {
		panic(fmt.Sprintf("Assertion failed: %s (%v < %v)", msg, a, b))
	}
}

func LessThan[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a >= b {
		panic(fmt.Sprintf("Assertion failed: %s (%v >= %v)", msg, a, b))
	}
}

func LessThanOrEqual[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a > b {
		panic(fmt.Sprintf("Assertion failed: %s (%v > %v)", msg, a, b))
	}
}

// Must asserts that the err value is not nil
func Must[T any](t T, err error) T {
	if err != nil {
		panic(err)
	}
	return t
}
