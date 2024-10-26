package assert

import "fmt"

// May be a good idea to disable asserts for speed
var ASSERT_ENABLE = true

func True(condition bool, msg string) {
	if !condition && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %v", msg))
	}
}

func Equal[T comparable](a, b T, msg string) {
	if a != b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (%v != %v)", msg, a, b))
	}
}

func NotEqual[T comparable](a, b T, msg string) {
	if a == b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (expected != %v)", msg, b))
	}
}

func GreaterThan[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a <= b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (%v <= %v)", msg, a, b))
	}
}

func GreaterThanOrEqual[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a < b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (%v < %v)", msg, a, b))
	}
}

func LessThan[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a >= b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (%v >= %v)", msg, a, b))
	}
}

func LessThanOrEqual[T ~int | ~float64 | ~float32](a, b T, msg string) {
	if a > b && ASSERT_ENABLE {
		panic(fmt.Sprintf("Assertion failed: %s (%v > %v)", msg, a, b))
	}
}
