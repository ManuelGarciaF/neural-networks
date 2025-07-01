package nn

import "math/rand"

func ceilingDiv(a, b int) int {
	return (a + b - 1) / b
}

func randomSubset[T any](ts []T, n int) []T {
	if n>len(ts) {
		n = len(ts)
	}

	perm := rand.Perm(n)
	subset := make([]T, n)

	for i := 0; i < n; i++ {
		subset[i] = ts[perm[i]]
	}
	return subset
}
