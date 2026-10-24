package main

import (
	"fmt"
	"math"
)

// <NAME>

func main() {

	var a, b, c, d, i, ans int64

	fmt.Scanf("%d %d\n", &a, &b)
	fmt.Scanf("%d %d\n", &c, &d)

	ans = -1

	for i = 0; i <= 100; i++ {
		j := float64(b+a*i-d) / float64(c)

		if j == math.Trunc(j) && j >= 0 {
			ans = a*i + b
			break
		}
	}

	fmt.Println(ans)

	// 20i +  2 (20*4+2 = 82)
	//  9i + 19 (9*7+19 = 82)

}

func max(a, b int64) int64 {
	if a > b {
		return a
	}

	return b
}