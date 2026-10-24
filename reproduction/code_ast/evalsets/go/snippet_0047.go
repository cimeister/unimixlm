/*
	https://adventofcode.com/2021/day/9
*/

package days

import (
	"fmt"
	"sort"
)

//  2 [1] 9  9  9  4  3  2  1 [0]
//  3  9  8  7  8  9  4  9  2  1
//  9  8 [5] 6  7  8  9  8  9  2
//  8  7  6  7  8  9  6  7  8  9
//  9  8  9  9  9  6 [5] 6  7  8

func isMin(r, c int, xs [][]int) bool {
	x := xs[r][c]

	if r >= 1 {
		if xs[r-1][c] <= x {
			return false
		}
	}
	if r+1 < len(xs) {
		if xs[r+1][c] <= x {
			return false
		}
	}
	if c >= 1 {
		if xs[r][c-1] <= x {
			return false
		}
	}
	if c+1 < len(xs[0]) {
		if xs[r][c+1] <= x {
			return false
		}
	}

	return true
}

func getMins(xs [][]int) (mins []int, minsWithPoing [][]int) {
	for ri := 0; ri < len(xs); ri++ {
		for ci := 0; ci < len(xs[0]); ci++ {
			if isMin(ri, ci, xs) {
				mins = append(mins, xs[ri][ci])
				minsWithPoing = append(minsWithPoing, []int{xs[ri][ci], ri, ci})
			}
		}
	}
	return mins, minsWithPoing
}

func smokeBasinP1(xs [][]int) int {
	mins, _ := getMins(xs)
	sum := 0
	for _, i := range mins {
		sum += i + 1
	}

	return sum
}

func smokeBasinP2(xs [][]int) int {
	seen := make(map[string]bool)
	basins := make([]int, 0)
	_, mins := getMins(xs)
	for _, min := range mins {
		_, r, c := min[0], min[1], min[2]
		basins = append(basins, getBasinSize(r, c, xs, seen))
	}
	sort.Ints(basins)
	res := 1
	for _, n := range basins[len(basins)-3:] {
		res *= n
	}
	return res
}

func getBasinSize(r, c int, xs [][]int, seen map[string]bool) int {
	size := 0
	if xs[r][c] < 9 && !seen[fmt.Sprintf("%dx%d", c, r)] {
		size++
		seen[fmt.Sprintf("%dx%d", c, r)] = true
		if r >= 1 {
			size += getBasinSize(r-1, c, xs, seen)
		}
		if r+1 < len(xs) {
			size += getBasinSize(r+1, c, xs, seen)
		}
		if c >= 1 {
			size += getBasinSize(r, c-1, xs, seen)
		}
		if c+1 < len(xs[0]) {
			size += getBasinSize(r, c+1, xs, seen)
		}
	}
	return size
}