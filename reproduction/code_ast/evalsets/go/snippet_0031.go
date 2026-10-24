// Nonempty is an example of an in-place slice algorithm
package main

import "fmt"

func nonempty(strings []string) []string {
	i := 0
	for _, s := range strings {
		if s != "" {
			strings[i] = s
			i++
		}
	}
	return strings[:i]
}

func nonempty2(strings []string) []string {
	out := strings[:0] // zero-length slice of the original
	for _, s := range strings {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func main() {
	data := []string{"one", "", "three"}
	fmt.Printf("%q\n", nonempty(data))
	fmt.Printf("%q\n", data)

	data2 := []string{"one", "", "three"}
	data2 = nonempty(data2)
	fmt.Printf("%q\n", data2)

	data3 := []string{"one", "", "three"}
	fmt.Printf("%q\n", nonempty2(data3))
	fmt.Printf("%q\n", data3)

	data4 := []string{"one", "", "three"}
	data4 = nonempty2(data4)
	fmt.Printf("%q\n", data4)
}