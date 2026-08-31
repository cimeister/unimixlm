package main

import "fmt"

func main() {

	var myGreeting = map[string]string{}
	myGreeting["Terry"] = "Hola!"
	myGreeting["Ernesto"] = "Good morning!"

	fmt.Println(myGreeting)

	var myOther = map[string]string{
		"Tim":  "Hola!",
		"Bert": "Good morning!",
	}
	myOther["Jack"] = "Arrrr Matey!!"
	fmt.Println(myOther)

	var myNumbers = map[int]int{}
	y := 100
	for x := 0; x < 101; x++ {
		myNumbers[x] = y
		y--
	}
	fmt.Println(myNumbers)
}