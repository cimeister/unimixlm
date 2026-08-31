package main

import (
	"github.com/Workiva/go-datastructures/queue"
	"github.com/pierods/chain/ring"
)

func main() {

	forever := make(chan bool)

	preLoad := 10000

	ringIn := queue.NewRingBuffer(uint64(preLoad))
	ringOut := queue.NewRingBuffer(uint64(preLoad))
	ringMiddle := queue.NewRingBuffer(uint64(preLoad))

	ring.Load(ringIn, preLoad)

	go ring.Sink(ringOut)

	go ring.Sleep1SmallUnit(ringIn, ringMiddle)

	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)
	go ring.Sleep10SmallUnits(ringMiddle, ringOut)

	<-forever
}