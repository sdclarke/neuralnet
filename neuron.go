package neuralnet

import (
	"math/rand"
	"time"
)

type neuron struct {
	Weights []float64
	Bias    float64
	Value   float64
	Delta   float64
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

func myRand() float64 {
	return rand.Float64()*2 - 1
}

func NewNeuron(dimension int) *neuron {
	n := &neuron{
		Weights: make([]float64, dimension),
		Bias:    myRand(),
		Value:   myRand(),
		Delta:   myRand(),
	}
	for i, _ := range n.Weights {
		n.Weights[i] = myRand()
	}
	return n
}
