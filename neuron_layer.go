package neuralnet

type neuralLayer struct {
	Neurons []*neuron
	Length  int
}

func NewNeuralLayer(n, p int) *neuralLayer {
	nl := &neuralLayer{
		Neurons: make([]*neuron, n),
		Length:  n,
	}
	for i, _ := range nl.Neurons {
		nl.Neurons[i] = NewNeuron(p)
	}
	return nl
}
