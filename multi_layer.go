package neuralnet

import (
	"math"
)

type multiLayer struct {
	Layers     []*neuralLayer
	Lrate      float64
	Tfunc      func(float64) float64
	TfuncPrime func(float64) float64
}

func NewMultiLayer(layers []int, lr float64, tf, tfp func(float64) float64) *multiLayer {
	ml := &multiLayer{
		Layers:     make([]*neuralLayer, len(layers)),
		Lrate:      lr,
		Tfunc:      tf,
		TfuncPrime: tfp,
	}
	for i, n := range layers {
		if i == 0 {
			ml.Layers[i] = NewNeuralLayer(n, 0)
		} else {
			ml.Layers[i] = NewNeuralLayer(n, layers[i-1])
		}
	}
	return ml
}

func (ml *multiLayer) Execute(s *Pattern) []float64 {
	v := 0.0

	for i, f := range s.Features {
		ml.Layers[0].Neurons[i].Value = f
	}

	for i, _ := range ml.Layers {
		if i == 0 {
			continue
		}
		for j, _ := range ml.Layers[i].Neurons {
			v = 0.0
			for k, _ := range ml.Layers[i-1].Neurons {
				v += ml.Layers[i].Neurons[j].Weights[k] * ml.Layers[i-1].Neurons[k].Value
			}
			v += ml.Layers[i].Neurons[j].Bias
			ml.Layers[i].Neurons[j].Value = ml.Tfunc(v)
		}
	}

	r := []float64{}
	for _, neuron := range ml.Layers[len(ml.Layers)-1].Neurons {
		r = append(r, neuron.Value)
	}
	return r
}

func (ml *multiLayer) BackPropagate(s *Pattern, label []float64) float64 {
	e := 0.0
	outputs := ml.Execute(s)
	outputErrors := []float64{}
	for i, _ := range ml.Layers[len(ml.Layers)-1].Neurons {
		outputErrors = append(outputErrors, label[i]-outputs[i])
	}

	for i := len(ml.Layers) - 2; i >= 0; i-- {
		for j, _ := range ml.Layers[i].Neurons {
			e = 0.0
			for k, _ := range ml.Layers[i+1].Neurons {
				e += ml.Layers[i+1].Neurons[k].Delta * ml.Layers[i+1].Neurons[k].Weights[j]
			}
			ml.Layers[i].Neurons[j].Delta = e * ml.TfuncPrime(ml.Layers[i].Neurons[j].Value)
		}
		for j, _ := range ml.Layers[i+1].Neurons {
			for k, _ := range ml.Layers[i].Neurons {
				ml.Layers[i+1].Neurons[j].Weights[k] += ml.Lrate * ml.Layers[i+1].Neurons[j].Delta * ml.Layers[i+1].Neurons[j].Value
			}
		}
	}
	r := 0.0
	for i, output := range outputs {
		r += math.Abs(output - label[i])
	}
	r = r / float64(len(outputs))
	return r
}
