package main

import (
	"encoding/csv"
	"io"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/sdclarke/neuralnet"
)

func main() {
	thresholdFunction := func(x float64) float64 {
		return 1.0 / (1 + math.Exp(-x))
	}
	thresholdFunctionPrime := func(x float64) float64 {
		return x * (1.0 - x)
		//return thresholdFunction(x) * (1.0 - thresholdFunction(x))
	}
	ml := neuralnet.NewMultiLayer([]int{4, 3, 3}, 0.1, thresholdFunction, thresholdFunctionPrime)
	for i := 0; i < 100; i++ {
		file, err := os.Open("data.csv")
		if err != nil {
			log.Fatalf("Error reading file: %v", err)
		}
		reader := csv.NewReader(file)
		for {
			line, err := reader.Read()
			if err != nil {
				if err == io.EOF {
					break
				}
				log.Fatalf("Error reading line: %v", err)
			}
			features := []float64{}
			labels := []float64{}
			for i, f := range line {
				floatValue, err := strconv.ParseFloat(f, 64)
				if err != nil {
					log.Fatalf("Error converting float to string: %v", err)
				}
				if i < 4 {
					features = append(features, floatValue)
				} else {
					labels = append(labels, floatValue)
				}
			}
			pattern := &neuralnet.Pattern{Features: features, Label: labels}

			ml.BackPropagate(pattern, pattern.Label)
		}
		file.Close()
	}
	file, err := os.Open("data_test.csv")
	if err != nil {
		log.Fatalf("Error reading file: %v", err)
	}
	reader := csv.NewReader(file)
	for {
		line, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalf("Error reading line: %v", err)
		}
		features := []float64{}
		labels := []float64{}
		for i, f := range line {
			floatValue, err := strconv.ParseFloat(f, 64)
			if err != nil {
				log.Fatalf("Error converting float to string: %v", err)
			}
			if i < 4 {
				features = append(features, floatValue)
			} else {
				labels = append(labels, floatValue)
			}
		}
		pattern := &neuralnet.Pattern{Features: features, Label: labels}

		outputs := ml.Execute(pattern)
		best := 0.0
		bestLabel := 0
		for i, output := range outputs {
			if output > best {
				best = output
				bestLabel = i
			}
		}
		log.Printf("Pattern label: %v, Predicted label: %v", labels, bestLabel)
	}
	file.Close()

}
