package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

// Run the model in predict() and predictFlat() using features from the input
// CSV for the model (xtest.csv) and check the predictions match what we expect
// in the output CSV (preds.csv).
func main() {
	if err := testModel(); err != nil {
		log.Fatal(err)
	}
}

func testModel() error {
	featuresRows, err := readCSV("xtest.csv")
	if err != nil {
		return err
	}

	predictionsRows, err := readCSV("preds.csv")
	if err != nil {
		return err
	}

	for i := range featuresRows {
		if err := testPrediction(featuresRows[i], predictionsRows[i][0]); err != nil {
			return err
		}
	}

	return nil
}

func readCSV(path string) ([][]string, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		return nil, err
	}

	return rows, nil
}

func testPrediction(
	featuresRow []string,
	predictionStr string,
) error {
	// Parse features into []*float32 (for predict) and []float32 (for
	// predictFlat, using NaN for missing values).

	ptrFeatures := make([]*float32, len(featuresRow))
	flatFeatures := make([]float32, len(featuresRow))
	for i, featureStr := range featuresRow {
		if featureStr == "" {
			flatFeatures[i] = float32(math.NaN())
			continue
		}

		feature64, err := strconv.ParseFloat(featureStr, 64)
		if err != nil {
			return err
		}
		feature := float32(feature64)
		ptrFeatures[i] = &feature
		flatFeatures[i] = feature
	}

	// Parse prediction into a float32.

	prediction64, err := strconv.ParseFloat(predictionStr, 64)
	if err != nil {
		return err
	}
	expectedPrediction := float32(prediction64)

	// Run the model via both APIs.

	gotPrediction := predict(ptrFeatures, false)
	if err := checkPrediction("predict", gotPrediction, expectedPrediction); err != nil {
		return err
	}

	gotFlatPrediction := predictFlat(flatFeatures, false)
	if err := checkPrediction("predictFlat", gotFlatPrediction, expectedPrediction); err != nil {
		return err
	}

	return nil
}

// checkPrediction verifies that gotPrediction is close to expectedPrediction.
//
// Allow for float32 rounding differences between XGBoost's prediction and
// the generated code. Regression objectives can produce large-magnitude
// outputs where a tight absolute bound is unrealistic, so accept the
// prediction if either the absolute or the relative error is small. The
// relative denominator is floored at 1.0 so that for small-magnitude
// outputs (the [0, 1] range of logistic objectives, or near-zero
// regression targets) the check stays effectively absolute, rather than a
// target near zero blowing the relative error up to infinity.
func checkPrediction(name string, gotPrediction, expectedPrediction float32) error {
	const tolerance = 0.00001
	absDelta := math.Abs(float64(gotPrediction - expectedPrediction))
	relDelta := absDelta / math.Max(math.Abs(float64(expectedPrediction)), 1.0)
	if absDelta > tolerance && relDelta > tolerance {
		return fmt.Errorf("%s: got %f, expected %f", name, gotPrediction, expectedPrediction)
	}
	return nil
}
