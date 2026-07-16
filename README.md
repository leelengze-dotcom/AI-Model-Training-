# Iris Flower Classifier (KNN)

A simple machine learning script that trains a K-Nearest Neighbors (KNN) classifier to predict Iris flower species from four measurements: sepal length, sepal width, petal length, and petal width.

## What It Does

1. Loads the classic Iris dataset (150 samples, 3 species) via scikit-learn.
2. Prints a dataset overview: first rows, shape, feature/class names, class balance, missing values, and summary statistics.
3. Splits the data into training (80%) and testing (20%) sets, stratified by species.
4. Scales features with `StandardScaler` (fit on training data only, to avoid data leakage).
5. Trains a `KNeighborsClassifier` (k=5) on the scaled training data.
6. Predicts species on the test set and prints a side-by-side comparison of actual vs. predicted labels.
7. Evaluates performance with accuracy, a confusion matrix, and a full classification report (precision, recall, F1-score).
8. Saves a confusion matrix chart to `results/confusion_matrix.png` and displays it.
9. Classifies one new, unseen flower sample as a demonstration.

## Requirements

- Python 3.8+
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

## Usage

Run the script directly:

```bash
python iris_classifier.py
```

## Output

- Console output: dataset overview, train/test shapes, predictions table, accuracy, confusion matrix, and classification report.
- `results/confusion_matrix.png`: a saved image of the confusion matrix heatmap (created automatically if the `results/` folder doesn't exist).

## Model Details

| Setting | Value |
|---|---|
| Algorithm | K-Nearest Neighbors |
| Neighbors (k) | 5 |
| Train/test split | 80% / 20% |
| Random state | 42 |
| Feature scaling | StandardScaler |
| Stratified split | Yes |

## Results

Running the script on the default 80/20 stratified split (`random_state=42`) with k=5 produced the following confusion matrix on the 30 test samples:

|            | Predicted: setosa | Predicted: versicolor | Predicted: virginica |
|---|---|---|---|
| **Actual: setosa**     | 10 | 0  | 0 |
| **Actual: versicolor** | 0  | 10 | 0 |
| **Actual: virginica**  | 0  | 2  | 8 |

- **Overall accuracy: 28/30 = 93.33%**
- Setosa and versicolor were classified perfectly.
- 2 virginica samples were misclassified as versicolor; no other errors occurred.

**Interpretation:** Setosa is linearly separable from the other two species, so it's classified with perfect accuracy. Versicolor and virginica overlap slightly in petal and sepal measurements, which is why the only errors in this run were virginica samples being confused for versicolor (never the reverse). This is a typical result for KNN on the Iris dataset — a natural next step would be to try different values of k (e.g. 3 or 7) or compare against another classifier to see if the virginica/versicolor boundary can be sharpened.

The chart is saved to `results/confusion_matrix.png` and shown below:

![Confusion Matrix](results/confusion_matrix.png)

## Notes

- The dataset is loaded directly from scikit-learn, so no external data file is needed.
- `random_state=42` is set for reproducible train/test splits.
- The scaler is fit only on the training set and then applied to the test set and any new samples, which prevents information from the test set leaking into training.
