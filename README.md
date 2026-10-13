# Diabetes Risk Check

A small machine learning web app that estimates the likelihood of diabetes from eight health measurements. It uses a scikit-learn model trained on the PIMA Indians Diabetes dataset and a Flask front end.

> **Disclaimer:** This is a learning project and a screening estimate, not a medical diagnosis. Do not use it to make health decisions. Talk to a doctor about your results.

## Features

- Ensemble model (random forest, logistic regression and gradient boosting) with a tuned decision threshold, built for higher recall
- Preprocessing (missing-value handling, feature engineering, scaling) lives inside one pipeline, so training and prediction behave the same
- Web form with live validation, a progress bar, a BMI calculator, an animated risk gauge, and saved inputs
- Light and dark theme, mobile friendly

## Project structure

```
DiabetesPrediction/
├── app.py                                   # Flask app
├── diabetes_model.pkl                       # saved model bundle (created by the notebook)
├── Diabetes_Prediction_model_improved.ipynb # training and evaluation notebook
├── diabetes.csv                             # PIMA dataset (with header row)
├── requirements.txt
└── templates/
    └── index.html                           # front end
```

## Setup

1. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   flask
   numpy
   pandas
   scikit-learn
   jupyter
   ```

3. **Get the dataset.** Place `diabetes.csv` next to the notebook. It needs a header row with these columns:
   `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

4. **Train the model.** Open `Diabetes_Prediction_model_improved.ipynb` and run all cells. The last cell writes `diabetes_model.pkl`.

5. **Run the app**

   ```bash
   python app.py
   ```

   Then open http://127.0.0.1:5000 in your browser.

## Using the app

Enter the eight values and select **Check my risk**. The result shows the model's estimated probability on a gauge, with a "higher chance" or "lower chance" label.

| Field | Unit | Notes |
|---|---|---|
| Pregnancies | count | |
| Glucose | mg/dL | Fasting plasma glucose |
| Blood pressure | mm Hg | Diastolic |
| Skin thickness | mm | Enter 0 if not measured |
| Insulin | mu U/mL | 2-hour serum. Enter 0 if not measured |
| BMI | kg/m² | Use the built-in height and weight calculator if needed |
| Pedigree function | score | Family history score |
| Age | years | |

Zeros in Glucose, Blood pressure, Skin thickness, Insulin and BMI are treated as missing values and filled in by the model pipeline.

## How the model works

1. **Cleaning:** zeros in the five columns above become missing values and are filled with the training-set median.
2. **Features:** Glucose × BMI, Glucose × Age, Age × Pregnancies, and flags for missing Insulin and Skin thickness.
3. **Model:** soft-voting ensemble of a class-weighted random forest, a class-weighted logistic regression, and a gradient boosting classifier.
4. **Threshold:** the decision cutoff is tuned on out-of-fold predictions from the training data only (about 0.34 in the last run), because the default 0.5 cutoff under-detects diabetic cases.
5. **Saved bundle:** `diabetes_model.pkl` is a dictionary with `model`, `threshold` and `columns`.

### Results

Numbers from the last notebook run on the public PIMA dataset (768 rows):

| Evaluation | Accuracy | Recall | F1 |
|---|---|---|---|
| Baseline random forest, 5-fold CV repeated 3 times | 0.766 | 0.616 | 0.646 |
| Improved ensemble, 5-fold CV repeated 3 times | 0.769 | 0.725 | 0.687 |
| Improved ensemble, held-out test split (154 rows) | 0.740 | 0.815 | 0.688 |

Accuracy on this dataset tends to plateau around 76 to 78 percent, so the main gain is in recall, at the cost of more false positives. The test split is small, so the cross-validated rows are the steadier comparison. Your own numbers may differ slightly.

## Troubleshooting

**`AttributeError: Can't get attribute 'add_features' on <module '__main__'>`**
The pickled pipeline refers to the `add_features` function by name. Define `add_features` in `app.py` (exactly as in the notebook) *before* calling `pickle.load`. A cleaner option is to put it in a `features.py` file and import it in both the notebook and `app.py`, then re-save the model.

**Model loads but predictions fail**
The pickle is a dictionary, not a bare model. Load it with `bundle = pickle.load(...)` and use `bundle["model"]` and `bundle["threshold"]`.

**Warnings or errors about scikit-learn versions**
Run the app with the same scikit-learn version used to train the model. Check with `pip show scikit-learn` in both environments, or re-run the notebook inside the app's environment.

**`FileNotFoundError: diabetes.csv`**
Place the CSV in the same folder as the notebook and make sure it has the header row listed above.

## Possible next steps

- Move `add_features` into a shared `features.py`
- Add unit and range checks on the server as well as in the browser
- Try probability calibration so the percentage better reflects real risk
- Add model explanations (for example, feature importance per prediction)

## Data source

PIMA Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases), widely mirrored on Kaggle and the UCI repository.
