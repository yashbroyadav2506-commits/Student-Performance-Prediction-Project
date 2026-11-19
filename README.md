
# Student Performance Prediction Project — Full Summary

## 1. What You Built

You created an end-to-end **machine learning application that predicts a student's performance category**:

- **Low**: final grade below 10
- **Medium**: final grade from 10 through 14
- **High**: final grade of 15 or higher

The project combines:

1. Data loading and exploratory analysis
2. Data preprocessing
3. Target variable creation
4. Machine learning model training
5. Model comparison and evaluation
6. Feature importance analysis
7. Model saving
8. A Flask web application
9. A multi-step browser interface

The dataset contains **395 students and 33 original columns**, plus the newly created performance target. It is loaded from `student-mat.csv`.

## 2. Understanding the Dataset

The original dataset includes:

- Personal information: age, sex, address and family size
- Family education and employment
- Travel time and study time
- Past failures
- School, family and paid support
- Extracurricular activities
- Family relationships
- Free time and going out
- Alcohol consumption
- Health
- Absences
- Grades G1, G2 and G3

The notebook begins in `exploration.ipynb`.

### Initial observations

- Dataset size: **395 rows and 33 original columns**
- Numerical columns: **16**
- Categorical/string columns: **17**
- Missing values: **None**
- Memory usage: approximately **102 KB**

After adding the performance category, the number of columns became **34**.

### Target class distribution

- Medium/Average: **192 students**
- Low: **130 students**
- High: **73 students**

This means the dataset is somewhat imbalanced. There are considerably fewer High-performing students than Medium-performing students.

## 3. How You Created the Prediction Target

You converted the numerical final grade `G3` into three categories.

In the final preprocessing logic:

```python
df["performance"] = df["G3"].apply(categorize_grade)
```

The category rules are:

```python
if score >= 15:
    High
elif score >= 10:
    Medium
else:
    Low
```

The notebook uses the bins `-1–9`, `10–14`, and `15–20`, which produce the same three categories.

This taught you an important concept:

> A regression problem can sometimes be converted into a classification problem when the exact result is less important than assigning students to meaningful groups.

## 4. Preventing Data Leakage

You removed:

- `G3`: the final grade used to calculate the target
- `G2`: another final-period grade
- `performance_category`
- `performance`

The remaining feature matrix was created with:

```python
X = df.drop(
    columns=["performance", "performance_category", "G2", "G3"]
)
```

This is an important improvement because it prevents the model from cheating by learning directly from grades that reveal the answer.

The target remained:

```python
y = df["performance"]
```

## 5. Data Preprocessing

### Numerical missing values

Numerical missing values are replaced with the median:

```python
X[numeric_cols] = X[numeric_cols].fillna(
    X[numeric_cols].median()
)
```

### Categorical missing values

Categorical missing values are replaced with the most frequent category:

```python
X[col] = X[col].fillna(X[col].mode()[0])
```

### One-hot encoding

Categorical variables were converted into numerical columns using `pd.get_dummies()`:

```python
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)
```

This increased the feature matrix from **30 features** to **40 encoded features**.

You learned that models cannot work directly with values such as:

- `"yes"` and `"no"`
- `"GP"` and `"MS"`
- `"mother"` and `"father"`

One-hot encoding converts those values into numerical 0/1 columns.

## 6. Data Splitting

You split the data into:

- **80% training data:** 316 records
- **20% test data:** 79 records

The split was performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` ensures that the Low, Medium and High classes are represented proportionally in both sets.

This teaches the fundamental machine learning rule:

> Always evaluate the model on data it has not used during training.

## 7. Feature Scaling

You used `StandardScaler` for the Logistic Regression model:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
```

Scaling changes features to comparable ranges.

However, the final Flask app uses the Random Forest model on unscaled features. Random Forests do not require scaling, so this was the appropriate deployment approach.

The saved `scaler.pkl` is retained as a notebook artifact but is not needed for the Random Forest predictions in the current app.

## 8. Machine Learning Models You Tested

### Logistic Regression

A baseline linear classifier was trained using the scaled data.

Test accuracy:

- **68.35%**

It achieved:

- Low recall: **81%**
- High recall: **67%**
- Medium recall: **61%**

### Decision Tree

A Decision Tree with a maximum depth of five was also trained.

Test accuracy:

- **81.01%**

This was the strongest baseline result on the test set.

### Random Forest

You trained a Random Forest containing:

- **100 trees**
- Maximum depth: **5**
- Random state: **42**

Initial test accuracy:

- **70.89%**

This model was less accurate than the Decision Tree on the held-out test set.

### Random Forest with balanced classes

You then retrained the Random Forest with:

```python
class_weight="balanced"
```

Balanced test accuracy:

- **73.42%**

Compared with the unbalanced Random Forest:

- High recall improved from **47% to 93%**
- Low recall improved from **54% to 92%**
- Medium recall fell from **92% to 53%**

This demonstrates an important trade-off:

> Better balance across one or more categories can reduce performance for another category.

## 9. Cross-Validation

You used five-fold cross-validation to obtain a more dependable estimate.

Results:

- Decision Tree: **69.31%**
- Random Forest: **69.28%**

Cross-validation is more reliable than a single train/test split because it tests the model across five different training and validation combinations.

## 10. Hyperparameter Tuning

You used `GridSearchCV` to search for the best Decision Tree settings.

Tested parameters:

- Maximum depth: 3, 5, 7, 10 or unlimited
- Minimum samples split: 2, 5 or 10

The best settings were:

- `max_depth`: **3**
- `min_samples_split`: **2**

Its five-fold cross-validation accuracy was:

- **72.45%**

This taught you how to tune model parameters instead of relying only on default values.

## 11. Feature Importance

The Random Forest also calculates feature importance.

You created a ranking with:

```python
feature_imp = pd.Series(
    rf_model.feature_importances_,
    index=X_train_encoded.columns
).sort_values(ascending=False)
```

This ranking identifies features that contributed most to the model's decisions.

It is useful because it helps answer:

- Which factors appear most related to performance?
- Where might schools provide the most support?
- Which variables should be examined more carefully?

Feature importance indicates correlation with the model's predictions, not necessarily direct cause and effect.

## 12. Saved Model Files

You saved two Random Forest versions:

- `student_model.pkl`
- `model.pkl`

You also saved:

- `columns.pkl`, containing the exact 40 encoded feature names
- `scaler.pkl`

The Flask app loads `model.pkl` and `columns.pkl`.

The saved model contains:

- **40 input features**
- Classes: **High, Low and Medium**
- A trained `RandomForestClassifier`

## 13. How the Flask Application Works

The web server is implemented in `app.py`.

### Loading the model

The app starts Flask and loads both artifacts:

```python
model = pickle.load(model_file)
feature_columns = pickle.load(columns_file)
```

It also verifies that:

- The model has the expected number of features
- The feature names and column order match

This is a good reliability practice because even a single column mismatch can produce incorrect predictions.

### Collecting student information

The app provides fields for:

- Student and family details
- Study habits and support
- Personal habits
- Absences
- G1 grade

G2 and G3 are intentionally excluded.

### Server-side validation

Every submitted value is validated:

- Required fields cannot be empty
- Numeric fields must be whole numbers
- Numeric fields must remain within valid ranges
- Categorical values must belong to approved categories

This prevents arbitrary or invalid values from reaching the model.

### Encoding the submitted data

The app performs the same one-hot encoding used during training:

```python
pd.get_dummies(
    input_frame,
    columns=CATEGORICAL_FIELDS,
    drop_first=True
)
```

It then rearranges the columns exactly according to `columns.pkl`.

This is essential because the model depends on each feature being in a consistent position.

### Making a prediction

The app requests:

- The predicted class:

  ```python
  model.predict(encoded)
  ```

- The probability of every class:

  ```python
  model.predict_proba(encoded)
  ```

The result includes:

- Predicted category
- Highest class probability
- A probability bar for High, Low and Medium

## 14. Front-End Interface

The interface is in `index.html`.

It provides:

- A responsive layout
- Three form steps
- Progress indicators
- Input validation
- Prefilled example values
- Result category
- Confidence percentage
- Class probability bars
- Mobile support
- Reduced-motion support
- Accessibility labels and live result regions

The browser JavaScript divides the form into three panels:

1. Student and family
2. Study and support
3. Habits and current grades

It validates each step before allowing the user to continue.

## 15. Dependencies

Your `requirements.txt` declares:

- Flask 3.x
- Pandas 2.2–3.x
- Scikit-learn 1.9.1

These packages cover:

- Web server and routing
- Data tables
- Machine learning
- Model evaluation
- Preprocessing
- Model serialization

## 16. What You Have Learned

This project helped you practice:

- Loading semicolon-separated CSV data
- Exploratory data analysis
- Pandas DataFrames
- Creating machine learning targets
- Handling categorical variables
- Median and mode imputation
- One-hot encoding
- Feature and target separation
- Data leakage prevention
- Stratified train/test splitting
- Feature scaling
- Logistic Regression
- Decision Trees
- Random Forests
- Class balancing
- Hyperparameter tuning
- Cross-validation
- Classification reports
- Confusion matrices
- Feature importance
- Pickle serialization
- Flask routes
- HTML forms
- JavaScript step navigation
- Front-end and back-end validation
- Model version consistency
- Probability interpretation
- Deployment preparation

## 17. What the Project Is Useful For

In an educational setting, this project could help identify:

- Students who may need additional support
- Students who may benefit from tutoring
- The effect of absences and study habits
- The possible influence of support services
- Patterns associated with performance categories

The result should function as a **support signal**, not as a final judgment about a student.

## 18. Important Limitations

- The dataset contains **395 records**, which is relatively small.
- It represents a particular educational context.
- Many factors are self-reported or simplified.
- The model may contain bias from the original dataset.
- It predicts a category, not a guaranteed final grade.
- Feature importance does not prove causation.
- Fairness and discrimination require additional checks.
- G3 is excluded, but G1 is still available.
- The test accuracy and cross-validation accuracy are not particularly high.
- The balanced model improves recall for Low and High students but reduces Medium recall.

## 19. How This Helps Your Future Learning

This project gives you a foundation for:

- Building classification projects with pandas
- Comparing and tuning machine learning models
- Creating data pipelines
- Detecting and preventing leakage
- Saving and loading trained models
- Exposing models through APIs
- Building applications with Flask
- Creating interactive prediction interfaces
- Writing more reliable training and deployment code
- Learning model evaluation and fairness
- Building portfolios and interview projects
- Moving from notebooks to production applications

## 20. Best Next Improvements

1. Add unit tests for preprocessing and input validation.
2. Perform a full hyperparameter search for Random Forest.
3. Compare Gradient Boosting or XGBoost.
4. Remove or investigate sensitive variables such as sex or family information.
5. Use cross-validation for model selection.
6. Track several metrics, especially macro F1 and recall.
7. Add a confidence threshold instead of only using the highest probability.
8. Create a proper data pipeline.
9. Persist the training and prediction encoders consistently.
10. Add a database for predictions and reports.
11. Add SHAP explanations for individual predictions.
12. Evaluate performance on separate future student cohorts.
13. Deploy it using Vercel or another hosted Python platform.
14. Convert the notebook into reusable Python modules.

## Run, test, and deploy

### Run locally

Use Python 3.12 and install the project dependencies. Scikit-learn is pinned to
the version used to serialize the model:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m flask --app app run
```

Open <http://127.0.0.1:5000>.

### Run tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs this test command for pushes and pull requests.

### Deploy to Vercel

1. Push this project to a GitHub repository, including `app.py`, `model.pkl`, `columns.pkl`, and `requirements.txt`.
2. In Vercel, import the repository and keep the project root set to the repository root.
3. Deploy. Vercel detects the Flask application in `app.py`, installs dependencies from `requirements.txt`, and uses `.python-version` to select Python 3.12.
4. Open the deployment URL and submit the example form to verify predictions.

The model is loaded from files shipped with the deployment; do not replace these trusted pickle artifacts with user-uploaded files.