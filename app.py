import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / "model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

with (BASE_DIR / "columns.pkl").open("rb") as columns_file:
    feature_columns = pickle.load(columns_file)

if model.n_features_in_ != len(feature_columns):
    raise ValueError(
        "model.pkl and columns.pkl do not match: "
        f"the model expects {model.n_features_in_} features, "
        f"but columns.pkl contains {len(feature_columns)}."
    )

if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != feature_columns:
    raise ValueError("columns.pkl is not in the same order used to train model.pkl.")

app = Flask(__name__)

FORM_GROUPS = [
    {
        "title": "Student and family",
        "description": "A few basics about the student's school and home life.",
        "fields": [
            {
                "name": "school",
                "label": "School",
                "options": ["GP", "MS"],
                "option_labels": {
                    "GP": "Gabriel Pereira (GP)",
                    "MS": "Mousinho da Silveira (MS)",
                },
                "default": "GP",
            },
            {
                "name": "sex",
                "label": "Sex",
                "options": ["F", "M"],
                "option_labels": {"F": "Female", "M": "Male"},
                "default": "F",
            },
            {"name": "age", "label": "Age", "min": 15, "max": 22, "default": 16},
            {
                "name": "address",
                "label": "Home area",
                "options": ["U", "R"],
                "option_labels": {"U": "Urban", "R": "Rural"},
                "default": "U",
            },
            {
                "name": "famsize",
                "label": "Family size",
                "options": ["GT3", "LE3"],
                "option_labels": {"GT3": "More than 3 people", "LE3": "3 or fewer people"},
                "default": "GT3",
            },
            {
                "name": "Pstatus",
                "label": "Parents' living situation",
                "options": ["T", "A"],
                "option_labels": {"T": "Living together", "A": "Living apart"},
                "default": "T",
            },
            {"name": "Medu", "label": "Mother's education (0-4)", "min": 0, "max": 4, "default": 2},
            {"name": "Fedu", "label": "Father's education (0-4)", "min": 0, "max": 4, "default": 2},
            {
                "name": "Mjob",
                "label": "Mother's job",
                "options": ["at_home", "health", "other", "services", "teacher"],
                "default": "other",
            },
            {
                "name": "Fjob",
                "label": "Father's job",
                "options": ["at_home", "health", "other", "services", "teacher"],
                "default": "other",
            },
            {
                "name": "reason",
                "label": "Reason for choosing this school",
                "options": ["course", "home", "other", "reputation"],
                "option_labels": {
                    "course": "Course options",
                    "home": "Close to home",
                    "other": "Other reason",
                    "reputation": "School reputation",
                },
                "default": "course",
            },
            {
                "name": "guardian",
                "label": "Guardian",
                "options": ["father", "mother", "other"],
                "default": "mother",
            },
        ],
    },
    {
        "title": "Study and support",
        "description": "Study habits and the support available to the student.",
        "fields": [
            {"name": "traveltime", "label": "Travel time (1-4)", "min": 1, "max": 4, "default": 1},
            {"name": "studytime", "label": "Weekly study time (1-4)", "min": 1, "max": 4, "default": 2},
            {"name": "failures", "label": "Past class failures (0-3)", "min": 0, "max": 3, "default": 0},
            {"name": "schoolsup", "label": "School support", "options": ["no", "yes"], "default": "no"},
            {"name": "famsup", "label": "Family support", "options": ["no", "yes"], "default": "yes"},
            {"name": "paid", "label": "Extra paid classes", "options": ["no", "yes"], "default": "no"},
            {"name": "activities", "label": "Extracurricular activities", "options": ["no", "yes"], "default": "yes"},
            {"name": "nursery", "label": "Attended nursery school", "options": ["no", "yes"], "default": "yes"},
            {"name": "higher", "label": "Wants higher education", "options": ["no", "yes"], "default": "yes"},
            {"name": "internet", "label": "Internet access at home", "options": ["no", "yes"], "default": "yes"},
            {"name": "romantic", "label": "In a romantic relationship", "options": ["no", "yes"], "default": "no"},
        ],
    },
    {
        "title": "Habits and current grades",
        "description": "Daily habits, wellbeing, absences, and the first-period grade.",
        "fields": [
            {"name": "famrel", "label": "Family relationship quality (1-5)", "min": 1, "max": 5, "default": 4},
            {"name": "freetime", "label": "Free time after school (1-5)", "min": 1, "max": 5, "default": 3},
            {"name": "goout", "label": "Going out with friends (1-5)", "min": 1, "max": 5, "default": 3},
            {"name": "Dalc", "label": "Workday alcohol consumption (1-5)", "min": 1, "max": 5, "default": 1},
            {"name": "Walc", "label": "Weekend alcohol consumption (1-5)", "min": 1, "max": 5, "default": 1},
            {"name": "health", "label": "Current health (1-5)", "min": 1, "max": 5, "default": 3},
            {"name": "absences", "label": "School absences (0-75)", "min": 0, "max": 75, "default": 0},
            {"name": "G1", "label": "First-period grade (0-20)", "min": 0, "max": 20, "default": 10},
        ],
    },
]

for group in FORM_GROUPS:
    for field in group["fields"]:
        if "options" in field:
            field.setdefault(
                "option_labels",
                {option: option.replace("_", " ").title() for option in field["options"]},
            )

CATEGORICAL_OPTIONS = {
    field["name"]: field["options"]
    for group in FORM_GROUPS
    for field in group["fields"]
    if "options" in field
}
CATEGORICAL_FIELDS = list(CATEGORICAL_OPTIONS)

PERFORMANCE_DESCRIPTIONS = {
    "Low": "Predicted final grade below 10.",
    "Medium": "Predicted final grade from 10 through 14.",
    "High": "Predicted final grade of 15 or higher.",
}


def parse_student_input(form):
    student = {}
    errors = []

    for group in FORM_GROUPS:
        for field in group["fields"]:
            name = field["name"]
            value = form.get(name, "").strip()
            if not value:
                errors.append(f"{field['label']} is required.")
                continue

            if "options" in field:
                if value not in field["options"]:
                    errors.append(f"Choose a valid value for {field['label'].lower()}.")
                else:
                    student[name] = value
                continue

            try:
                number = int(value)
            except ValueError:
                errors.append(f"{field['label']} must be a whole number.")
                continue

            if not field["min"] <= number <= field["max"]:
                errors.append(
                    f"{field['label']} must be between {field['min']} and {field['max']}."
                )
                continue
            student[name] = number

    if errors:
        raise ValueError(" ".join(errors))
    return student


def encode_student_features(student):
    input_frame = pd.DataFrame([student])
    for name in CATEGORICAL_FIELDS:
        input_frame[name] = pd.Categorical(
            input_frame[name],
            categories=sorted(CATEGORICAL_OPTIONS[name]),
        )

    encoded = pd.get_dummies(
        input_frame,
        columns=CATEGORICAL_FIELDS,
        drop_first=True,
        dtype=float,
    )
    return encoded.reindex(columns=feature_columns, fill_value=0).astype(float)


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = {}
    error = None
    prediction = None
    confidence = None
    probabilities = []

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            student = parse_student_input(request.form)
        except ValueError as exc:
            error = str(exc)
        else:
            encoded = encode_student_features(student)

            # This Random Forest was trained on unscaled encoded features.
            prediction = str(model.predict(encoded)[0])
            class_probabilities = model.predict_proba(encoded)[0]
            probabilities = sorted(
                [
                    {"label": str(label), "value": float(probability)}
                    for label, probability in zip(model.classes_, class_probabilities)
                ],
                key=lambda item: item["value"],
                reverse=True,
            )
            confidence = max(item["value"] for item in probabilities)

    return render_template(
        "index.html",
        form_groups=FORM_GROUPS,
        form_values=form_values,
        error=error,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        description=PERFORMANCE_DESCRIPTIONS,
    )


if __name__ == "__main__":
    app.run()
