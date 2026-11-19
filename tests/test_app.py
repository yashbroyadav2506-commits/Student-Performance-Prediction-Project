import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

from werkzeug.datastructures import MultiDict

from app import (
    FORM_GROUPS,
    app,
    encode_student_features,
    parse_student_input,
)


class StudentPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.form_data = {
            field["name"]: str(field.get("default", ""))
            for group in FORM_GROUPS
            for field in group["fields"]
        }

    def test_default_input_matches_model_features(self):
        student = parse_student_input(MultiDict(self.form_data))
        encoded = encode_student_features(student)
        with (Path(__file__).resolve().parents[1] / "columns.pkl").open("rb") as columns_file:
            feature_columns = pickle.load(columns_file)

        self.assertEqual(list(encoded.columns), feature_columns)
        self.assertEqual(encoded.shape, (1, len(feature_columns)))

    def test_invalid_input_is_reported_to_the_user(self):
        response = self.client.post("/", data={**self.form_data, "age": "99"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Age must be between 15 and 22.", response.data)

    def test_prediction_error_is_not_reported_as_input_error(self):
        with patch("app.model.predict", side_effect=ValueError("model failure")):
            with self.assertRaisesRegex(ValueError, "model failure"):
                self.client.post("/", data=self.form_data)

    def test_homepage_and_prediction_render(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.post("/", data=self.form_data)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Predicted category", response.data)


if __name__ == "__main__":
    unittest.main()
