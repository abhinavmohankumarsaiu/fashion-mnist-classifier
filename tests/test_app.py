import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import app as application


class DummyModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 28, 28, 1)
        logits = np.zeros((1, 10), dtype=np.float32)
        logits[0, 9] = 5.0
        return logits


def image_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_official_label_order():
    assert application.CLASS_NAMES[0] == "T-shirt/top"
    assert application.CLASS_NAMES[6] == "Shirt"
    assert application.CLASS_NAMES[9] == "Ankle boot"


def test_preprocessing_inverts_white_background_and_centers_item():
    image = Image.new("RGB", (180, 120), "white")
    ImageDraw.Draw(image).rectangle((55, 20, 125, 100), fill="black")
    batch = application.preprocess_image(image)

    assert batch.shape == (1, 28, 28, 1)
    assert batch.dtype == np.float32
    assert float(batch[0, 14, 14, 0]) > 0.8
    assert float(batch[0, 0, 0, 0]) == 0.0


def test_predict_contract(monkeypatch):
    monkeypatch.setattr(application, "load_classifier", lambda: DummyModel())
    image = Image.new("L", (28, 28), 0)
    ImageDraw.Draw(image).rectangle((5, 4, 22, 25), fill=220)

    with TestClient(application.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.png", image_bytes(image), "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"]["label"] == "Ankle boot"
    assert len(payload["alternatives"]) == 2
    assert payload["predictions"][0]["class"] == "Ankle boot"


def test_rejects_non_image(monkeypatch):
    monkeypatch.setattr(application, "load_classifier", lambda: DummyModel())
    with TestClient(application.app) as client:
        response = client.post(
            "/api/predict",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 400

