"""End-to-end smoke test using one official test image from each class."""

import io

from fastapi.testclient import TestClient
from PIL import Image
import tensorflow as tf

from app import CLASS_NAMES, app


def main():
    (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
    selected = [next(i for i, label in enumerate(labels) if label == class_id)
                for class_id in range(len(CLASS_NAMES))]
    correct = 0

    with TestClient(app) as client:
        for index in selected:
            buffer = io.BytesIO()
            Image.fromarray(images[index]).save(buffer, format="PNG")
            response = client.post(
                "/predict",
                files={"file": ("sample.png", buffer.getvalue(), "image/png")},
            )
            response.raise_for_status()
            predicted = response.json()["prediction"]["label"]
            expected = CLASS_NAMES[int(labels[index])]
            correct += predicted == expected
            print(f"expected={expected:<12} predicted={predicted}")

    print(f"Smoke-test accuracy: {correct}/{len(selected)}")


if __name__ == "__main__":
    main()

