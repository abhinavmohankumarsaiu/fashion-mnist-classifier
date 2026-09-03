# Fashion-MNIST CNN Classifier

A complete FastAPI deployment for the supplied Fashion-MNIST CNN notebook and
trained model. It serves predictions and includes the responsive FashionAI chat
frontend in the same Render application.

## Correct label mapping

The model uses the official integer order from the dataset:

`0 T-shirt/top`, `1 Trouser`, `2 Pullover`, `3 Dress`, `4 Coat`, `5 Sandal`,
`6 Shirt`, `7 Sneaker`, `8 Bag`, `9 Ankle boot`.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`, or use the API docs at
`http://127.0.0.1:8000/docs`.

## API

- `GET /health` reports readiness and the class mapping.
- `POST /predict` and `POST /api/predict` accept a multipart image in `file`.
- Maximum upload size: 10 MB.

The API returns the top prediction and two alternatives with confidence values
between 0 and 1. A legacy percentage-based `predictions` list is included for
the basic frontend in this repository.

## Verification

The locally trained model reached **90.88% test accuracy**. Run the automated API tests
and the one-sample-per-class end-to-end smoke test with:

```bash
python -m pytest -q
python verify_model.py
```

The saved model's SHA-256 is
`E3BD0D2DF600C22EC9F36B74CA89928195DDB9BDC088EBECAB3390A9C8A5C154`.

## Important accuracy note

Fashion-MNIST was trained on centered 28x28 grayscale catalogue silhouettes,
not arbitrary colour photographs. The API therefore inverts light backgrounds,
crops the foreground, preserves aspect ratio, and centers the item before
inference. This substantially improves product-photo compatibility, but the
most reliable results still come from one centered item on a plain background.

## Deploy on Render

Push the repository, create a Blueprint from `render.yaml`, and wait for the
build to install the application dependencies. The trained
`fashion_cnn_model.keras` file is already included at the repository root, as
required by the assignment. Then wait for `/health` to return `"ready": true`.

```text
NEXT_PUBLIC_PREDICT_ENDPOINT=https://YOUR-SERVICE.onrender.com/predict
```
