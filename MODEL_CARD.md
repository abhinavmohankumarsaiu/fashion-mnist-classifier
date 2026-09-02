# Model card

- **Task:** single-label clothing image classification
- **Dataset:** Fashion-MNIST (60,000 training and 10,000 test images)
- **Input:** normalized 28x28 grayscale tensor with one channel
- **Output:** logits in the official 10-class Fashion-MNIST order
- **Architecture:** two convolution/max-pooling blocks, a 128-unit dense layer,
  30% dropout, and a 10-logit output layer
- **Held-out test accuracy:** 90.88%
- **Artifact:** `fashion_cnn_model.keras`

The model is suitable for demonstrations and coursework. It should not be used
for high-stakes decisions. Real product photos differ from Fashion-MNIST's
catalogue silhouettes, so the API applies polarity correction, foreground
cropping, aspect-ratio preservation, and centering before inference.

