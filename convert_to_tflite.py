"""
Run this ONCE locally in your bloodcancer conda env:
    python convert_to_tflite.py

It converts my_blood_cancer_model.keras → model.tflite with dynamic-range
quantisation, reducing the file from ~129 MB to ~32-40 MB and cutting
inference-time RAM from ~400 MB to ~50 MB — compatible with Render free tier.
"""
import tensorflow as tf  # noqa: E402  (requires your local TF install)

MODEL_IN  = "my_blood_cancer_model.keras"
MODEL_OUT = "model.tflite"

print(f"Loading {MODEL_IN} …")

# Same Keras 2→3 compat patch used by app.py
_orig = tf.keras.layers.Dense.__init__
def _compat(self, *a, quantization_config=None, **kw):
    _orig(self, *a, **kw)
tf.keras.layers.Dense.__init__ = _compat
model = tf.keras.models.load_model(MODEL_IN, compile=False)
tf.keras.layers.Dense.__init__ = _orig

print("Converting to TFLite with dynamic-range quantisation …")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(MODEL_OUT, "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / 1024 / 1024
print(f"Saved {MODEL_OUT}  ({size_mb:.1f} MB)")
print("Done — commit model.tflite and push.")
