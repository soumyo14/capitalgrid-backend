from tensorflow.keras.models import load_model

model = load_model("models/model_SBIN.h5")
print(f"Expected Input Shape: {model.input_shape}")