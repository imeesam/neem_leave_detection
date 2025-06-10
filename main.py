# Remove the `if __name__ == "__main__":` block completely!
# Do NOT call app.run() in production

# Just leave it like this:
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="leaf_model_85_percent.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(io.BytesIO(request.data)).resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    label = "Unhealthy" if score > 0.5 else "Healthy"
    return jsonify({"label": label, "score": score})
