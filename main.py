from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

interpreter = tf.lite.Interpreter(model_path="leaf_model_85_percent.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict")
async def predict(request: Request):
    image_data = await request.body()
    img = Image.open(io.BytesIO(image_data)).resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, 0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    label = "Unhealthy" if score > 0.5 else "Healthy"
    return JSONResponse(content={"label": label, "score": score})
