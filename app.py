
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

interpreter = tf.lite.Interpreter(model_path="leaf_model_85_percent.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        input_tensor = preprocess(content)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = "Unhealthy" if output > 0.5 else "Healthy"
        return JSONResponse({"label": label, "confidence": float(output)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
