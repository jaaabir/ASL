from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os 

app = FastAPI()
# model_path = os.path.join(os.path.dirname(__file__), 'models', model_name := 'temporal_transformer_b_best_script.pt')
model_path = 'temporal_transformer_b_best_script.pt'
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')
# === Load the TorchScript scripted model ===
try:
    model = torch.jit.load(model_path).to(device).eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# === Define Input Schema ===
class KeypointInput(BaseModel):
    data: list[list[float]]  # Expecting shape (T=16, F=144)


# === GET endpoint ===
@app.get("/")
def root():
    return {"message": "hello world"}


# === POST endpoint ===
@app.post("/detect")
def detect(input_data: KeypointInput):
    try:
        arr = np.array(input_data.data, dtype=np.float32)
        if arr.shape != (16, 144):
            raise ValueError("Input must be of shape (16, 144)")

        x = torch.tensor(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x)
        pred = torch.argmax(output, dim=1).item()
        return {"prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
