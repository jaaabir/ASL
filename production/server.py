from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from collections import Counter
from typing import Literal
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
    data: list[list[list[float]]]  # Expecting shape (B=n, T=16, F=144)
    score: Literal["hard", "soft"] 


# === GET endpoint ===
@app.get("/")
def root():
    return {"message": "hello world"}


# === POST endpoint ===
@app.post("/detect")
def detect(input_data: KeypointInput):
    try:
        score = input_data.score
        arr = np.array(input_data.data, dtype=np.float32)
        x = torch.tensor(arr).to(device)
        print(input_data)
        print(x.shape)
        with torch.no_grad():
            output = model(x)
        if score == 'hard':
            pred = torch.argmax(output, dim=1)
            pred = Counter(pred.tolist()).most_common(1)[0][0]
        if score == 'soft':
            pred = torch.mean(output, dim=0).argmax().item()

        print(pred)
        return {"prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
