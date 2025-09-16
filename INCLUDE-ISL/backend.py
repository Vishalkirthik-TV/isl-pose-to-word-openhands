from fastapi import FastAPI
from pydantic import BaseModel
import torch
from models.transformer import Transformer
from configs import TransformerConfig
import uvicorn

app = FastAPI()

# Load transformer model and config
config = TransformerConfig()
model = Transformer(config)
model.load_state_dict(torch.load('path_to_pretrained_transformer.pth', map_location='cpu'))  # Update path
model.eval()

class KeypointsRequest(BaseModel):
    keypoints: list

@app.post('/predict')
def predict(request: KeypointsRequest):
    keypoints = request.keypoints
    # Adapt input shape as needed for your model
    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return { 'prediction': int(pred) }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
