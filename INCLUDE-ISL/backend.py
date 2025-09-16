# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import List, Dict, Any

# Load label map
def load_label_map():
    label_map_path = "label_maps/label_map_include.json"
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    # Create reverse mapping (index to word)
    index_to_word = {v: k for k, v in label_map.items()}
    return label_map, index_to_word

# BERT Model Architecture (matching the checkpoint structure)
class BERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Decoder architecture matching the checkpoint
        self.decoder = nn.ModuleDict({
            'l1': nn.Linear(config['num_points'] * 3, config['hidden_size']),
            'embedding': nn.ModuleDict({
                'position_embeddings': nn.Embedding(config['max_position_embeddings'], config['hidden_size']),
                'LayerNorm': nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps']),
                'dropout': nn.Dropout(config['hidden_dropout_prob'])
            }),
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'attention': nn.ModuleDict({
                        'self': nn.ModuleDict({
                            'query': nn.Linear(config['hidden_size'], config['hidden_size']),
                            'key': nn.Linear(config['hidden_size'], config['hidden_size']),
                            'value': nn.Linear(config['hidden_size'], config['hidden_size']),
                        }),
                        'output': nn.ModuleDict({
                            'dense': nn.Linear(config['hidden_size'], config['hidden_size']),
                            'LayerNorm': nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps']),
                        })
                    }),
                    'intermediate': nn.ModuleDict({
                        'dense': nn.Linear(config['hidden_size'], config['intermediate_size']),
                    }),
                    'output': nn.ModuleDict({
                        'dense': nn.Linear(config['intermediate_size'], config['hidden_size']),
                        'LayerNorm': nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps']),
                    })
                }) for _ in range(config['num_hidden_layers'])
            ]),
            'l2': nn.Linear(config['hidden_size'], config['num_classes'])
        })
        
        # CLS parameter (learnable parameter for classification)
        if config['cls_token']:
            self.cls_param = nn.Parameter(torch.randn(config['hidden_size']))
        
        # Position IDs buffer
        self.register_buffer('position_ids', torch.arange(config['max_position_embeddings']).unsqueeze(0))
        
    def forward(self, x):
        batch_size, num_frames, num_points, coords = x.shape
        
        # Flatten pose data
        x = x.view(batch_size, num_frames, -1)  # (batch_size, num_frames, num_points * coords)
        
        # Linear projection
        x = self.decoder['l1'](x)  # (batch_size, num_frames, hidden_size)
        
        # Add CLS token if configured
        if hasattr(self, 'cls_param'):
            cls_tokens = self.cls_param.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            num_frames += 1
        
        # Add position embeddings
        position_ids = self.position_ids[:, :num_frames]
        position_embeddings = self.decoder['embedding']['position_embeddings'](position_ids)
        x = x + position_embeddings
        
        # Apply layer norm and dropout
        x = self.decoder['embedding']['LayerNorm'](x)
        x = self.decoder['embedding']['dropout'](x)
        
        # Pass through transformer layers
        for layer in self.decoder['layers']:
            # Self-attention
            query = layer['attention']['self']['query'](x)
            key = layer['attention']['self']['key'](x)
            value = layer['attention']['self']['value'](x)
            
            # Attention computation (simplified)
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.config['hidden_size'] ** 0.5)
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_probs, value)
            
            # Attention output
            attention_output = layer['attention']['output']['dense'](attention_output)
            attention_output = layer['attention']['output']['LayerNorm'](attention_output + x)
            
            # Feed forward
            intermediate_output = torch.relu(layer['intermediate']['dense'](attention_output))
            layer_output = layer['output']['dense'](intermediate_output)
            layer_output = layer['output']['LayerNorm'](layer_output + attention_output)
            
            x = layer_output
        
        # Use CLS token for classification (or average pooling)
        if hasattr(self, 'cls_param'):
            cls_output = x[:, 0]  # CLS token output
        else:
            cls_output = torch.mean(x, dim=1)  # Average pooling
        
        # Classification
        logits = self.decoder['l2'](cls_output)
        
        return logits

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load label mappings
label_map, index_to_word = load_label_map()
num_classes = len(label_map)

# Model configuration from config.yaml and checkpoint analysis
model_config = {
    'num_points': 18,  # Based on checkpoint: 54 / 3 = 18 points
    'max_position_embeddings': 121,
    'layer_norm_eps': 1e-12,
    'hidden_dropout_prob': 0.1,
    'hidden_size': 96,
    'num_attention_heads': 6,
    'num_hidden_layers': 6,
    'cls_token': True,
    'num_classes': num_classes,
    'intermediate_size': 3072  # Based on checkpoint: 96 * 32 = 3072
}

# Load the BERT model
def load_bert_model():
    checkpoint_path = "../include_bert/include/bert/epoch=328-step=18094.ckpt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint (PyTorch Lightning format)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = BERTClassifier(model_config)
    
    # Load model state dict (PyTorch Lightning format)
    if 'state_dict' in checkpoint:
        # Extract model weights from PyTorch Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from keys to match our model structure
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                # Handle special cases for cls_param and position_ids
                if new_key == 'decoder.cls_param':
                    new_key = 'cls_param'
                elif new_key == 'decoder.embedding.position_ids':
                    new_key = 'position_ids'
                model_state_dict[new_key] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

# Load model
try:
    bert_model = load_bert_model()
    print(f"Model loaded successfully with {num_classes} classes")
except Exception as e:
    print(f"Error loading model: {e}")
    bert_model = None

class KeypointsRequest(BaseModel):
    keypoints: List[float]
    num_frames: int = 1

@app.get("/")
async def root():
    return {"message": "ISL Prediction API", "status": "running", "model_loaded": bert_model is not None}

@app.get("/classes")
async def get_classes():
    return {"classes": list(label_map.keys()), "total": len(label_map)}

@app.post("/predict")
async def predict(request: KeypointsRequest):
    if bert_model is None:
        return {"error": "Model not loaded", "prediction": None}
    
    try:
        # Convert keypoints to tensor
        keypoints = np.array(request.keypoints, dtype=np.float32)
        
        # Reshape to (batch_size, num_frames, num_points, coords)
        # Based on checkpoint: 18 points with 3 coordinates (x, y, z)
        expected_points = 18
        expected_coords = 3
        
        # Calculate actual number of frames from keypoints
        total_points = len(keypoints) // expected_coords
        actual_frames = total_points // expected_points
        
        if total_points % expected_points != 0:
            return {"error": f"Invalid keypoints format. Expected {expected_points} points per frame, got {total_points} total points", "prediction": None}
        
        # Reshape keypoints
        keypoints = keypoints.reshape(1, actual_frames, expected_points, expected_coords)
        
        # Convert to tensor
        input_tensor = torch.tensor(keypoints, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            logits = bert_model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Get prediction text
        prediction_text = index_to_word.get(predicted_class, f"class_{predicted_class}")
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=-1)
        top5_predictions = []
        for i in range(5):
            idx = top5_indices[0, i].item()
            prob = top5_probs[0, i].item()
            word = index_to_word.get(idx, f"class_{idx}")
            top5_predictions.append({"word": word, "confidence": prob, "class_id": idx})
        
        return {
            "prediction": prediction_text,
            "confidence": confidence,
            "class_id": predicted_class,
            "top5_predictions": top5_predictions,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "prediction": None}

# ------------------------
# WebSocket for realtime predictions
# ------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if bert_model is None:
                await websocket.send_json({"error": "Model not loaded"})
                continue

            keypoints = data.get("keypoints", [])
            try:
                np_kp = np.array(keypoints, dtype=np.float32)
                expected_points = 18
                expected_coords = 3

                total_points = len(np_kp) // expected_coords
                if total_points % expected_points != 0:
                    await websocket.send_json({"error": "Invalid keypoints length"})
                    continue

                frames = total_points // expected_points
                np_kp = np_kp.reshape(1, frames, expected_points, expected_coords)
                input_tensor = torch.tensor(np_kp, dtype=torch.float32)

                with torch.no_grad():
                    logits = bert_model(input_tensor)
                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(logits, dim=-1).item()
                    confidence = float(probabilities[0, predicted_class].item())

                prediction_text = index_to_word.get(predicted_class, f"class_{predicted_class}")

                await websocket.send_json({
                    "prediction": prediction_text,
                    "class_id": predicted_class,
                    "confidence": confidence
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)