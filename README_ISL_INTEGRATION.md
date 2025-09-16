# ISL Real-time Prediction Integration

This project integrates a trained BERT model for Indian Sign Language (ISL) prediction with a MediaPipe-based frontend for real-time gesture recognition.

## Overview

The system consists of:
- **Backend**: FastAPI server with BERT model for ISL prediction
- **Frontend**: MediaPipe Holistic detection with real-time prediction display
- **Model**: Pre-trained BERT model trained on INCLUDE dataset (263 ISL signs)

## Quick Start

### 1. Start the Backend Server

```bash
# Make the script executable (if not already done)
chmod +x start_backend.sh

# Start the backend server
./start_backend.sh
```

The server will start on `http://localhost:8000`

### 2. Open the Frontend

Open your web browser and navigate to:
```
file:///path/to/MediaPipe-in-JavaScript/isl_prediction.html
```

Or serve it using a local web server:
```bash
cd MediaPipe-in-JavaScript
python -m http.server 8080
# Then open http://localhost:8080/isl_prediction.html
```

### 3. Use the System

1. Allow camera access when prompted
2. Perform Indian Sign Language gestures in front of the camera
3. The system will predict the gesture in real-time
4. Predictions appear in the top-left corner with confidence scores

## Architecture

### Backend (`INCLUDE-ISL/backend.py`)
- **FastAPI server** with CORS enabled
- **BERT model** loaded from checkpoint
- **Pose preprocessing** to match model input format
- **Real-time prediction** endpoint at `/predict`

### Frontend (`MediaPipe-in-JavaScript/isl_prediction.html`)
- **MediaPipe Holistic** for pose, hand, and face detection
- **Real-time keypoint extraction** (27 points)
- **API communication** with backend
- **Prediction display** with confidence scores
- **Stable prediction** filtering to reduce noise

### Model Configuration
- **Input**: 27 pose keypoints (x, y, z coordinates)
- **Architecture**: BERT-style transformer with pose flattener
- **Classes**: 263 ISL signs
- **Output**: Word prediction with confidence scores

## API Endpoints

### GET `/`
Returns server status and model loading status.

### GET `/classes`
Returns list of all supported ISL signs.

### POST `/predict`
Predicts ISL sign from pose keypoints.

**Request:**
```json
{
  "keypoints": [0.1, 0.2, 0.3, ...],  // 81 values (27 points × 3 coords)
  "num_frames": 1
}
```

**Response:**
```json
{
  "prediction": "hello",
  "confidence": 0.95,
  "class_id": 100,
  "top5_predictions": [
    {"word": "hello", "confidence": 0.95, "class_id": 100},
    {"word": "good", "confidence": 0.03, "class_id": 84},
    ...
  ],
  "status": "success"
}
```

## Supported ISL Signs

The system supports 263 Indian Sign Language signs including:
- Basic words: "hello", "thank you", "good", "bad", "yes", "no"
- Colors: "red", "blue", "green", "yellow", "black", "white"
- Numbers: "one", "two", "three", etc.
- Family: "father", "mother", "brother", "sister"
- Objects: "book", "car", "house", "computer"
- And many more...

See `INCLUDE-ISL/label_maps/label_map_include.json` for the complete list.

## Technical Details

### Model Architecture
- **Pose Flattener**: Converts 3D keypoints to sequence
- **BERT Encoder**: 6 layers, 6 attention heads, 96 hidden size
- **Classification Head**: 263-class output
- **Position Embeddings**: For temporal sequence modeling

### Keypoint Processing
- **Input**: MediaPipe Holistic landmarks
- **Filtering**: 27 minimal pose points (removes face/hand details)
- **Normalization**: Center and scale normalization
- **Format**: [x1, y1, z1, x2, y2, z2, ...] for 27 points

### Real-time Optimization
- **Prediction Cooldown**: 500ms between predictions
- **Stable Prediction**: Requires 3 consecutive same predictions
- **Confidence Filtering**: Only shows predictions > 10% confidence
- **Top-5 Display**: Shows alternative predictions

## Troubleshooting

### Backend Issues
1. **Model not loading**: Check if `include_bert/include/bert/epoch=328-step=18094.ckpt` exists
2. **Port already in use**: Change port in `backend.py` or kill existing process
3. **Dependencies missing**: Run `pip install -r requirements.txt` in INCLUDE-ISL directory

### Frontend Issues
1. **Camera not working**: Check browser permissions and HTTPS requirements
2. **No predictions**: Verify backend is running on `localhost:8000`
3. **CORS errors**: Backend includes CORS middleware, check browser console

### Performance Issues
1. **Slow predictions**: Model runs on CPU, consider GPU acceleration
2. **High latency**: Reduce prediction frequency or optimize model
3. **Memory usage**: Monitor system resources during extended use

## File Structure

```
ISL-Testings/
├── include_bert/
│   └── include/bert/
│       ├── config.yaml
│       └── epoch=328-step=18094.ckpt
├── INCLUDE-ISL/
│   ├── backend.py                 # FastAPI server
│   ├── label_maps/
│   │   └── label_map_include.json # ISL sign mappings
│   └── requirements.txt
├── MediaPipe-in-JavaScript/
│   ├── isl_prediction.html        # Main prediction interface
│   ├── js/isl_prediction.js       # Frontend logic
│   └── index.html                 # Navigation
└── start_backend.sh               # Startup script
```

## Development

### Adding New Features
1. **New prediction modes**: Modify `isl_prediction.js`
2. **Different models**: Update `backend.py` model loading
3. **Enhanced UI**: Edit `isl_prediction.html` styles

### Model Retraining
1. Use the training scripts in `INCLUDE-ISL/`
2. Update model configuration in `backend.py`
3. Replace checkpoint file in `include_bert/`

## License

This project uses:
- MediaPipe (Google)
- FastAPI (Python web framework)
- PyTorch (Deep learning)
- Bulma (CSS framework)

Please refer to individual component licenses for usage terms.
