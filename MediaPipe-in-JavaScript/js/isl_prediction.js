const video1 = document.getElementsByClassName('input_video1')[0];
const out1 = document.getElementsByClassName('output1')[0];
const controlsElement1 = document.getElementsByClassName('control1')[0];
const canvasCtx1 = out1.getContext('2d');

const fpsControl = new FPS();
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

// Prediction state management
let predictionHistory = [];
let lastPredictionTime = 0;
const PREDICTION_COOLDOWN = 500; // ms between predictions
const MAX_HISTORY = 10;

// UI elements
const predictionBox = document.getElementById('predictionBox');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const topPredictions = document.getElementById('topPredictions');
const loadingIndicator = document.getElementById('loadingIndicator');

// Backend API configuration
const API_BASE_URL = 'http://localhost:8000';

function removeElements(landmarks, elements) {
  for (const element of elements) {
    delete landmarks[element];
  }
}

function removeLandmarks(results) {
  if (results.poseLandmarks) {
    // Remove unnecessary landmarks to match the minimal 27 point configuration
    removeElements(
        results.poseLandmarks,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
  }
}

function extractMinimalKeypoints(results) {
  const keypoints = [];
  
  // Extract pose landmarks (filtered to 18 points)
  // Based on the model checkpoint which expects 18 points (54 features total)
  if (results.poseLandmarks) {
    // Select specific pose landmarks that are most relevant for ISL
    const selectedIndices = [
      11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
    ]; // Adjust these indices based on MediaPipe pose landmarks
    
    selectedIndices.forEach(index => {
      if (results.poseLandmarks[index]) {
        const lm = results.poseLandmarks[index];
        keypoints.push(lm.x, lm.y, lm.z || 0);
      } else {
        keypoints.push(0, 0, 0); // Pad with zeros if landmark not available
      }
    });
  }
  
  // If we don't have enough pose landmarks, pad with zeros
  while (keypoints.length < 18 * 3) {
    keypoints.push(0, 0, 0);
  }
  
  // Take only the first 18 points (54 values total)
  return keypoints.slice(0, 54);
}

async function getISLPrediction(keypoints) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ 
        keypoints: keypoints,
        num_frames: 1
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Prediction error:', error);
    return { 
      error: error.message, 
      prediction: null,
      confidence: 0
    };
  }
}

function updatePredictionDisplay(predictionData) {
  if (predictionData.error) {
    predictionText.textContent = `Error: ${predictionData.error}`;
    predictionText.className = 'prediction-text error-text';
    confidenceText.textContent = '';
    topPredictions.innerHTML = '';
    return;
  }

  if (predictionData.prediction) {
    predictionText.textContent = predictionData.prediction;
    predictionText.className = 'prediction-text';
    confidenceText.textContent = `Confidence: ${(predictionData.confidence * 100).toFixed(1)}%`;
    
    // Display top 5 predictions
    if (predictionData.top5_predictions) {
      topPredictions.innerHTML = '<strong>Top 5:</strong><br>';
      predictionData.top5_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.textContent = `${index + 1}. ${pred.word} (${(pred.confidence * 100).toFixed(1)}%)`;
        topPredictions.appendChild(item);
      });
    }
  }
}

function addToHistory(prediction) {
  predictionHistory.push({
    prediction: prediction.prediction,
    confidence: prediction.confidence,
    timestamp: Date.now()
  });
  
  if (predictionHistory.length > MAX_HISTORY) {
    predictionHistory.shift();
  }
}

function getStablePrediction() {
  if (predictionHistory.length < 3) return null;
  
  // Get the most recent predictions
  const recent = predictionHistory.slice(-3);
  
  // Check if we have consistent predictions
  const predictions = recent.map(p => p.prediction);
  const uniquePredictions = [...new Set(predictions)];
  
  if (uniquePredictions.length === 1) {
    // All recent predictions are the same
    return recent[recent.length - 1];
  }
  
  return null;
}

async function onResultsHolistic(results) {
  document.body.classList.add('loaded');
  removeLandmarks(results);
  fpsControl.tick();

  canvasCtx1.save();
  canvasCtx1.clearRect(0, 0, out1.width, out1.height);
  canvasCtx1.drawImage(
      results.image, 0, 0, out1.width, out1.height);
  
  // Draw pose landmarks
  canvasCtx1.lineWidth = 5;
  drawConnectors(
      canvasCtx1, results.poseLandmarks, POSE_CONNECTIONS,
      {color: '#00FF00'});
  drawLandmarks(
      canvasCtx1, results.poseLandmarks,
      {color: '#00FF00', fillColor: '#FF0000'});
  
  // Draw hand landmarks
  if (results.rightHandLandmarks) {
    drawConnectors(
        canvasCtx1, results.rightHandLandmarks, HAND_CONNECTIONS,
        {color: '#00CC00'});
    drawLandmarks(
        canvasCtx1, results.rightHandLandmarks, {
          color: '#00FF00',
          fillColor: '#FF0000',
          lineWidth: 2,
          radius: (data) => {
            return lerp(data.from.z, -0.15, .1, 10, 1);
          }
        });
  }
  
  if (results.leftHandLandmarks) {
    drawConnectors(
        canvasCtx1, results.leftHandLandmarks, HAND_CONNECTIONS,
        {color: '#CC0000'});
    drawLandmarks(
        canvasCtx1, results.leftHandLandmarks, {
          color: '#FF0000',
          fillColor: '#00FF00',
          lineWidth: 2,
          radius: (data) => {
            return lerp(data.from.z, -0.15, .1, 10, 1);
          }
        });
  }
  
  // Draw face landmarks
  if (results.faceLandmarks) {
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_TESSELATION,
        {color: '#C0C0C070', lineWidth: 1});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_RIGHT_EYE,
        {color: '#FF3030'});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_RIGHT_EYEBROW,
        {color: '#FF3030'});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_LEFT_EYE,
        {color: '#30FF30'});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_LEFT_EYEBROW,
        {color: '#30FF30'});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_FACE_OVAL,
        {color: '#E0E0E0'});
    drawConnectors(
        canvasCtx1, results.faceLandmarks, FACEMESH_LIPS,
        {color: '#E0E0E0'});
  }

  // ISL Prediction Logic
  const currentTime = Date.now();
  if (currentTime - lastPredictionTime >= PREDICTION_COOLDOWN) {
    const keypoints = extractMinimalKeypoints(results);
    
    // Only predict if we have valid keypoints
    const hasValidKeypoints = keypoints.some(kp => kp !== 0);
    
    if (hasValidKeypoints) {
      lastPredictionTime = currentTime;
      
      // Show loading indicator
      loadingIndicator.style.display = 'block';
      predictionBox.style.display = 'block';
      
      try {
        const predictionData = await getISLPrediction(keypoints);
        
        if (predictionData.prediction && predictionData.confidence > 0.1) {
          addToHistory(predictionData);
          const stablePrediction = getStablePrediction();
          
          if (stablePrediction) {
            updatePredictionDisplay(stablePrediction);
          } else {
            updatePredictionDisplay(predictionData);
          }
        } else {
          updatePredictionDisplay({ error: 'Low confidence prediction' });
        }
      } catch (error) {
        updatePredictionDisplay({ error: 'Prediction failed' });
      } finally {
        loadingIndicator.style.display = 'none';
      }
    }
  }

  canvasCtx1.restore();
}

// Initialize MediaPipe Holistic
const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/${file}`;
}});

holistic.onResults(onResultsHolistic);

// Initialize camera
const camera = new Camera(video1, {
  onFrame: async () => {
    await holistic.send({image: video1});
  },
  width: 480,
  height: 480
});

// Start camera
camera.start();

// Initialize control panel
new ControlPanel(controlsElement1, {
      selfieMode: true,
      upperBodyOnly: false,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    .add([
      new StaticText({title: 'ISL Prediction with MediaPipe Holistic'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Toggle({title: 'Upper-body Only', field: 'upperBodyOnly'}),
      new Toggle(
          {title: 'Smooth Landmarks', field: 'smoothLandmarks'}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    .on(options => {
      video1.classList.toggle('selfie', options.selfieMode);
      holistic.setOptions(options);
    });

// Check backend connection on load
window.addEventListener('load', async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    console.log('Backend status:', data);
    
    if (!data.model_loaded) {
      predictionText.textContent = 'Backend model not loaded';
      predictionText.className = 'prediction-text error-text';
      predictionBox.style.display = 'block';
    } else {
      predictionText.textContent = 'Ready for ISL prediction';
      predictionText.className = 'prediction-text';
      predictionBox.style.display = 'block';
    }
  } catch (error) {
    console.error('Backend connection failed:', error);
    predictionText.textContent = 'Backend connection failed';
    predictionText.className = 'prediction-text error-text';
    predictionBox.style.display = 'block';
  }
});
