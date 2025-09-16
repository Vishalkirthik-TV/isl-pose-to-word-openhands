const videoEl = document.getElementsByClassName('input_video_ws')[0];
const canvasEl = document.getElementsByClassName('output_ws')[0];
const ctx = canvasEl.getContext('2d');
const predBox = document.getElementById('ws_prediction');

const WS_URL = 'ws://localhost:8000/ws';
let ws;

// Minimal payload: 18 pose points (54 floats). We downsample and avoid hands/face.
function extract18PoseKeypoints(results) {
  const keypoints = [];
  if (!results.poseLandmarks || results.poseLandmarks.length < 33) {
    for (let i = 0; i < 18; i++) keypoints.push(0, 0, 0);
    return keypoints;
  }
  // Use first 18 landmarks (0..17). Cheap and consistent.
  for (let i = 0; i < 18; i++) {
    const lm = results.poseLandmarks[i];
    if (lm && (lm.visibility ?? 1) > 0.3) keypoints.push(lm.x, lm.y, lm.z || 0);
    else keypoints.push(0, 0, 0);
  }
  return keypoints;
}

// Throttle sending frames to WS to ~10 Hz to reduce CPU/network
let lastSent = 0;
const SEND_INTERVAL_MS = 100;

async function onResults(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (results.image) ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  // Draw only pose (cheaper than face+hands)
  if (results.poseLandmarks) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00'});
    drawLandmarks(ctx, results.poseLandmarks, {color: '#00FF00'});
  }

  const now = performance.now();
  if (ws && ws.readyState === WebSocket.OPEN && (now - lastSent) >= SEND_INTERVAL_MS) {
    lastSent = now;
    const keypoints = extract18PoseKeypoints(results);
    // Quick validity check
    let nonZero = 0; for (const v of keypoints) if (Math.abs(v) > 1e-2) nonZero++;
    if (nonZero >= 6) {
      ws.send(JSON.stringify({ keypoints }));
    }
  }
  ctx.restore();
}

function showPrediction(prediction, confidence) {
  predBox.style.display = 'block';
  predBox.textContent = `${prediction} (${(confidence * 100).toFixed(1)}%)`;
}

function setupWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { console.log('WS connected'); };
  ws.onclose = () => { console.log('WS closed'); setTimeout(setupWS, 1000); };
  ws.onerror = (e) => { console.warn('WS error', e); };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.prediction) showPrediction(msg.prediction, msg.confidence || 0);
    } catch (_) {}
  };
}

function main() {
  setupWS();
  const holistic = new Holistic({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/${f}` });
  holistic.setOptions({
    modelComplexity: 0, // light model
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    refineFaceLandmarks: false
  });
  holistic.onResults(onResults);

  const cam = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 320,
    height: 240
  });
  cam.start();
}

window.addEventListener('load', main);


