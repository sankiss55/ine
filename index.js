import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const demosSection = document.getElementById("demos");
const enableWebcamButton = document.getElementById("webcamButton");

let objectDetector;
let runningMode = "VIDEO";

// 1. Inicializar el detector de objetos
const initializeObjectDetector = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  
  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "best.tflite", // ASEGÚRATE QUE EL ARCHIVO ESTÉ EN ESTA CARPETA
      delegate: "GPU"
    },
    scoreThreshold: 0.5,
    runningMode: runningMode
  });
  
  demosSection.classList.remove("invisible");
};

initializeObjectDetector();

// 2. Revisar soporte de Webcam
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("Tu navegador no soporta getUserMedia()");
}

// 3. Habilitar Cámara
async function enableCam(event) {
  if (!objectDetector) {
    alert("El modelo aún está cargando...");
    return;
  }

  event.target.classList.add("removed");

  const constraints = { video: true };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let children = [];

// 4. Predicción en Tiempo Real
async function predictWebcam() {
  let startTimeMs = performance.now();

  // Detectar objetos en el frame del video
  const detections = await objectDetector.detectForVideo(video, startTimeMs);

  // Limpiar dibujos anteriores
  for (let child of children) {
    liveView.removeChild(child);
  }
  children.splice(0);

  // Dibujar resultados
  for (let detection of detections.detections) {
    const p = document.createElement("p");
    p.innerText = 
      detection.categories[0].categoryName + 
      " - " + Math.round(parseFloat(detection.categories[0].score) * 100) + "%";
    
    // Posicionamiento de la etiqueta
    p.style = `
      left: ${detection.boundingBox.originX}px; 
      top: ${detection.boundingBox.originY - 30}px; 
      width: ${detection.boundingBox.width}px;
      position: absolute;
      background-color: #007bff;
      color: white;
      margin: 0;
      padding: 5px;
      font-size: 12px;
      z-index: 2;
    `;

    const highlighter = document.createElement("div");
    highlighter.setAttribute("class", "highlighter");
    highlighter.style = `
      left: ${detection.boundingBox.originX}px; 
      top: ${detection.boundingBox.originY}px; 
      width: ${detection.boundingBox.width}px; 
      height: ${detection.boundingBox.height}px;
      position: absolute;
      border: 3px solid #007bff;
      z-index: 1;
    `;

    liveView.appendChild(highlighter);
    liveView.appendChild(p);
    children.push(highlighter);
    children.push(p);
  }

  window.requestAnimationFrame(predictWebcam);
}