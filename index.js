const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const enableWebcamButton = document.getElementById("webcamButton");

// ¡IMPORTANTE! Cambia esto por los nombres de las clases de TU modelo
const CLASS_NAMES = ["ine"]; 

let model;
let children = [];

// 1. Cargar el modelo en formato TFJS
async function loadModel() {
  // Pon la ruta correcta hacia tu archivo model.json
  model = await tf.loadGraphModel("best_web_model/model.json");
  console.log("¡Modelo YOLO cargado con éxito!");
  enableWebcamButton.addEventListener("click", enableCam);
}
loadModel();

// 2. Habilitar la cámara
function enableCam(event) {
  event.target.classList.add("removed");
  const constraints = { video: { width: 640, height: 480 } };

  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

// 3. Predicción y procesamiento de la matriz de YOLO
async function predictWebcam() {
  // Tomar frame, redimensionar a 640x640 y normalizar (0 a 1)
  const input = tf.tidy(() => {
    return tf.browser.fromPixels(video)
      .resizeBilinear([640, 640])
      .div(255.0)
      .expandDims(0);
  });

  // Ejecutar el modelo
  const res = await model.executeAsync(input);
  
  // Decodificar la salida de YOLO (Matriz [1, NumClases+4, 8400])
  const data = await res.data();
  const numCols = res.shape[1]; 
  const numRows = res.shape[2]; // 8400 anclas
  res.dispose(); // Liberar memoria de la GPU
  input.dispose();

  let boxes = [];
  let scores = [];
  let classes = [];

  // Filtrar las mejores predicciones
  for (let i = 0; i < numRows; i++) {
    let maxScore = 0;
    let maxClass = -1;

    // Las probabilidades de las clases empiezan en el índice 4
    for (let c = 4; c < numCols; c++) {
      let score = data[c * numRows + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c - 4;
      }
    }

    if (maxScore > 0.5) { // Nivel de confianza mínimo
      const cx = data[0 * numRows + i];
      const cy = data[1 * numRows + i];
      const w = data[2 * numRows + i];
      const h = data[3 * numRows + i];

      const y1 = cy - h / 2;
      const x1 = cx - w / 2;
      const y2 = cy + h / 2;
      const x2 = cx + w / 2;

      boxes.push([y1, x1, y2, x2]);
      scores.push(maxScore);
      classes.push(maxClass);
    }
  }

  // Limpiar dibujos del frame anterior
  for (let child of children) liveView.removeChild(child);
  children.splice(0);

  // Aplicar Non-Max Suppression (eliminar cajas duplicadas)
  if (boxes.length > 0) {
    const boxesTensor = tf.tensor2d(boxes);
    const scoresTensor = tf.tensor1d(scores);
    
    const nmsIndices = await tf.image.nonMaxSuppressionAsync(boxesTensor, scoresTensor, 20, 0.45, 0.5);
    const selected = await nmsIndices.data();

    for (let i = 0; i < selected.length; i++) {
      const idx = selected[i];
      const box = boxes[idx];
      const cls = classes[idx];
      const score = Math.round(scores[idx] * 100);

      // Escalar coordenadas de 640x640 al tamaño real del video
      const scaleX = video.offsetWidth / 640;
      const scaleY = video.offsetHeight / 640;

      const y1 = box[0] * scaleY;
      const x1 = box[1] * scaleX;
      const width = (box[3] - box[1]) * scaleX;
      const height = (box[2] - box[0]) * scaleY;

      // Dibujar caja
      const highlighter = document.createElement("div");
      highlighter.setAttribute("class", "highlighter");
      highlighter.style = `left: ${x1}px; top: ${y1}px; width: ${width}px; height: ${height}px; position: absolute; border: 3px solid #00FF00; z-index: 1;`;

      // Dibujar etiqueta
      const p = document.createElement("p");
      const className = CLASS_NAMES[cls] || `Clase ${cls}`;
      p.innerText = `${className} - ${score}%`;
      p.style = `left: ${x1}px; top: ${y1 - 25}px; position: absolute; background: #00FF00; color: black; padding: 2px 5px; margin: 0; font-family: sans-serif; font-weight: bold; z-index: 2;`;

      liveView.appendChild(highlighter);
      liveView.appendChild(p);
      children.push(highlighter);
      children.push(p);
    }

    boxesTensor.dispose();
    scoresTensor.dispose();
    nmsIndices.dispose();
  }

  window.requestAnimationFrame(predictWebcam);
}