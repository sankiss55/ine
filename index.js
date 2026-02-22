let model;
const status = document.getElementById('status');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 1. Configurar Motor (GPU vs CPU) y Cargar Modelo
async function loadModel() {
    try {
        // Intentamos usar WebGL (GPU)
        await tf.ready();
        console.log("Backend actual:", tf.getBackend());
        
        const modelUrl = "https://sankiss55.github.io/ine/best_web_model/model.json";
        
        // Cargamos el modelo
        status.innerText = "Cargando modelo... por favor espera.";
        model = await tf.loadGraphModel(modelUrl);
        
        status.innerText = "¡Modelo listo! Sube una imagen.";
        fileInput.disabled = false;
    } catch (err) {
        status.innerText = "Error al iniciar: " + err.message;
        console.error(err);
    }
}

// 2. Manejador de archivo
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (f) => {
        imagePreview.src = f.target.result;
        // Cuando la imagen termine de cargar visualmente, disparamos la IA
        imagePreview.onload = () => predict();
    };
    reader.readAsDataURL(file);
});

// 3. Predicción y Dibujo
async function predict() {
    if (!model) return;
    
    status.innerText = "Procesando...";
    
    // Ajustar resolución del dibujo al tamaño real mostrado
    canvas.width = imagePreview.width;
    canvas.height = imagePreview.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Preprocesamiento (Ajustado a 640x640 para YOLO)
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imagePreview)
                  .resizeNearestNeighbor([640, 640])
                  .cast('float32')
                  .expandDims(0)
                  .div(255.0);
    });

    try {
        // Ejecución en GPU (si está disponible)
        const result = await model.executeAsync(tensor);
        
        // Basado en tus logs: result[0] tiene forma [1, 300, 6]
        const boxesData = result[0].dataSync(); 
        const numDetections = result[0].shape[1];

        let found = false;
        for (let i = 0; i < numDetections; i++) {
            const index = i * 6;
            const x1 = boxesData[index];
            const y1 = boxesData[index + 1];
            const x2 = boxesData[index + 2];
            const y2 = boxesData[index + 3];
            const score = boxesData[index + 4];
            const classId = boxesData[index + 5];

            // Umbral de confianza: 0.4 (40%)
            if (score > 0.4) {
                found = true;
                drawBoundingBox(x1, y1, x2, y2, score, classId);
            }
        }

        status.innerText = found ? "Detección finalizada." : "No se detectaron objetos claros.";

        // Limpiar tensores de salida
        result.forEach(t => t.dispose());
    } catch (err) {
        console.error("Error en predicción:", err);
        status.innerText = "Error al analizar.";
    }

    tensor.dispose();
}

// 4. Función para pintar los cuadros
function drawBoundingBox(x1, y1, x2, y2, score, classId) {
    // Escalar de 640x640 al tamaño del preview
    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;

    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    // Dibujar rectángulo
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);

    // Dibujar etiqueta
    ctx.fillStyle = "#00FF00";
    ctx.font = "bold 16px Arial";
    const label = `Clase: ${classId} (${Math.round(score * 100)}%)`;
    ctx.fillText(label, x, y > 20 ? y - 5 : 20);
}

loadModel();