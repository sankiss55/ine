let model;
const status = document.getElementById('status');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 1. Cargar el modelo al iniciar
async function loadModel() {
    try {
        // Forzamos CPU si WebGL da problemas (puedes quitarlo si arreglas WebGL)
        await tf.setBackend('cpu'); 
        
        const modelUrl = "https://sankiss55.github.io/ine/best_web_model/model.json";
        model = await tf.loadGraphModel(modelUrl);
        
        status.innerText = "Modelo listo. Sube una imagen.";
        fileInput.disabled = false;
    } catch (err) {
        status.innerText = "Error al cargar modelo. Revisa la consola.";
        console.error(err);
    }
}

// 2. Procesar la imagen cuando el usuario la selecciona
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (f) => {
        imagePreview.src = f.target.result;
        imagePreview.onload = () => predict(); // Ejecutar detección al cargar imagen
    };
    reader.readAsDataURL(file);
});
// 3. Ejecutar la predicción
async function predict() {
    status.innerText = "Analizando...";
    
    // Ajustar canvas al tamaño de la imagen mostrada
    canvas.width = imagePreview.width;
    canvas.height = imagePreview.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Preprocesamiento de la imagen CORREGIDO
    const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(imagePreview)
                  .resizeNearestNeighbor([640, 640]) 
                  .cast('float32') // <--- CAMBIO AQUÍ
                  .expandDims(0)
                  .div(255.0);
    });

    try {
        const predictions = await model.executeAsync(tensor);
        console.log("Resultado del modelo:", predictions);
        status.innerText = "Detección completada. Revisa los datos en la consola.";
        
        // Limpieza de memoria
        if (Array.isArray(predictions)) {
            predictions.forEach(p => p.dispose());
        } else {
            predictions.dispose();
        }
    } catch (err) {
        console.error("Error en la ejecución del modelo:", err);
        status.innerText = "Error al procesar la imagen.";
    }

    tensor.dispose();
}
loadModel();