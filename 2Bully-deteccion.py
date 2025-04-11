import cv2 as cv
import numpy as np
import os
import time
from action_classifier import ActionClassifier

# Mostrar el directorio actual
print(f"Directorio actual: {os.getcwd()}")

# Verificar si el archivo del modelo existe
model_path = "/Users/sebastiandevillasante/Desktop/Open_pose_video/Copia de graph_opt.pb 2"
print(f"Buscando el archivo: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: No se encuentra el archivo {model_path}")
    exit(1)
else:
    print("¡Archivo encontrado!")

# Variables ajustables
width, height = 256, 256  # Reducido de 368x368 para mejor rendimiento
thr = 0.2  # Umbral más bajo para detectar más puntos

# Definir partes del cuerpo relevantes
BODY_PARTS = {
    "Neck": 1,  # Añadido el cuello
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7
}

# Cargar modelo
try:
    net = cv.dnn.readNetFromTensorflow(model_path)
    print("Modelo cargado exitosamente")
except cv.error as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Inicializar el clasificador de acciones
action_classifier = ActionClassifier()

# Verificar si el modelo está entrenado
if not action_classifier.is_trained:
    print("ADVERTENCIA: El modelo no está entrenado. Por favor, ejecuta train_classifier.py primero.")
    print("Presiona 'q' para salir o cualquier otra tecla para continuar...")
    if cv.waitKey(0) & 0xFF == ord('q'):
        exit(1)

# Buffer para almacenar secuencias de keypoints
keypoint_buffer = []
MAX_BUFFER_SIZE = 60  # Actualizado de 20 a 60 para coincidir con el entrenamiento

# Detector de poses
def poseDetector(frame):
    global keypoint_buffer
    
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    frame_small = cv.resize(frame, (width, height))
    
    net.setInput(cv.dnn.blobFromImage(frame_small, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    
    if out.shape[1] != 19:
        print(f"Error: El modelo no está devolviendo el número correcto de puntos. Esperado: 19, Recibido: {out.shape[1]}")
        return frame

    people_points = []
    
    # Procesar puntos
    for i in range(19):
        if i not in BODY_PARTS.values():
            people_points.append([None])
            continue
            
        heatMap = out[0, i, :, :]
        heatMap = cv.resize(heatMap, (frameWidth, frameHeight))
        
        peaks = []
        heatMap_flat = heatMap.flatten()
        indices = np.where(heatMap_flat > thr * 0.5)[0]
        
        for idx in indices:
            y = idx // frameWidth
            x = idx % frameWidth
            
            if x > 0 and x < frameWidth-1 and y > 0 and y < frameHeight-1:
                if heatMap[y, x] >= heatMap[y-1:y+2, x-1:x+2].max():
                    peaks.append((x, y))
        
        if not peaks:
            peaks.append(None)
        people_points.append(peaks)
    
    # Preparar keypoints para el clasificador
    current_keypoints = []
    for part in BODY_PARTS.values():
        if people_points[part] and people_points[part][0] is not None:
            current_keypoints.append(list(people_points[part][0]))
        else:
            current_keypoints.append([0, 0])
    
    # Verificar que tenemos todos los puntos necesarios
    if len(current_keypoints) != 7:
        print(f"Error: Número incorrecto de keypoints detectados: {len(current_keypoints)}")
        return frame
    
    # Actualizar buffer de keypoints
    keypoint_buffer.append(current_keypoints)
    if len(keypoint_buffer) > MAX_BUFFER_SIZE:
        keypoint_buffer.pop(0)
    
    # Si tenemos suficientes frames, usar el clasificador
    action = "neutra"
    confidence_action = 0.0
    confidence_other = 0.0
    
    if len(keypoint_buffer) == MAX_BUFFER_SIZE:
        try:
            # Convertir el buffer a numpy array con la forma correcta
            sequence = np.array(keypoint_buffer, dtype=np.float32)
            print(f"Forma del buffer antes de la predicción: {sequence.shape}")
            
            if sequence.shape == (60, 7, 2):
                action, confidence_action, confidence_other = action_classifier.predict(sequence)
                if action:
                    print(f"Predicción exitosa: {action}")
            else:
                print(f"Forma incorrecta del buffer: {sequence.shape}")
        except Exception as e:
            print(f"Error en la predicción: {e}")
            print(f"Tipo de error: {type(e)}")
            import traceback
            print(traceback.format_exc())
    
    # Dibujar los keypoints
    for i, part in enumerate(BODY_PARTS.values()):
        if people_points[part] and people_points[part][0] is not None:
            point = people_points[part][0]
            cv.circle(frame, point, 3, (0, 255, 0), -1)
    
    # Mostrar la predicción en pantalla
    action_text = "Neutra"
    if action == "violencia":
        action_text = "Violencia"
    
    # Dibujar fondo para el texto
    cv.rectangle(frame, (5, 5), (300, 85), (0, 0, 0), -1)
    
    # Mostrar texto con color según la acción
    color = (0, 255, 0)  # Verde para neutra
    if action == "violencia":
        color = (0, 0, 255)  # Rojo para violencia
    
    # Mostrar acción y confianza
    conf_text = f"Acción: {action_text}"
    cv.putText(frame, conf_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Mostrar porcentaje de confianza para la acción detectada
    conf_text = f"Confianza {action_text}: {confidence_action:.1f}%"
    cv.putText(frame, conf_text, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Mostrar porcentaje de confianza para la otra acción
    other_action = "Violencia" if action == "neutra" else "Neutra"
    other_color = (0, 0, 255) if other_action == "Violencia" else (0, 255, 0)
    conf_text = f"Confianza {other_action}: {confidence_other:.1f}%"
    cv.putText(frame, conf_text, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, other_color, 2)
    
    return frame

# Inicializar la cámara
print("\nInicializando cámara...")
print("Intentando abrir la cámara con diferentes configuraciones...")

# Intentar diferentes configuraciones de cámara
camera_configs = [
    (0, cv.CAP_AVFOUNDATION),  # macOS
    (0, None),                 # Configuración por defecto
    (1, None),                 # Cámara secundaria
    (2, None)                  # Tercera cámara
]

cap = None
for camera_id, backend in camera_configs:
    try:
        if backend:
            cap = cv.VideoCapture(camera_id, backend)
        else:
            cap = cv.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"✓ Cámara {camera_id} abierta exitosamente")
            
            # Configurar la resolución de la cámara
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv.CAP_PROP_FPS, 30)
            
            # Verificar si la configuración se aplicó
            actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            actual_fps = cap.get(cv.CAP_PROP_FPS)
            
            print(f"  - Resolución configurada: 640x480 (actual: {actual_width}x{actual_height})")
            print(f"  - FPS configurado: 30 (actual: {actual_fps})")
            
            # Intentar leer un frame para verificar
            ret, frame = cap.read()
            if ret:
                print("  - ✓ Lectura de frame exitosa")
                break
            else:
                print("  - ❌ No se pudo leer frame de la cámara")
                cap.release()
                cap = None
    except Exception as e:
        print(f"  - ❌ Error al abrir cámara {camera_id}: {e}")
        if cap:
            cap.release()
            cap = None

if not cap or not cap.isOpened():
    print("\n❌ No se pudo inicializar ninguna cámara")
    print("Por favor, verifica que:")
    print("1. La cámara está conectada")
    print("2. No hay otras aplicaciones usando la cámara")
    print("3. Tienes permisos para acceder a la cámara")
    exit(1)

print("\nIniciando detección de poses...")
print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: No se pudo obtener un frame de la cámara")
        break

    # Voltear horizontalmente la imagen (para corregir el efecto espejo)
    frame = cv.flip(frame, 1)

    output = poseDetector(frame)
    cv.imshow("Detección de Poses", output)

    # Salir del bucle si se presiona la tecla 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

def process_frame(frame, keypoints):
    """Procesa un frame con los keypoints detectados"""
    try:
        # Verificar que tenemos suficientes keypoints
        if len(keypoints) != 17:  # MediaPipe detecta 17 keypoints
            print(f"Error: Se esperaban 17 keypoints, pero se recibieron {len(keypoints)}")
            return frame
        
        # Convertir keypoints al formato esperado
        keypoints_array = np.array([[kp.x, kp.y] for kp in keypoints], dtype=np.float32)
        
        # Actualizar el buffer
        keypoints_buffer.append(keypoints_array)
        if len(keypoints_buffer) > MAX_BUFFER_SIZE:
            keypoints_buffer.pop(0)
        
        # Si tenemos suficientes frames, hacer la predicción
        if len(keypoints_buffer) == MAX_BUFFER_SIZE:
            # Convertir el buffer a la forma esperada por el clasificador
            sequence = np.array(keypoints_buffer)
            print(f"Forma de la secuencia antes de predecir: {sequence.shape}")
            
            # Hacer la predicción
            action = action_classifier.predict(sequence)
            print(f"Acción detectada: {action}")
            
            # Dibujar el resultado
            if action == "violencia":
                color = (0, 0, 255)  # Rojo
            elif action == "neutra":
                color = (0, 255, 0)  # Verde
            else:
                color = (255, 0, 0)  # Azul
            
            # Dibujar el texto
            cv.putText(frame, f"Accion: {action}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Dibujar los keypoints
        for kp in keypoints:
            cv.circle(frame, (int(kp.x * frame.shape[1]), int(kp.y * frame.shape[0])), 
                     5, (0, 255, 0), -1)
        
        return frame
    except Exception as e:
        print(f"Error en process_frame: {e}")
        import traceback
        print(traceback.format_exc())
        return frame
