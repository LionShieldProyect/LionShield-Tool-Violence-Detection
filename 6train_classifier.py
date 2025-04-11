import cv2 as cv
import numpy as np
import os
import time
from action_classifier import ActionClassifier

# Configuración
FRAMES_PER_SEQUENCE = 60  # Aumentado para capturar más frames de la acción
MIN_FRAMES_REQUIRED = 45  # Aumentado proporcionalmente
THRESHOLD = 0.2  # Mismo umbral que en bully-detection.py

# Rutas
DESKTOP_PATH = "/Users/sebastiandevillasante/Desktop"
TRAINING_DATA_PATH = os.path.join(DESKTOP_PATH, "training_data")
MODEL_PATH = "/Users/sebastiandevillasante/Desktop/Open_pose_video/Copia de graph_opt.pb 2"
CLASSIFIER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_classifier_model_v2.pkl")

BODY_PARTS = {
    "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7
}

def detect_keypoints(frame, net, frameWidth, frameHeight, action_type="neutra"):
    try:
        # Redimensionar el frame como en bully-detection.py
        frame_small = cv.resize(frame, (256, 256))
        
        # Crear el blob como en bully-detection.py
        net.setInput(cv.dnn.blobFromImage(frame_small, 1.0, (256, 256), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # Asegurarnos de tener 19 puntos como en bully-detection.py
        
        if out.shape[1] != 19:
            print(f"Error: El modelo no está devolviendo el número correcto de puntos. Esperado: 19, Recibido: {out.shape[1]}")
            return [[0, 0] for _ in range(len(BODY_PARTS))]
        
        # Ajustar el umbral según el tipo de acción
        current_threshold = THRESHOLD
        if action_type == "violencia":
            # Usar un umbral más bajo para videos de violencia
            current_threshold = THRESHOLD * 0.7
        
        points = []
        for part in BODY_PARTS.values():
            try:
                # Obtener el mapa de calor como en bully-detection.py
                heatMap = out[0, part, :, :]
                heatMap = cv.resize(heatMap, (frameWidth, frameHeight))
                
                # Encontrar picos como en bully-detection.py
                heatMap_flat = heatMap.flatten()
                indices = np.where(heatMap_flat > current_threshold * 0.5)[0]
                
                peaks = []
                for idx in indices:
                    y = idx // frameWidth
                    x = idx % frameWidth
                    
                    if x > 0 and x < frameWidth-1 and y > 0 and y < frameHeight-1:
                        if heatMap[y, x] >= heatMap[y-1:y+2, x-1:x+2].max():
                            peaks.append((x, y))
                
                if peaks:
                    # Usar el pico con mayor probabilidad
                    max_peak = max(peaks, key=lambda p: heatMap[p[1], p[0]])
                    points.append([max_peak[0], max_peak[1]])
                else:
                    points.append([0, 0])
                    
            except Exception as e:
                print(f"Error al procesar keypoint {part}: {e}")
                points.append([0, 0])
        
        return points
    except Exception as e:
        print(f"Error en detect_keypoints: {e}")
        return [[0, 0] for _ in range(len(BODY_PARTS))]

def draw_keypoints(frame, points):
    # Dibujar los puntos
    for i, point in enumerate(points):
        if point[0] != 0 or point[1] != 0:  # Si el punto no es (0,0)
            cv.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
    
    # Dibujar las líneas entre puntos
    # Torso (solo cuello)
    if points[0][0] != 0 and points[1][0] != 0:  # Cuello a hombro derecho
        cv.line(frame, (int(points[0][0]), int(points[0][1])), 
                (int(points[1][0]), int(points[1][1])), (0, 255, 0), 2)
    if points[0][0] != 0 and points[4][0] != 0:  # Cuello a hombro izquierdo
        cv.line(frame, (int(points[0][0]), int(points[0][1])), 
                (int(points[4][0]), int(points[4][1])), (0, 255, 0), 2)
    
    # Brazo derecho
    if points[1][0] != 0 and points[2][0] != 0:  # Hombro derecho a codo derecho
        cv.line(frame, (int(points[1][0]), int(points[1][1])), 
                (int(points[2][0]), int(points[2][1])), (0, 255, 0), 2)
    if points[2][0] != 0 and points[3][0] != 0:  # Codo derecho a muñeca derecha
        cv.line(frame, (int(points[2][0]), int(points[2][1])), 
                (int(points[3][0]), int(points[3][1])), (0, 255, 0), 2)
    
    # Brazo izquierdo
    if points[4][0] != 0 and points[5][0] != 0:  # Hombro izquierdo a codo izquierdo
        cv.line(frame, (int(points[4][0]), int(points[4][1])), 
                (int(points[5][0]), int(points[5][1])), (0, 255, 0), 2)
    if points[5][0] != 0 and points[6][0] != 0:  # Codo izquierdo a muñeca izquierda
        cv.line(frame, (int(points[5][0]), int(points[5][1])), 
                (int(points[6][0]), int(points[6][1])), (0, 255, 0), 2)
    
    return frame

def process_video(video_path, action_type, net):
    print(f"\nProcesando: {video_path}")
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video")
        return None
    
    # Obtener información del video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    duration = total_frames / fps
    
    print(f"✓ Video: {duration:.1f} segundos, {total_frames} frames, {fps} FPS")
    
    keypoints_sequence = []
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    # Procesar todos los frames del video
    frame_count = 0
    valid_frames = 0
    
    while True:  # Cambiado para procesar hasta el final del video
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        points = detect_keypoints(frame, net, frameHeight, frameWidth, action_type)
        valid_points = sum(1 for point in points if point[0] != 0 or point[1] != 0)
        
        if valid_points >= 3:  # Si tenemos al menos 3 puntos válidos
            keypoints_sequence.append(points)
            valid_frames += 1
            if frame_count % 30 == 0:  # Mostrar progreso cada 30 frames
                print(f"✓ Frame {frame_count}/{total_frames} procesado con {valid_points} keypoints")
        else:
            if frame_count % 30 == 0:  # Mostrar progreso cada 30 frames
                print(f"⚠️ Frame {frame_count} descartado por falta de keypoints válidos")
            
        frame_count += 1
    
    cap.release()
    
    # Verificar que tenemos suficientes frames válidos
    if len(keypoints_sequence) >= 10:  # Mínimo 10 frames válidos
        # Convertir a numpy array y asegurar la forma correcta
        sequence_array = np.array(keypoints_sequence, dtype=np.float32)
        
        # Verificar y corregir la forma si es necesario
        if len(sequence_array.shape) == 2 and sequence_array.shape[1] == len(BODY_PARTS) * 2:
            # Reshape de (N, 14) a (N, 7, 2)
            sequence_array = sequence_array.reshape(len(sequence_array), len(BODY_PARTS), 2)
        
        if sequence_array.shape[1] == len(BODY_PARTS) and sequence_array.shape[2] == 2:
            print(f"✓ Secuencia completa obtenida ({len(keypoints_sequence)} frames)")
            print(f"✓ Forma final de la secuencia: {sequence_array.shape}")
            return sequence_array
        else:
            print(f"❌ Error: Forma incorrecta de la secuencia: {sequence_array.shape}")
            return None
    else:
        print(f"❌ Error: Solo se obtuvieron {len(keypoints_sequence)} frames de 10 requeridos")
        return None

def main():
    print("\n=== PROCESAMIENTO DE VIDEOS ===")
    
    # Verificar que el modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: No se encuentra el modelo en {MODEL_PATH}")
        return
    
    # Cargar modelo de pose
    try:
        net = cv.dnn.readNetFromTensorflow(MODEL_PATH)
        print("✓ Modelo de pose cargado")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Preparar datos de entrenamiento
    sequences = []
    labels = []
    
    # Procesar videos de cada acción
    for action in ["neutra", "violencia"]:
        action_dir = os.path.join(TRAINING_DATA_PATH, action)
        if not os.path.exists(action_dir):
            print(f"\n⚠️ No se encuentra el directorio {action_dir}")
            continue
        
        print(f"\nProcesando videos de {action}...")
        for video_file in os.listdir(action_dir):
            if video_file.lower().endswith('.mov'):
                video_path = os.path.join(action_dir, video_file)
                print(f"\nProcesando video: {video_file}")
                
                sequence = process_video(video_path, action, net)
                if sequence is not None:
                    sequences.append(sequence)
                    labels.append(action)
                    print(f"✓ Video '{video_file}' procesado correctamente")
                    print(f"  - Frames procesados: {len(sequence)}")
                    print(f"  - Forma de la secuencia: {sequence.shape}")
    
    if not sequences:
        print("\n❌ No se pudieron procesar secuencias para el entrenamiento")
        return
    
    print(f"\n✓ Total de secuencias para entrenamiento: {len(sequences)}")
    print(f"✓ Distribución de acciones:")
    for action in ["neutra", "violencia"]:
        count = sum(1 for l in labels if l == action)
        print(f"  - {action}: {count} videos")
    
    # Entrenar el modelo
    print("\nEntrenando modelo...")
    classifier = ActionClassifier()
    classifier.train(sequences, labels)
    
    # Guardar el modelo entrenado
    try:
        classifier.save_model(CLASSIFIER_MODEL_PATH)
        print(f"✓ Modelo guardado exitosamente en: {CLASSIFIER_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")
        return

if __name__ == "__main__":
    main() 
