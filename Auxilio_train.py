import cv2 as cv
import numpy as np
import os
import time
from action_classifier import ActionClassifier

# Configuración
FRAMES_PER_SEQUENCE = 10
THRESHOLD = 0.05
REPETICIONES_VIDEO = 100  # Número de veces que se repetirá cada video

# Rutas
DESKTOP_PATH = "/Users/sebastiandevillasante/Desktop"
TRAINING_DATA_PATH = os.path.join(DESKTOP_PATH, "training_data")
MODEL_PATH = "/Users/sebastiandevillasante/Desktop/Open_pose_video/Copia de graph_opt.pb 2"

BODY_PARTS = {
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7
}

def detect_keypoints(frame, net, frameWidth, frameHeight):
    try:
        frame_small = cv.resize(frame, (256, 256))
        blob = cv.dnn.blobFromImage(frame_small, 1.0, (256, 256), 
                                   (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()
        
        points = []
        for part in BODY_PARTS.values():
            try:
                prob_map = out[0, part, :, :]
                prob_map = cv.resize(prob_map, (frameWidth, frameHeight))
                min_val, prob, min_loc, point = cv.minMaxLoc(prob_map)
                if prob < THRESHOLD:
                    point = (0, 0)
                points.append([point[0], point[1]])
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
    # Brazo derecho
    if points[0][0] != 0 and points[1][0] != 0:  # Hombro derecho a codo derecho
        cv.line(frame, (int(points[0][0]), int(points[0][1])), 
                (int(points[1][0]), int(points[1][1])), (0, 255, 0), 2)
    if points[1][0] != 0 and points[2][0] != 0:  # Codo derecho a muñeca derecha
        cv.line(frame, (int(points[1][0]), int(points[1][1])), 
                (int(points[2][0]), int(points[2][1])), (0, 255, 0), 2)
    
    # Brazo izquierdo
    if points[3][0] != 0 and points[4][0] != 0:  # Hombro izquierdo a codo izquierdo
        cv.line(frame, (int(points[3][0]), int(points[3][1])), 
                (int(points[4][0]), int(points[4][1])), (0, 255, 0), 2)
    if points[4][0] != 0 and points[5][0] != 0:  # Codo izquierdo a muñeca izquierda
        cv.line(frame, (int(points[4][0]), int(points[4][1])), 
                (int(points[5][0]), int(points[5][1])), (0, 255, 0), 2)
    
    return frame

def process_video(video_path, action_type, net, repeticion):
    print(f"\nProcesando: {video_path} (Repetición {repeticion}/{REPETICIONES_VIDEO})")
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video")
        return None
    
    # Obtener información del video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    duration = total_frames / fps
    
    print(f"✓ Video: {duration:.1f} segundos, {total_frames} frames, {fps} FPS")
    
    # Calcular índices de frames
    frame_indices = np.linspace(0, total_frames-1, FRAMES_PER_SEQUENCE, dtype=int)
    print(f"✓ Tomando frames en índices: {frame_indices}")
    
    keypoints_sequence = []
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    for frame_idx in frame_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Error al leer frame {frame_idx}")
            continue
        
        points = detect_keypoints(frame, net, frameWidth, frameHeight)
        keypoints_sequence.append(points)
        
        # Mostrar el frame con los keypoints
        frame_with_keypoints = draw_keypoints(frame.copy(), points)
        cv.putText(frame_with_keypoints, f"Frame {len(keypoints_sequence)}/{FRAMES_PER_SEQUENCE}", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Procesando Video", frame_with_keypoints)
        cv.waitKey(1)  # Mostrar el frame por 1ms
        
        print(f"✓ Frame {len(keypoints_sequence)}/{FRAMES_PER_SEQUENCE}")
    
    cap.release()
    cv.destroyAllWindows()
    
    # Preguntar si se acepta este video
    print("\n¿Los keypoints se detectaron correctamente?")
    print("Presiona 's' para aceptar este video o 'n' para rechazarlo")
    while True:
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            print("✓ Video aceptado")
            break
        elif key == ord('n'):
            print("❌ Video rechazado")
            return None
    
    # Si tenemos al menos 8 frames, completamos hasta 10
    if len(keypoints_sequence) >= 8:
        # Completar hasta 10 frames si es necesario
        while len(keypoints_sequence) < FRAMES_PER_SEQUENCE:
            keypoints_sequence.append(keypoints_sequence[-1])
            print(f"✓ Frame {len(keypoints_sequence)}/{FRAMES_PER_SEQUENCE} (repetido)")
        
        print(f"✓ Secuencia completada con {len(keypoints_sequence)} frames")
        return np.array(keypoints_sequence)
    else:
        print(f"❌ Error: Solo se obtuvieron {len(keypoints_sequence)} frames de {FRAMES_PER_SEQUENCE} requeridos")
        return None

def main():
    print("\n=== PROCESAMIENTO DE VIDEOS ===")
    print(f"Cada video se procesará {REPETICIONES_VIDEO} veces para mejorar el entrenamiento")
    
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
    
    # Procesar videos
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
                
                # Repetir el procesamiento del video REPETICIONES_VIDEO veces
                for rep in range(REPETICIONES_VIDEO):
                    sequence = process_video(video_path, action, net, rep + 1)
                    if sequence is not None:
                        sequences.append(sequence)
                        labels.append(action)
                        print(f"✓ Video procesado exitosamente: {video_file} (Repetición {rep + 1}/{REPETICIONES_VIDEO})")
    
    if not sequences:
        print("\n❌ No se pudieron procesar videos")
        return
    
    print(f"\n✓ Total de secuencias procesadas: {len(sequences)}")
    print(f"✓ Distribución de acciones:")
    for action in ["neutra", "violencia"]:
        count = labels.count(action)
        print(f"  - {action}: {count} secuencias")
    
    # Preguntar si se procede con el entrenamiento
    print("\n¿Deseas proceder con el entrenamiento del modelo?")
    print("Presiona 's' para entrenar o 'n' para cancelar")
    while True:
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            print("\nEntrenando modelo...")
            classifier = ActionClassifier()
            classifier.train(sequences, labels)
            print("✓ Modelo entrenado y guardado")
            break
        elif key == ord('n'):
            print("❌ Entrenamiento cancelado")
            return

if __name__ == "__main__":
    main() 