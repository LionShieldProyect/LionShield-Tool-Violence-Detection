import cv2 as cv
import numpy as np
import os
import time

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
width, height = 368, 368
thr = 0.2  # Umbral más bajo para detectar más puntos

# Parámetros para detección de violencia
VELOCITY_THRESHOLD = 100  # Umbral de velocidad para considerar movimiento violento (píxeles por segundo)
CONTACT_DISTANCE = 50     # Distancia en píxeles para considerar contacto entre personas
VIOLENCE_COOLDOWN = 2.0   # Tiempo en segundos entre alertas de violencia
last_violence_alert = 0   # Tiempo de la última alerta de violencia

# Cargar modelo
try:
    net = cv.dnn.readNetFromTensorflow(model_path)  # Cambiar a readNetFromTensorflow
    print("Modelo cargado exitosamente")
except cv.error as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Definir partes del cuerpo
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Clase para rastrear personas
class PersonTracker:
    def __init__(self):
        self.people = []  # Lista de personas detectadas
        self.next_id = 0  # ID para la siguiente persona
        self.prev_positions = {}  # Posiciones anteriores para cálculo de velocidad
        self.last_update_time = time.time()
    
    def update(self, people_points):
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Si no hay personas detectadas, reiniciar
        if not any(people_points):
            self.people = []
            self.prev_positions = {}
            return []
        
        # Agrupar puntos por persona
        new_people = self._group_points_by_person(people_points)
        
        # Asignar IDs a nuevas personas
        for person in new_people:
            if not person.get('id'):
                person['id'] = self.next_id
                self.next_id += 1
        
        # Calcular velocidades
        self._calculate_velocities(new_people, dt)
        
        # Actualizar personas
        self.people = new_people
        
        return new_people
    
    def _group_points_by_person(self, people_points):
        # Implementación simplificada: cada conjunto de puntos forma una persona
        people = []
        
        # Obtener puntos no nulos para cada parte del cuerpo
        valid_points = {}
        for part, points in enumerate(people_points):
            if points and points[0] is not None:
                valid_points[part] = points
        
        # Agrupar puntos cercanos
        used_points = set()
        for part, points in valid_points.items():
            for point in points:
                if point in used_points:
                    continue
                
                # Crear nueva persona con este punto
                person = {'id': None, 'points': {}, 'velocities': {}}
                person['points'][part] = point
                used_points.add(point)
                
                # Buscar otros puntos cercanos para la misma persona
                for other_part, other_points in valid_points.items():
                    if other_part == part:
                        continue
                    
                    for other_point in other_points:
                        if other_point in used_points:
                            continue
                        
                        # Calcular distancia
                        dist = np.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                        if dist < CONTACT_DISTANCE * 2:  # Usar un umbral mayor para agrupar
                            person['points'][other_part] = other_point
                            used_points.add(other_point)
                
                people.append(person)
        
        return people
    
    def _calculate_velocities(self, people, dt):
        if dt <= 0:
            return
        
        for person in people:
            person_id = person['id']
            
            # Inicializar velocidades si no existen
            if person_id not in self.prev_positions:
                self.prev_positions[person_id] = {}
                for part, point in person['points'].items():
                    self.prev_positions[person_id][part] = point
                continue
            
            # Calcular velocidades para cada parte del cuerpo
            for part, point in person['points'].items():
                if part in self.prev_positions[person_id]:
                    prev_point = self.prev_positions[person_id][part]
                    dx = point[0] - prev_point[0]
                    dy = point[1] - prev_point[1]
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    person['velocities'][part] = velocity
                
                # Actualizar posición anterior
                self.prev_positions[person_id][part] = point

# Detector de poses
def poseDetector(frame):
    global last_violence_alert
    
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(BODY_PARTS) == out.shape[1])

    # Lista para almacenar los puntos de cada persona detectada
    people_points = []
    
    # Procesar cada mapa de calor
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        heatMap = cv.resize(heatMap, (frameWidth, frameHeight))
        
        # Encontrar múltiples puntos en el mapa de calor
        peaks = []
        heatMap_flat = heatMap.flatten()
        indices = np.where(heatMap_flat > thr)[0]
        
        for idx in indices:
            y = idx // frameWidth
            x = idx % frameWidth
            
            # Verificar si es un máximo local
            if x > 0 and x < frameWidth-1 and y > 0 and y < frameHeight-1:
                if heatMap[y, x] >= heatMap[y-1:y+2, x-1:x+2].max():
                    peaks.append((x, y))
        
        if not peaks:  # Si no se encuentran picos, agregar None
            peaks.append(None)
        people_points.append(peaks)

    # Actualizar el rastreador de personas
    people = person_tracker.update(people_points)
    
    # Detectar violencia
    violence_detected = False
    current_time = time.time()
    
    if len(people) >= 2:  # Necesitamos al menos 2 personas para detectar violencia
        for i, person1 in enumerate(people):
            for j, person2 in enumerate(people):
                if i >= j:  # Evitar comparar la misma persona o comparaciones duplicadas
                    continue
                
                # Verificar contacto y velocidad
                for part1, point1 in person1['points'].items():
                    # Solo verificar manos para violencia
                    if part1 not in [BODY_PARTS["RWrist"], BODY_PARTS["LWrist"]]:
                        continue
                    
                    # Verificar velocidad de la mano
                    velocity1 = person1['velocities'].get(part1, 0)
                    if velocity1 < VELOCITY_THRESHOLD:
                        continue
                    
                    # Verificar contacto con otra persona
                    for part2, point2 in person2['points'].items():
                        dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                        if dist < CONTACT_DISTANCE:
                            # Verificar si ha pasado suficiente tiempo desde la última alerta
                            if current_time - last_violence_alert > VIOLENCE_COOLDOWN:
                                violence_detected = True
                                last_violence_alert = current_time
                                break
                    
                    if violence_detected:
                        break
            
            if violence_detected:
                break

    # Dibujar las conexiones para cada conjunto de puntos detectados
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    # Dibujar las conexiones
    for pair in POSE_PAIRS:
        partFrom, partTo = pair[0], pair[1]
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        
        if people_points[idFrom] and people_points[idTo]:
            for pointFrom in people_points[idFrom]:
                if pointFrom is None:
                    continue
                for pointTo in people_points[idTo]:
                    if pointTo is None:
                        continue
                    
                    # Solo conectar puntos que estén relativamente cerca
                    dist = np.sqrt((pointFrom[0] - pointTo[0])**2 + (pointFrom[1] - pointTo[1])**2)
                    if dist < frameWidth * 0.2:  # Ajusta este valor según sea necesario
                        color = colors[int(dist / (frameWidth * 0.2) * len(colors))]
                        cv.line(frame, pointFrom, pointTo, color, 2)
                        cv.circle(frame, pointFrom, 3, color, thickness=-1, lineType=cv.FILLED)
                        cv.circle(frame, pointTo, 3, color, thickness=-1, lineType=cv.FILLED)
    
    # Mostrar alerta de violencia
    if violence_detected:
        cv.putText(frame, "¡VIOLENCIA DETECTADA!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.rectangle(frame, (30, 30), (frameWidth - 30, 80), (0, 0, 255), 2)
    
    return frame

# Inicializar el rastreador de personas
person_tracker = PersonTracker()

# Inicializar la cámara
cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)  # Cambia el índice si es necesario

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
else:
    print("Cámara abierta exitosamente.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo obtener un frame de la cámara.")
            break

        # Voltear horizontalmente la imagen (para corregir el efecto espejo)
        frame = cv.flip(frame, 1)

        output = poseDetector(frame)
        cv.imshow("Detección de Poses y Violencia", output)

        # Salir del bucle si se presiona la tecla 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
