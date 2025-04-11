import numpy as np
import os
import pickle

class ActionClassifier:
    def __init__(self):
        self.actions = ['neutra', 'violencia']
        self.num_keypoints = 7
        self.training_data = []
        self.training_labels = []
        self.is_trained = False
        
        # Usar una ubicación fija para el modelo
        self.model_path = '/Users/sebastiandevillasante/Desktop/OP_Vivo_Multiple people/trained_classifier_model_v2.pkl'
        
        # Intentar cargar el modelo si existe
        if os.path.exists(self.model_path):
            print(f"\nEncontrado modelo en: {self.model_path}")
            try:
                self.load_model()
                if self.is_trained:
                    print(f"✓ Modelo cargado exitosamente con {len(self.training_data)} ejemplos")
                else:
                    print("❌ El modelo existe pero no está entrenado")
            except Exception as e:
                print(f"❌ Error al cargar el modelo: {e}")
                self.training_data = []
                self.training_labels = []
                self.is_trained = False
        else:
            print(f"\nNo se encontró el modelo en: {self.model_path}")
            print("Se creará un nuevo modelo al entrenar")
    
    def preprocess_sequence(self, sequence):
        """Preprocesa una secuencia de keypoints para la predicción"""
        try:
            # Convertir a numpy array
            sequence = np.array(sequence, dtype=np.float32)
            
            # Verificar la forma
            print(f"Forma de la secuencia recibida: {sequence.shape}")
            
            # Asegurar que la secuencia tenga la forma correcta (N, 7, 2)
            if len(sequence.shape) != 3 or sequence.shape[1] != self.num_keypoints or sequence.shape[2] != 2:
                print(f"Error: La secuencia tiene forma incorrecta: {sequence.shape}")
                print(f"Se esperaba: (N, {self.num_keypoints}, 2)")
                return None
            
            return sequence
            
        except Exception as e:
            print(f"❌ Error en preprocess_sequence: {e}")
            print(f"Tipo de error: {type(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def predict(self, sequence):
        """Predice la acción para una secuencia de keypoints"""
        try:
            # Preprocesar la secuencia
            sequence = self.preprocess_sequence(sequence)
            if sequence is None:
                return None, 0.0, 0.0
                
            print(f"Forma de la secuencia a predecir: {sequence.shape}")
            
            # Verificar que tenemos secuencias de entrenamiento
            if not self.training_data:
                print("❌ No hay secuencias de entrenamiento disponibles")
                return None, 0.0, 0.0
                
            print(f"Numero de secuencias de entrenamiento: {len(self.training_data)}")
            
            # Normalizar la secuencia de entrada
            sequence = sequence / np.max(np.abs(sequence))
            
            # Encontrar las secuencias más similares para cada acción
            best_match = None
            min_distance = float('inf')
            distances = {"neutra": float('inf'), "violencia": float('inf')}
            
            for i, train_seq in enumerate(self.training_data):
                # Convertir a numpy array y verificar forma
                train_seq = np.array(train_seq, dtype=np.float32)
                
                # Si la secuencia tiene forma (N, 14), convertirla a (N, 7, 2)
                if train_seq.shape[1] == self.num_keypoints * 2:
                    train_seq = train_seq.reshape(train_seq.shape[0], self.num_keypoints, 2)
                elif train_seq.shape[1] != self.num_keypoints or train_seq.shape[2] != 2:
                    print(f"❌ Secuencia de entrenamiento {i} tiene forma incorrecta: {train_seq.shape}")
                    continue
                    
                # Normalizar la secuencia de entrenamiento
                train_seq = train_seq / np.max(np.abs(train_seq))
                
                # Calcular distancia usando la longitud más corta
                min_len = min(len(sequence), len(train_seq))
                distance = np.linalg.norm(sequence[:min_len] - train_seq[:min_len])
                
                # Actualizar la distancia mínima para esta acción
                action = self.training_labels[i]
                if distance < distances[action]:
                    distances[action] = distance
                
                # Actualizar la mejor coincidencia general
                if distance < min_distance:
                    min_distance = distance
                    best_match = i
                    
            if best_match is None:
                print("❌ No se encontraron secuencias de entrenamiento válidas")
                return None, 0.0, 0.0
                
            print(f"Mejor coincidencia encontrada: {best_match}")
            print(f"Distancia: {min_distance}")
            
            # Definir umbral de distancia
            DISTANCE_THRESHOLD = 15.0  # Aumentado de 2.0 a 15.0 para ser menos estricto
            
            # Calcular confianza para cada acción
            confidences = {}
            for action, distance in distances.items():
                # Normalizar la distancia para que esté entre 0 y 1
                normalized_distance = distance / DISTANCE_THRESHOLD
                # Convertir a confianza (distancia más pequeña = mayor confianza)
                confidence = (1.0 - normalized_distance) * 100
                # Dar más peso a la confianza de neutra
                if action == "neutra":
                    confidence = confidence * 1.2  # Aumentar en un 20%
                # Asegurar que esté entre 0 y 100
                confidences[action] = max(0.0, min(100.0, confidence))
            
            print(f"Confianza de las predicciones:")
            print(f"  - Neutra: {confidences['neutra']:.2f}%")
            print(f"  - Violencia: {confidences['violencia']:.2f}%")
            
            # Determinar la acción final
            if confidences['violencia'] > confidences['neutra'] and confidences['violencia'] >= 30.0:
                return "violencia", confidences['violencia'], confidences['neutra']
            else:
                return "neutra", confidences['neutra'], confidences['violencia']
            
        except Exception as e:
            print(f"❌ Error en predict: {e}")
            print(f"Tipo de error: {type(e)}")
            import traceback
            print(traceback.format_exc())
            return None, 0.0, 0.0
    
    def train(self, sequences, labels):
        """Entrena el modelo con datos de ejemplo"""
        if len(sequences) != len(labels):
            raise ValueError("El número de secuencias y etiquetas debe ser igual")
        
        # Preprocesar todas las secuencias
        processed_sequences = []
        for seq in sequences:
            # Convertir a numpy array
            seq_array = np.array(seq, dtype=np.float32)
            
            # Verificar y corregir la forma
            if len(seq_array.shape) == 2 and seq_array.shape[1] == self.num_keypoints * 2:
                # Reshape a (N, 7, 2)
                seq_array = seq_array.reshape(len(seq_array), self.num_keypoints, 2)
            elif seq_array.shape[1] != self.num_keypoints or seq_array.shape[2] != 2:
                print(f"❌ Error: Forma incorrecta de secuencia: {seq_array.shape}")
                continue
                
            processed_sequences.append(seq_array)
        
        if not processed_sequences:
            raise ValueError("No se pudieron procesar secuencias válidas")
        
        # Combinar con datos de entrenamiento anteriores si existen
        if self.is_trained and self.training_data:
            print(f"\nCombinando con {len(self.training_data)} secuencias de entrenamiento anteriores")
            self.training_data.extend(processed_sequences)
            self.training_labels.extend(labels)
        else:
            # Si no hay datos anteriores, usar solo los nuevos
            self.training_data = processed_sequences
            self.training_labels = labels
        
        self.is_trained = True
        
        # Guardar el modelo
        self.save_model(self.model_path)
        
        print(f"✓ Modelo entrenado exitosamente con {len(self.training_data)} ejemplos en total")
        print(f"  - Secuencias anteriores: {len(self.training_data) - len(processed_sequences)}")
        print(f"  - Nuevas secuencias: {len(processed_sequences)}")
    
    def save_model(self, model_path):
        """Guarda el modelo en un archivo"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"✓ Modelo guardado en: {model_path}")
        except Exception as e:
            print(f"❌ Error al guardar el modelo: {e}")
    
    def load_model(self):
        """Carga el modelo desde un archivo"""
        try:
            with open(self.model_path, 'rb') as f:
                loaded_model = pickle.load(f)
                self.training_data = loaded_model.training_data
                self.training_labels = loaded_model.training_labels
                self.is_trained = loaded_model.is_trained
            print(f"Modelo cargado exitosamente con {len(self.training_data)} ejemplos")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            raise e 