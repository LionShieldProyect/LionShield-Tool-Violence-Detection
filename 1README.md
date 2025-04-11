# Sistema de Detección de Violencia en Tiempo Real

Este proyecto implementa un sistema de detección de violencia en tiempo real utilizando visión por computadora y aprendizaje automático. El sistema analiza secuencias de poses humanas para identificar comportamientos violentos.

## Características

- Detección de poses en tiempo real usando OpenCV y OpenPose
- Clasificación de acciones (violencia/neutra) con confianza
- Interfaz visual en tiempo real
- Sistema de entrenamiento personalizable

## Requisitos

- Python 3.8+
- OpenCV
- NumPy
- OpenPose (modelo pre-entrenado)

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Descargar el modelo de OpenPose y colocarlo en la carpeta correcta

## Uso

### Entrenamiento
Para entrenar el modelo con nuevos videos:
```bash
python train_classifier.py
```

### Detección
Para iniciar la detección en tiempo real:
```bash
python Bully-deteccion
```

## Estructura del Proyecto

- `train_classifier.py`: Script para entrenar el modelo
- `action_classifier.py`: Clase principal para la clasificación de acciones
- `Bully-deteccion`: Script principal de detección
- `training_data/`: Directorio para videos de entrenamiento
  - `neutra/`: Videos de acciones neutras
  - `violencia/`: Videos de acciones violentas

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos.

## Licencia

Este proyecto está bajo la Licencia MIT. 
