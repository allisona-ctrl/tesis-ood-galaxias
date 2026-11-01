# Detección de Out-of-Distribution en Clasificación Morfológica de Galaxias

Este repositorio contiene el código fuente, los datos procesados y el framework experimental desarrollado para la tesis **"Detección de Out-of-Distribution en clasificación morfológica de galaxias mediante técnicas de calibración fina en el dataset Galaxy10 DECaLS"**, presentada en la Universidad Bernardo O'Higgins.

**Autora:** Allison Aguilar   
**Profesor Guía:** John Henry Rios Griego  

## Descripción del Proyecto

El proyecto evalúa la capacidad de un clasificador ResNet-34 para detectar morfologías galácticas desconocidas (Out-of-Distribution, OOD). Se comparan cuatro métodos de detección:
* SoftMax (Baseline)
* Score de Distancia a Prototipos (inspirado en IsoMaxPlus)
* MC Dropout
* Distancia de Mahalanobis (Calibración Fina)
* *Energy Score* (como comparación adicional)

## Demo en Vivo (Dashboard)
El proyecto incluye un dashboard interactivo desarrollado en Streamlit que permite subir una imagen de galaxia y obtener su clasificación junto con el score de anomalía de Mahalanobis.
* **Enlace al Dashboard:** [https://tesis-ood-galaxias.streamlit.app/](https://tesis-ood-galaxias.streamlit.app/)

## Estructura del Repositorio

* `pipeline_ood.ipynb`: Cuaderno de Google Colab con el código completo (entrenamiento, evaluación y generación de figuras).
* `app.py`: Código fuente del dashboard interactivo en Streamlit.
* `requirements.txt`: Lista de librerías y dependencias necesarias.
* `modelos/`: Carpeta con los pesos de los modelos entrenados (`.pth`).
* `figuras/`: Gráficos generados (matrices de confusión, histogramas, etc.).

## Requisitos e Instalación

Para replicar los resultados o ejecutar el código localmente, necesitas Python 3.10 y las siguientes librerías (puedes instalarlas con `pip install -r requirements.txt`):

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
scipy>=1.10.0
streamlit>=1.25.0
```
Reproducibilidad y Semillas

Para garantizar que los resultados experimentales sean idénticos en cada ejecución, se utilizaron las siguientes semillas (seeds):
* **Semilla Global:** `42` (utilizada en `np.random.seed(42)` para el muestreo del dataset y en `random_state=42` para la división de datos).
* **Semilla Bootstrap:** `42` (utilizada con `np.random.default_rng(42)` para el cálculo de los intervalos de confianza al 95%).

## Dataset

El modelo fue entrenado utilizando el dataset público **Galaxy10 DECaLS**. El archivo original en formato HDF5 (`Galaxy10_DECals_NoDuplicated.h5`) no se incluye en este repositorio por su tamaño, pero puede descargarse desde [Zenodo](https://doi.org/10.5281/zenodo.10845026). 

Para ejecutar el cuaderno, se espera que el archivo HDF5 se encuentre en la ruta:
`/content/drive/MyDrive/Tesis/Galaxy10_DECals_NoDuplicated.h5` (si se usa Google Colab).

Resultados Principales

* **Baseline (SoftMax):** Clasificó erróneamente el 100% de las anomalías con una confianza promedio del 61.04%.
* **Distancia de Mahalanobis:** Logró la mejor tasa de detección OOD (**62.17%**, IC 95% [60.40–63.83]) manteniendo solo un **4.78%** de falsos positivos.
* **MC Dropout (Híbrido):** Mostró un rendimiento limitado (0.43% de detección) debido al ruido introducido por el Dropout agresivo.
* **Energy Score:** Aunque detectó el 99.87% de las anomalías, su tasa de falsos positivos fue del 93.89%, lo que lo hace inviable en la práctica.
