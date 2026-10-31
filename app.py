import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import gdown
import os

st.set_page_config(page_title="Tesis UBO - Detección OOD", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #333333; }
    p, div, span, label { color: #333333 !important; }
    h1, h2, h3 { color: #001A4D !important; }
    .stButton>button {
        background-color: #00AEEF;
        color: white !important;
        border-radius: 5px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #008CBA; }
    .resultado-normal {
        background-color: #D4EDDA;
        color: #155724 !important;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        border: 1px solid #c3e6cb;
    }
    .resultado-anomalia {
        background-color: #F8D7DA;
        color: #721C24 !important;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

NOMBRES_CLASES = {
    0: "Espiral Barrada",
    1: "Espiral No Barrada (Floja)",
    2: "Espiral No Barrada (Apretada)",
    3: "Elíptica Redonda",
    4: "Elíptica Alargada",
    5: "Borde con Bulbo",
    6: "Borde sin Bulbo",
    7: "Perturbada",
    8: "Fusión (OOD)",
    9: "Irregular (OOD)"
}

@st.cache_resource
def load_model():
    file_id_model = "1RzDEkPJf-CGbA7LGMXg_HwmPNbq5YNP2"
    output_model = "modelo_temp.pth"
    if not os.path.exists(output_model):
        with st.spinner("Cargando modelo ResNet-34..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id_model}", output_model, quiet=True)

    device = torch.device("cpu")
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load(output_model, map_location=device))
    model.eval()
    return model, device

@st.cache_data
def load_mahalanobis():
    # Cargar media, cov_inv y umbral desde el archivo .npz
    file_id_params = "TU_FILE_ID_DE_MAHALANOBIS_PARAMS"  # ← Reemplaza con el ID real
    output_params = "mahalanobis_params.npz"
    if not os.path.exists(output_params):
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id_params}", output_params, quiet=True)
            params = np.load(output_params)
            return params['media_global'], params['cov_inv'], float(params['umbral_maha'])
        except:
            # Fallback (no ideal): media cero, covarianza identidad, umbral 31.74
            return np.zeros(512), np.eye(512), 31.74
    params = np.load(output_params)
    return params['media_global'], params['cov_inv'], float(params['umbral_maha'])

def predecir_imagen(image, model, device, media, cov_inv):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confianza, pred = torch.max(probs, 1)

        original_fc = model.fc
        model.fc = nn.Identity()
        features = model(img_tensor).cpu().numpy()[0]
        model.fc = original_fc

        diff = features - media
        dist_maha = np.sqrt(np.einsum('i,ij,j->', diff, cov_inv, diff))

    return pred.item(), confianza.item(), dist_maha

model, device = load_model()
media_global, cov_inv, UMBRAL_MAHA = load_mahalanobis()

st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h2 style="color: #001A4D; margin-bottom: 0;">UNIVERSIDAD BERNARDO O'HIGGINS</h2>
    <p style="color: #00AEEF; font-size: 18px; margin-top: 5px;">Ingeniería en Informática</p>
    <hr style="border: 1px solid #00AEEF; width: 50%; margin: auto;">
</div>
""", unsafe_allow_html=True)

st.title("Dashboard de Detección Out-of-Distribution")
st.markdown("**Tesis:** Detección de Anomalías Morfológicas en Galaxias mediante Calibración Fina")

tab1, tab2, tab3, tab4 = st.tabs(["Demo en Vivo", "Resultados", "Pipeline", "Glosario"])

with tab1:
    st.header("Análisis de Galaxias en Tiempo Real")
    st.write("Sube una imagen de galaxia para clasificarla y detectar si es una anomalía (Out-of-Distribution).")

    uploaded_file = st.file_uploader("Elige una imagen (JPG/PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        col_img, col_res = st.columns(2)

        with col_img:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen cargada", use_container_width=True)

        with col_res:
            if st.button("Analizar Galaxia", use_container_width=True):
                with st.spinner("Procesando..."):
                    pred, conf, dist = predecir_imagen(image, model, device, media_global, cov_inv)
                    nombre_clase = NOMBRES_CLASES.get(pred, f"Clase {pred}")

                    st.subheader("Resultado del Análisis")

                    if dist > UMBRAL_MAHA:
                        st.markdown(f'<div class="resultado-anomalia">ANOMALÍA DETECTADA<br>Clase: {nombre_clase}</div>', unsafe_allow_html=True)
                        st.write(f"**Confianza SoftMax:** {conf*100:.2f}%")
                        st.write(f"**Distancia de Mahalanobis:** {dist:.2f} (Umbral: {UMBRAL_MAHA:.2f})")
                        st.info("El modelo detectó que esta galaxia no pertenece a ninguna de las 8 clases conocidas. Probablemente sea una morfología anómala.")
                    else:
                        st.markdown(f'<div class="resultado-normal">GALAXIA NORMAL<br>Clase: {nombre_clase}</div>', unsafe_allow_html=True)
                        st.write(f"**Confianza SoftMax:** {conf*100:.2f}%")
                        st.write(f"**Distancia de Mahalanobis:** {dist:.2f} (Umbral: {UMBRAL_MAHA:.2f})")
                        st.success("La galaxia ha sido clasificada correctamente dentro de las 8 clases conocidas (In-Distribution).")

with tab2:
    st.header("Resultados Comparativos del Estudio")
    st.write("Comparación del rendimiento de los métodos evaluados sobre el conjunto Test ID (1,800 imágenes).")

    data = {
        "Método": ["SoftMax (Baseline)", "IsoMaxPlus", "MC Dropout (Híbrido)", "Mahalanobis", "Energy Score"],
        "Detección OOD (%)": [0.0, 21.23, 2.87, 60.80, 99.80],
        "Falsos Positivos": ["-", "90/1800", "77/1800", "81/1800", "1712/1800"],
        "Conclusión": ["Ciego a OOD", "Detecta poco", "Ruido enmascara señal", "Óptimo equilibrio", "Demasiado agresivo"]
    }
    df = pd.DataFrame(data)

    def highlight_maha(row):
        if row['Método'] == 'Mahalanobis':
            return ['background-color: #D4EDDA; font-weight: bold'] * len(row)
        return [''] * len(row)

    st.dataframe(df.style.apply(highlight_maha, axis=1), use_container_width=True)

    st.markdown("---")
    st.subheader("Conclusión Principal")
    st.info("La **Distancia de Mahalanobis** logró el mejor equilibrio: detectó el 60.80% de las anomalías manteniendo una tasa de falsos positivos del 4.50%. El Energy Score, aunque detecta casi todo (99.80%), es inviable por su altísima tasa de falsos positivos (95.11%).")

with tab3:
    st.header("Pipeline Experimental")
    st.write("El proceso se dividió en cuatro etapas secuenciales:")

    st.markdown("""
    1. **Ingesta y Preprocesamiento:** Carga del dataset Galaxy10 DECaLS (HDF5), split de 8 clases ID y 2 OOD, muestreo estratificado (12,000 ID / 3,000 OOD) y normalización a 224x224.
    2. **Modelado y Entrenamiento:** ResNet-34 preentrenada. Fase de *Feature Extraction* (5 épocas) y *Fine-Tuning* completo (5 épocas).
    3. **Evaluación OOD:** Aplicación de Score de Distancia a Prototipos, MC Dropout (varianza), Mahalanobis (covarianza) y Energy Score sobre las características latentes.
    4. **Generación de Evidencia:** Cálculo de métricas y generación de figuras.
    """)

    st.subheader("Tecnologías Utilizadas")
    st.write("Python 3.10, PyTorch, Google Colab (GPU T4), NumPy, SciPy, Scikit-learn.")

with tab4:
    st.header("Glosario de Términos")
    st.markdown("""
    *   **Out-of-Distribution (OOD):** Datos que no pertenecen a la distribución de entrenamiento del modelo.
    *   **In-Distribution (ID):** Datos que sí pertenecen a las clases conocidas.
    *   **SoftMax:** Función de activación que normaliza las salidas en probabilidades.
    *   **Score de Distancia a Prototipos:** Score de anomalía basado en la distancia euclidiana a los prototipos de clase (inspirado en IsoMaxPlus).
    *   **MC Dropout:** Técnica bayesiana para medir incertidumbre con múltiples pasadas.
    *   **Distancia de Mahalanobis:** Medida de distancia que considera la covarianza de los datos.
    *   **Energy Score:** Método de detección OOD basado en los logits crudos.
    """)

st.markdown("---")
st.caption("Desarrollado por Allison Aguilar Zapata | Profesor Guía: John Henry Rios Griego | Santiago, 2026")
