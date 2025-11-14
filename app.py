import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


# 0. Definición del transformador personalizado
#    (tiene que ser idéntico al del notebook)
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para reproducir exactamente el preprocesamiento
    del notebook:
      - selección de columnas (cols_keep)
      - creación de variables derivadas (flags + log)
    """

    def __init__(self, cols_keep):
        self.cols_keep = cols_keep

    def fit(self, X, y=None):
        # No aprende nada, solo por compatibilidad con sklearn
        return self

    def transform(self, X):
        X = X.copy()

        # 1) Selección de columnas originales usadas en el modelo
        X = X[self.cols_keep]

        # 2) Creación de flags exactamente igual que en el notebook
        #    (ajusta los umbrales si en el notebook son otros)
        X["flag_aq_gt5"] = (X["AQ"] > 5).astype(int)
        X["flag_uss_le2"] = (X["USS"] <= 2).astype(int)
        X["flag_voc_ge6"] = (X["VOC"] >= 6).astype(int)
        X["flag_foot_lt40"] = (X["footfall"] < 40).astype(int)

        # 3) Variable logarítmica de footfall
        X["footfall_log"] = np.log1p(X["footfall"])

        return X

# 1. Cargar preprocesador y modelo
# Cargamos el bloque de preprocesado (selección columnas + flags + log + escalado)
preprocessor = joblib.load("models/preprocessor_sensorfail.pkl")

# Cargamos el modelo final (SVM optimizado)
model = joblib.load("models/model_sensorfail_svm.pkl")

# 2. Definir las columnas que se piden al usuario
# Son las variables originales que entran al preprocesador
FEATURES = [
    "footfall",
    "AQ",
    "USS",
    "CS",
    "VOC",
    "RP",
    "IP",
    "Temperature",
]

st.title("Predicción de fallo de sensores")

st.write(
    """
    Introduce los valores actuales de los sensores y el modelo
    predecirá si se espera **fallo** o **no fallo**, junto con
    la probabilidad estimada.
    """
)

# 3. Función para pedir los datos al usuario
def get_user_input():
    """
    Construyo el formulario de entrada y devuelvo los datos en un DataFrame
    de una sola fila, con las columnas que el preprocesador espera.
    """
    data = {}

    for feat in FEATURES:
        valor = st.number_input(f"Valor de {feat}", value=0.0)
        data[feat] = valor

    df = pd.DataFrame([data])
    return df


# Genero los inputs y obtengo los datos del usuario
input_df = get_user_input()

st.write("### Datos introducidos")
st.dataframe(input_df)

# 4. Inferencia (preprocesar + predecir)

if st.button("Predecir fallo"):
    # 4.1 Aplicar el preprocesador (ingeniería + escalado, etc.)
    X_proc = preprocessor.transform(input_df)

    # 4.2 Usar el modelo sobre los datos ya preprocesados
    pred = model.predict(X_proc)[0]

    # Probabilidades por clase: [P(no fallo), P(fallo)]
    proba = model.predict_proba(X_proc)[0]
    prob_no_fallo = proba[0]
    prob_fallo = proba[1]

    st.write("### Resultado")

    if pred == 1:
        st.error("⚠️ Predicción: FALLO de sensor")
    else:
        st.success("✅ Predicción: NO se espera fallo de sensor")

    st.write(f"Probabilidad de NO fallo: {prob_no_fallo:.2%}")
    st.write(f"Probabilidad de fallo: {prob_fallo:.2%}")

    st.caption(
        "El modelo etiqueta como `1` (fallo) cuando la probabilidad de fallo "
        "es mayor o igual al 50%."
    )
