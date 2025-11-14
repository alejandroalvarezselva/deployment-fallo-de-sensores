# 🔧 Predicción de Fallo de Sensores — Demo en Streamlit

Esta es la **aplicación web de demostración** del proyecto de predicción de fallos de sensores.  
La app permite introducir datos de sensores y obtener una predicción automática sobre si existe riesgo de fallo, usando un modelo **SVM** entrenado previamente.

📌 **Este repositorio contiene únicamente el código del despliegue (Streamlit).**  
📌 El código completo del proyecto (notebooks, análisis, entrenamiento, etc.) está en otro repositorio.


## 🚀 Probar la aplicación

Puedes probar la app directamente aquí:

👉 **https://alejandroalvarezselva-fallodesensores.streamlit.app/**  

No necesitas instalar nada: simplemente entra y pruébala.


## 🧠 ¿Qué hace esta aplicación?

- Recibe como entrada valores de sensores industriales.  
- Aplica el mismo preprocesamiento que se usó durante el entrenamiento:
  - Selección de columnas
  - Ingeniería de variables
  - Escalado
- Utiliza un modelo **SVM** para predecir si el sensor puede fallar.  
- Muestra el resultado de forma simple e inmediata.

Esta app está pensada como una **demo rápida** del modelo final.


## 📂 Contenido de este repositorio

app.py # Código de la aplicación Streamlit
models/ # Pipeline y modelo SVM entrenado (joblib)
requirements.txt # Dependencias necesarias para Streamlit
train_model.py # Script usado para preparar el modelo del deployment
README.md # Documentación del proyecto


## 📘 Proyecto completo

El desarrollo completo del proyecto (EDA, notebooks, análisis, entrenamiento, comparaciones de modelos…) se encuentra en el siguiente repositorio:

https://github.com/alejandroalvarezselva/sensor-failure-ml-project


## 👤 Autor

**Alejandro Álvarez Selva**
