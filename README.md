# Modelos de Riesgo de Crédito con Redes Neuronales Artificiales

Este proyecto tiene como objetivo crear un modelo para predecir la probabilidad de incumplimiento en el pago de un crédito. Además, se representa esta probabilidad mediante una scorecard, se analizan las variables que influyen en el riesgo de incumplimiento y se desarrolla una aplicación web interactiva para mostrar los resultados.

---

## 📌 **Reto del Proyecto**

1. **Crear y validar un modelo de probabilidad de incumplimiento** basado en redes neuronales artificiales, optimizando su arquitectura.  
2. **Representar el modelo mediante una scorecard** que convierta las probabilidades en puntajes de crédito.  
3. **Analizar las variables** que incrementan el riesgo de incumplimiento.  
4. **Desarrollar una aplicación web** que permita a las personas conocer su scorecard y compararse contra la población.  

---

## 📂 **Estructura del Proyecto**

### 📝 **Análisis Descriptivo y Exploratorio**
- **Ubicación:** [`notebooks/1_exploracion.ipynb`](notebooks/1_exploracion.ipynb)  
- **Descripción:** 
  - Análisis de variables numéricas y categóricas.
  - Identificación de correlaciones e importancia de las variables.
  - Formulación de hipótesis sobre los factores de riesgo.
  - Preprocesamiento de los datos.

### 🤖 **Modelado Analítico**
- **Ubicación:** [`notebooks/2_modelado.ipynb`](notebooks/2_modelado.ipynb)  
- **Descripción:** 
  - Entrenamiento de cinco modelos basados en redes neuronales artificiales.
  - Evaluación del desempeño de los modelos.
  - Desarrollo de una función para convertir probabilidades de incumplimiento en puntajes de crédito (scorecard).

### 🌐 **Aplicación Web**
- **Ubicación:** [`app/`](app/)  
- **Descripción:** 
  - Aplicación web desarrollada con Streamlit.
  - Permite a los usuarios ingresar sus características personales y obtener su scorecard.
  - Compara el puntaje del usuario con la población general.

---

## ⚙️ **Requisitos del Sistema**

- **Lenguaje:** Python 3.10+  
- **Librerías principales:**  
  - TensorFlow / PyTorch (para el modelo de RNA).  
  - Streamlit (para la aplicación web).  
  - Pandas y NumPy (para análisis de datos).  
  - Scikit-learn (para evaluación de modelos).  
- Ver el archivo [`requirements.txt`](requirements.txt) para más detalles.

---

## 🚀 **Cómo Ejecutar el Proyecto**

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/aleuse/rna_trabajo2_riesgo_crediticio.git
   cd rna_trabajo2_riesgo_crediticio
