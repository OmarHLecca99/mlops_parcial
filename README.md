# 🧠 Proyecto MLOps — Pipeline Automatizado con DVC + MLflow

Este proyecto implementa un flujo completo de **Machine Learning reproducible** utilizando:
- **DVC (Data Version Control)** para versionar datos y gestionar pipeline.
- **MLflow** para registrar experimentos, métricas y gestionar modelos.
- **Scikit-learn** para la implementación del modelo de clasificación.
- **Pipeline modular** dividido en etapas (`preprocess`, `train`, `monitor_data_drift`).

---

## ⚙️ Requisitos previos

Antes de ejecutar el proyecto asegúrate de tener instalado:

- **Python 3.11**
- **Git**
- **DVC**
- **MLflow**

---

## 🚀 Instrucciones de ejecución

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/OmarHLecca99/mlops_project.git
cd mlops_project
```

### 2️⃣ Crear y activar un entorno virtual
```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

### 3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4️⃣ Descargar los datos versionados con DVC
```bash
dvc pull
```

### 5️⃣ Ejecutar el pipeline completo 
Se usa el -f train para entrenar nuevamente y se genere el mlruns para el mlflow (el entrenamiento demora aprox 16 min)
```bash
dvc repro     #Para volver a entrenar: dvc repro -f train
```

### 6️⃣ Visualizar resultados en MLflow UI
```bash
mlflow ui &
```
Luego abre en tu navegador:
👉 http://127.0.0.1:5000


### 7️⃣ (Opcional) Revisar resultados sin reentrenar 
📂 Resultados de MLflow

Los experimentos originales (runs, métricas, modelos) están disponibles en el siguiente enlace de Google Drive:
👉 [Descargar mlruns.zip](https://drive.google.com/file/d/1Eq_RhXQdW9VtCVQeZyL126Fdh_qc-0wL/view?usp=drive_link)

Pasos para revisar los resultados sin reentrenar:
1. Descarga y extrae el ZIP dentro de la carpeta raíz del proyecto (`mlops_parcial/`)
2. Visualizar resultados en MLflow UI
```bash
mlflow ui &
```
