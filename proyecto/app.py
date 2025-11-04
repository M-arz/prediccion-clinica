#from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import pickle
import os
import io, base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from flask import render_template, request
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
app.secret_key = "clave_secreta"

# Cargar modelos
models = {
    "rna": {
        "modelo": pickle.load(open("models/modelo_rna.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador_rna.pkl", "rb"))
    },
    "logistica": {
        "modelo": pickle.load(open("models/modelo_logistica.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador.pkl", "rb"))
    }
}

# Variables y nombres en español
variables = [
    "AST (SGOT)", "ALT (SGPT)", "total_proteins", "direct_bilirubin",
    "total_bilirubin", "lymphocytes", "hemoglobin", "hematocrit",
    "age", "urea", "red_blood_cells", "monocytes",
    "white_blood_cells", "creatinine", "ALP (alkaline_phosphatase)"
]
nombres_variables = {
    "AST (SGOT)": "AST (SGOT)",
    "ALT (SGPT)": "ALT (SGPT)",
    "total_proteins": "Proteínas Totales",
    "direct_bilirubin": "Bilirrubina Directa",
    "total_bilirubin": "Bilirrubina Total",
    "lymphocytes": "Linfocitos",
    "hemoglobin": "Hemoglobina",
    "hematocrit": "Hematocrito",
    "age": "Edad",
    "urea": "Urea",
    "red_blood_cells": "Glóbulos Rojos",
    "monocytes": "Monocitos",
    "white_blood_cells": "Glóbulos Blancos",
    "creatinine": "Creatinina",
    "ALP (alkaline_phosphatase)": "Fosfatasa Alcalina (ALP)"
}

# Rangos basados en dataset (edad step=1)
data = pd.read_excel("DEMALE-HSJM_2025_data (1).xlsx")
rangos = {}
for v in variables:
    min_val = float(data[v].min())
    max_val = float(data[v].max())
    step_val = 1 if v == "age" else round((max_val-min_val)/100,2)
    rangos[v] = {"min": min_val, "max": max_val, "step": step_val}

# Rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/individual', methods=['GET','POST'])
def individual():
    resultado = None
    if request.method == 'POST':
        try:
            modelo_seleccionado = request.form.get("modelo")
            if modelo_seleccionado not in models:
                flash("Modelo no válido", "error")
                return redirect(url_for('individual'))

            datos = [float(request.form.get(v)) for v in variables]
            df = pd.DataFrame([datos], columns=variables)

            scaler = models[modelo_seleccionado]["scaler"]
            modelo = models[modelo_seleccionado]["modelo"]

            pred = modelo.predict(scaler.transform(df))[0]
            resultado = "Positivo para enfermedad" if pred==1 else "Negativo para enfermedad"
        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return render_template('individual.html', resultado=resultado, columnas=variables, nombres=nombres_variables, rangos=rangos)

@app.route('/lotes', methods=['GET', 'POST'])
def lotes():
    tabla = None
    metricas = None
    cm_img = None
    model_name = None
    modelo_seleccionado = None

    if request.method == 'POST':
        try:
            # archivo y modelo elegido
            archivo = request.files.get("dataset") or request.files.get("file")
            modelo_seleccionado = request.form.get("modelo") or request.form.get("model", "logistica")

            # validaciones
            if not archivo:
                flash("⚠️ Debes subir un archivo antes de predecir.", "error")
                return redirect(url_for('lotes'))

            if modelo_seleccionado not in models:
                flash("⚠️ Modelo no válido o no encontrado.", "error")
                return redirect(url_for('lotes'))

            # leer archivo
            if archivo.filename.endswith('.csv'):
                df = pd.read_csv(archivo)
            elif archivo.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(archivo)
            else:
                flash("⚠️ Formato no soportado. Usa .csv, .xls o .xlsx", "error")
                return redirect(url_for('lotes'))

            # comprobar columnas features
            missing_cols = [v for v in variables if v not in df.columns]
            if missing_cols:
                flash(f"⚠️ Faltan columnas requeridas: {', '.join(missing_cols)}", "error")
                return redirect(url_for('lotes'))

            # detectar columna target (si existe)
            posibles_target = ["target", "real", "diagnóstico", "diagnostico", "etiqueta", "label", "clase", "Clase", "diagnosis"]
            col_target = next((c for c in posibles_target if c in df.columns), None)

            # preprocesamiento mínimo: quitar filas con NA en features (opcional)
            df_proc = df.copy()
            df_proc = df_proc.dropna(subset=variables)  # evita errores al escalar

            # tomar X
            X = df_proc[variables]

            # cargar scaler y modelo desde tu dict `models`
            scaler = models[modelo_seleccionado]["scaler"]
            model = models[modelo_seleccionado]["modelo"]

            # escalar
            X_scaled = scaler.transform(X)

            # predecir (manejar sklearn y keras)
            y_pred_raw = None
            if hasattr(model, "predict_proba") and not hasattr(model, "predict_classes"):
                # sklearn clásico
                y_pred_raw = model.predict(X_scaled)
            else:
                # posible Keras (o modelos que devuelven probabilidades)
                try:
                    y_tmp = model.predict(X_scaled)
                    # si devuelve probabilidades (shape Nx1 o Nx2), convertir a etiquetas
                    if isinstance(y_tmp, np.ndarray):
                        if y_tmp.ndim == 2 and y_tmp.shape[1] > 1:
                            y_pred_raw = np.argmax(y_tmp, axis=1)
                        else:
                            # binario: prob -> 0/1
                            y_pred_raw = (y_tmp.ravel() > 0.5).astype(int)
                    else:
                        y_pred_raw = np.array(y_tmp)
                except Exception:
                    # fallback a predict directo
                    y_pred_raw = model.predict(X_scaled)

            # asegurar array 1D int
            y_pred = np.array(y_pred_raw).astype(int).ravel()

            # agregar predicción legible
            df_proc["Predicción_num"] = y_pred
            df_proc["Predicción"] = df_proc["Predicción_num"].apply(lambda x: "Positivo" if int(x) == 1 else "Negativo")

            # guardar para descarga (completo)
            os.makedirs("uploads", exist_ok=True)
            ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
            # unimos la columna de predicción al df original alineando por índice
            df_out = df.copy()
            df_out.loc[df_proc.index, "Predicción"] = df_proc["Predicción"].values
            df_out.to_excel(ruta_salida, index=False)

            # tabla preview (primeras 50 filas)
            tabla = df_out.head(50).to_html(classes="tabla-scroll", index=False, justify="center")

            # Si existe columna target -> calcular métricas y matriz de confusión
            if col_target:
                # obtener y_true y convertir a 0/1 si hace falta
                y_true_raw = df_proc[col_target].copy().reset_index(drop=True)

                # normalizar valores comunes a 0/1
                # si vienen como texto "Positivo"/"Negativo"
                if y_true_raw.dtype == object:
                    y_true = y_true_raw.str.strip().str.lower().replace({
                        "positivo": 1, "pos": 1, "p": 1,
                        "negativo": 0, "neg": 0, "n": 0,
                        "1": 1, "0": 0, "si": 1, "sí": 1, "s": 1
                    })
                else:
                    y_true = y_true_raw

                # intentar forzar tipo int
                try:
                    y_true = y_true.astype(int).values
                except Exception:
                    # si no se puede, intentar mapear etiquetas únicas a índices
                    unique_labels = list(pd.Series(y_true_raw).unique())
                    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
                    # mapear y_pred (si tus preds son 0/1 pero y_true original es e.g. 'positivo'), mapear y_pred a nombres posibles:
                    # si unique_labels están en texto como "Positivo"/"Negativo" intentamos mapear 1->'positivo', 0->'negativo'
                    if set([str(u).lower() for u in unique_labels]) >= {"positivo", "negativo"}:
                        # mapear 1->'positivo', 0->'negativo'
                        y_true = pd.Series(y_true_raw).str.strip().str.lower().replace({
                            "positivo": 1, "pos": 1, "p": 1,
                            "negativo": 0, "neg": 0, "n": 0
                        }).astype(int).values
                    else:
                        # fallback: convertir y_pred a las etiquetas detectadas por índice
                        # crear y_pred_labels en forma de strings comparables
                        # si y_pred es 0/1 y unique_labels len==2: mapear 0->unique_labels[0], 1->unique_labels[1]
                        if len(unique_labels) == 2:
                            mapping = {0: unique_labels[0], 1: unique_labels[1]}
                            y_pred_labels = [mapping.get(int(v), v) for v in y_pred]
                            y_true = [str(v) for v in y_true_raw]
                        else:
                            # no podemos calcular métricas fiables
                            y_true = None

                # si ya tenemos y_true y y_pred numéricos compatibles -> calcular métricas
                if y_true is not None:
                    # si y_true es string -> convertimos y_pred a labels comparables
                    if isinstance(y_true[0], (str,)) :
                        y_pred_labels = [str(x) for x in y_pred]
                        y_true_labels = [str(x) for x in y_true]
                    else:
                        y_pred_labels = y_pred
                        y_true_labels = y_true

                    # métricas
                    try:
                        acc = round(accuracy_score(y_true_labels, y_pred_labels) * 100, 2)
                        prec = round(precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0) * 100, 2)
                        rec = round(recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0) * 100, 2)
                        f1 = round(f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0) * 100, 2)
                        metricas = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
                    except Exception:
                        metricas = None

                    # matriz de confusión (si ambas listas no vacías)
                    try:
                        # si labels binarias 0/1 -> labels = [0,1]
                        labels_cm = None
                        if y_true is not None and np.unique(y_true).size <= 10:
                            labels_cm = sorted(list(np.unique(y_true)))
                        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels_cm)
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_cm)
                        fig, ax = plt.subplots(figsize=(5, 5))
                        disp.plot(ax=ax, cmap='Blues', colorbar=False)
                        plt.title('Matriz de Confusión')
                        plt.tight_layout()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        plt.close(fig)
                        buf.seek(0)
                        cm_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    except Exception:
                        cm_img = None

            flash("✅ Predicción por lotes completada correctamente.", "success")

        except Exception as e:
            flash(f"❌ Ocurrió un error durante la predicción: {str(e)}", "error")
            return redirect(url_for('lotes'))

    return render_template('lotes.html',
                           tabla=tabla,
                           metricas=metricas,
                           cm=cm_img,
                           model_name=("RNA" if modelo_seleccionado in ["rna", "nn", "red"] else "Regresión Logística"))


@app.route('/descargar_resultados')
def descargar_resultados():
    ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
    if not os.path.exists(ruta_salida):
        flash("⚠️ No hay resultados disponibles para descargar.", "error")
        return redirect(url_for('lotes'))
    return send_file(ruta_salida, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)