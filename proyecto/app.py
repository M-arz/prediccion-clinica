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
from waitress import serve   # ✅ Import necesario para ejecutar en Vercel

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
    step_val = 1 if v == "age" else round((max_val-min_val)/100, 2)
    rangos[v] = {"min": min_val, "max": max_val, "step": step_val}

# Rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/individual', methods=['GET', 'POST'])
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
            resultado = "Positivo para enfermedad" if pred == 1 else "Negativo para enfermedad"
        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return render_template(
        'individual.html',
        resultado=resultado,
        columnas=variables,
        nombres=nombres_variables,
        rangos=rangos
    )

@app.route('/lotes', methods=['GET', 'POST'])
def lotes():
    tabla = None
    metricas = None
    cm_img = None
    modelo_seleccionado = None

    if request.method == 'POST':
        try:
            archivo = request.files.get("dataset")
            modelo_seleccionado = request.form.get("modelo")

            if not archivo:
                flash("⚠️ Debes subir un archivo antes de predecir.", "error")
                return redirect(url_for('lotes'))

            if modelo_seleccionado not in models:
                flash("⚠️ Modelo no válido o no encontrado.", "error")
                return redirect(url_for('lotes'))

            # Leer archivo
            if archivo.filename.endswith('.csv'):
                df = pd.read_csv(archivo)
            elif archivo.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(archivo)
            else:
                flash("⚠️ Formato no soportado. Usa .csv o .xlsx", "error")
                return redirect(url_for('lotes'))

            # Validar columnas
            columnas_faltantes = [v for v in variables if v not in df.columns]
            if columnas_faltantes:
                flash(f"⚠️ Faltan columnas requeridas: {', '.join(columnas_faltantes)}", "error")
                return redirect(url_for('lotes'))

            # Escalado y predicción
            scaler = models[modelo_seleccionado]["scaler"]
            modelo = models[modelo_seleccionado]["modelo"]
            X = scaler.transform(df[variables])
            predicciones = modelo.predict(X)
            df["Predicción"] = ["Positivo" if p == 1 else "Negativo" for p in predicciones]

            # Verificar columna de etiquetas reales
            posibles_columnas_y = ["target", "real", "diagnóstico", "etiqueta", "label"]
            col_y = next((c for c in posibles_columnas_y if c in df.columns), None)

            if col_y:
                y_real = df[col_y]
                y_pred = predicciones

                acc = accuracy_score(y_real, y_pred)
                prec = precision_score(y_real, y_pred)
                rec = recall_score(y_real, y_pred)
                f1 = f1_score(y_real, y_pred)

                metricas = {
                    "accuracy": round(acc * 100, 2),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2)
                }

                # Matriz de confusión
                cm = confusion_matrix(y_real, y_pred)
                fig, ax = plt.subplots(figsize=(5, 5))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap='Blues')
                plt.title('Matriz de Confusión')
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                cm_img = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Guardar resultados
            os.makedirs("uploads", exist_ok=True)
            ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
            df.to_excel(ruta_salida, index=False)

            # Tabla de resultados
            tabla = df.head(50).to_html(classes="tabla-scroll", index=False, justify="center")

            flash("✅ Predicción por lotes completada correctamente.", "success")

        except Exception as e:
            flash(f"❌ Ocurrió un error durante la predicción: {str(e)}", "error")
            return redirect(url_for('lotes'))

    return render_template(
        'lotes.html',
        tabla=tabla,
        metricas=metricas,
        cm=cm_img
    )


@app.route('/descargar_resultados')
def descargar_resultados():
    ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
    if not os.path.exists(ruta_salida):
        flash("⚠️ No hay resultados disponibles para descargar.", "error")
        return redirect(url_for('lotes'))
    return send_file(ruta_salida, as_attachment=True)


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=8080)
